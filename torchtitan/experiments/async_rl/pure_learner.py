# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Panocular AI.
#
# PureLearnerReplica: the shared base for the decoupled-generation strategies
# (async_inference, heloco_async_inference). A pure learner runs NO local
# generation -- one trainer GPU, all training rollouts arrive from a pool of
# REMOTE generator workers through a queue -- which is prime-rl's
# decoupled-generation shape (arXiv:2505.07291): inference lives on separate
# machines from training. Colocating generation with the trainer would just
# be the single-node "local" baseline with an async buffer, so a pure learner
# spawns no vLLM at all.
#
# This base owns everything the two decoupled strategies share:
#   - a generator-less actor spawn (_spawn_trainer_only): trainer mesh only,
#     no GeneratorRouter, no TorchStore;
#   - the rollout consumer (_consume_remote_rollouts) + bounded buffer +
#     fail-fast liveness check (_buffer_get_checked) + staleness-bounded batch
#     assembly (_collect_and_build);
#   - a no-op _refresh_generators (nothing local to refresh) and a
#     _validate_fixed that skips trainer-side validation (a pure learner has
#     no generator; the learning curve is the per-window mean reward of the
#     remote workers' rollouts, which the mixin's train() already logs as it
#     consumes them).
#
# Subclasses supply: the rollout SOURCE (_next_rollout_batch), the staleness
# REFERENCE (_staleness_reference), the coordination boundary (_window_sync),
# and the trainer actor class + any client/relay wiring in setup_async.

import asyncio
import logging
from dataclasses import dataclass

from torchtitan.experiments.async_rl.controller import RLControllerMixin
from torchtitan.experiments.async_rl.rl_trainer import RLTrainer, setup_mesh_elastic_env

logger = logging.getLogger(__name__)


class PureLearnerReplica(RLControllerMixin, RLTrainer):
    """Base for a trainer that runs no local generation and consumes all
    training rollouts from a remote generator pool via a queue. Subclasses
    implement ``_next_rollout_batch`` (the queue source), ``_staleness_reference``
    (the version this replica compares consumed rollouts against), and
    ``_window_sync`` (their coordination), and call ``_spawn_trainer_only`` from
    their ``setup_async``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLTrainer.Config):
        sync_every: int = 4
        """Local training steps per logged window."""
        num_outer_steps: int = 0
        """Fixed number of windows to run (step-bound). Mutually exclusive with
        train_seconds: set exactly one."""
        train_seconds: float = 0.0
        """Wall-clock training budget (time-bound). Mutually exclusive with
        num_outer_steps: set exactly one."""
        replica_id: int = 0
        """Identifier for logging."""
        num_generators: int = 0
        """Always 0: a pure learner runs no local vLLM (generation is fully
        decoupled onto the remote worker pool). Kept as an explicit field so
        ``async_rl/train.py`` provisions only the trainer GPU and spawns no
        generator meshes."""
        max_staleness: int = 4
        """Maximum version lag (in the subclass's staleness-reference space) of
        a consumed rollout batch; staler batches are dropped at consume time
        (prime-rl's async_level, enforced by dropping)."""
        buffer_groups: int = 0
        """Rollout-buffer capacity in groups; a full buffer back-pressures the
        consumer. 0 = two steps' worth
        (2 * async_loop.num_prompts_per_train_step)."""
        queue_poll_interval_s: float = 0.5
        """Seconds to wait before re-polling the rollout source when it yields
        nothing (only relevant for a non-blocking source, e.g. a remote hub)."""
        rollout_stall_timeout_s: float = 900.0
        """Fail the run if no rollout arrives for this many seconds while the
        trainer is waiting on the buffer. The remote generator pool is this
        trainer's ONLY rollout source, and a worker crash is invisible from
        here (the queue/hub stays reachable, just empty forever) -- without
        this bound the trainer polls silently for the rest of the run. Must
        comfortably exceed a worker cold start (vLLM engine init + first
        checkpoint pull + first round, ~3-5 min). 0 disables the bound."""

        def __post_init__(self):
            # A pure learner runs no local generator, so the base Controller
            # config's generator-centric validations (num_generators>=1,
            # generator checkpoint/hot_swap/cudagraph/batch_invariant-generator)
            # don't apply. Do the trainer-side essentials the base would
            # otherwise do: the SP-divisibility check and mirroring the batcher
            # width into trainer.training.seq_len for the model build.
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                seq_len = self.async_loop.batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be "
                        f"divisible by sequence parallel degree ({sp_degree})."
                    )
            self.trainer.training.seq_len = self.async_loop.batcher.batch.seq_len

            if self.sync_every < 1:
                raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
            if (self.num_outer_steps > 0) == (self.train_seconds > 0):
                raise ValueError(
                    "Set exactly one of num_outer_steps (step-bound run) or "
                    "train_seconds (time-bound run); got num_outer_steps="
                    f"{self.num_outer_steps}, train_seconds={self.train_seconds}"
                )
            if self.max_staleness < 1:
                raise ValueError(
                    f"max_staleness must be >= 1, got {self.max_staleness}"
                )
            if self.num_generators != 0:
                raise ValueError(
                    "a pure learner runs no local generators (generation is "
                    "fully decoupled onto the remote worker pool); "
                    f"num_generators must be 0, got {self.num_generators}"
                )

    # ------------------------------------------------------------------ #
    # Generator-less actor spawn.
    # ------------------------------------------------------------------ #

    async def _spawn_trainer_only(
        self, *, trainer_mesh, generator_meshes, policy_trainer_cls
    ):
        """Spawn ONLY the trainer actor (one learner GPU, ``policy_trainer_cls``)
        -- no generator mesh, no GeneratorRouter, no TorchStore. ``generator_meshes``
        must be empty (train.py spawns none when num_generators=0); it's accepted
        only to match the launcher's setup_async signature. ``generator_router``
        stays None (from RLTrainer.__init__), which close() tolerates."""
        cfg = self.config
        if generator_meshes:
            raise ValueError(
                "a pure learner spawns no generators, but "
                f"{len(generator_meshes)} generator mesh(es) were provisioned; "
                "set num_generators=0"
            )
        trainer_parallelism = cfg.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )
        self._proc_meshes = [trainer_mesh]
        await setup_mesh_elastic_env(trainer_mesh)
        self.trainer = trainer_mesh.spawn(
            "trainer",
            policy_trainer_cls,
            cfg.trainer,
            model_spec=cfg.model_spec,
            hf_assets_path=cfg.hf_assets_path,
            generator_dtype=cfg.generator.model_dtype,
            compile_config=cfg.compile,
            output_dir=cfg.dump_folder,
        )

    # ------------------------------------------------------------------ #
    # Rollout source + staleness reference (subclass-provided).
    # ------------------------------------------------------------------ #

    async def _next_rollout_batch(self):
        """Return one ``(worker_id, version, groups)`` batch from the remote
        generator pool, or ``None`` if none is available right now (the base
        consumer then waits ``queue_poll_interval_s`` and retries)."""
        raise NotImplementedError

    def _staleness_reference(self) -> int:
        """The current version a consumed rollout's stamped version is compared
        against for the ``max_staleness`` bound (e.g. the latest published
        checkpoint version, or the shared parameter-server revision)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Consumer side.
    # ------------------------------------------------------------------ #

    async def _consume_remote_rollouts(self) -> None:
        """Drain the remote generator pool into the local buffer, tagging each
        group with the batch's generation version. Runs until cancelled."""
        while True:
            batch = await self._next_rollout_batch()
            if batch is None:
                await asyncio.sleep(self.config.queue_poll_interval_s)
                continue
            _worker_id, version, groups = batch
            for group in groups:
                await self._buffer.put((group, version))

    async def _buffer_get_checked(self):
        """buffer.get that fails fast when the feed is gone, instead of
        waiting forever (there are no local producers):
          - the single remote-consumer task died or exited; or
          - nothing has arrived for rollout_stall_timeout_s. A crashed remote
            WORKER is invisible from here -- the queue/hub stays reachable,
            just empty forever -- so an empty buffer past any plausible worker
            cold start means the generator pool is dead."""
        timeout = self.config.rollout_stall_timeout_s
        deadline = asyncio.get_running_loop().time() + timeout if timeout > 0 else None
        get_task = asyncio.ensure_future(self._buffer.get())
        try:
            while not get_task.done():
                consumer = self._remote_consumer_task
                if consumer.done():
                    if not consumer.cancelled() and consumer.exception() is not None:
                        raise RuntimeError(
                            "remote rollout consumer died"
                        ) from consumer.exception()
                    raise RuntimeError(
                        "remote rollout consumer exited while the trainer "
                        "still needs data"
                    )
                remaining = None
                if deadline is not None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise RuntimeError(
                            f"no rollout arrived for {timeout:.0f}s -- the "
                            "remote generator pool is likely dead (worker "
                            "crash?) or unable to reach this trainer's queue; "
                            "check the worker logs. (rollout_stall_timeout_s "
                            "bounds this wait; 0 disables it.)"
                        )
                await asyncio.wait(
                    {get_task, consumer},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=remaining,
                )
            return get_task.result()
        finally:
            if not get_task.done():
                get_task.cancel()

    async def _collect_and_build(self, step: int):
        """Assemble one training step's batch from the buffer, dropping groups
        staler than max_staleness (in the subclass's staleness-reference
        space). This overrides the mixin's generator-fanout collector, so
        _run_window's always-on pipeline overlaps this pop+build+batch with
        training instead of driving local generation.

        Uses the upstream pipeline (TrainingSampleBuilder -> Batcher) provided
        by the RLTrainer base: each surviving remote group is built into a
        training-sample group and fed to the packing batcher until it yields a
        batch. Returns ``(packed_batch, rollout_groups)``."""
        await self.trainer.sync_log_step.call(step)
        rollout_groups = []
        packed = None
        while packed is None:
            group, version = await self._buffer_get_checked()
            if self._staleness_reference() - version > self.config.max_staleness:
                self._num_dropped += 1
                continue
            rollout_groups.append(group)
            training_sample_group = self._training_sample_builder.build_from_group(
                rollout_group=group
            )
            packed = self._batcher.add_training_samples(
                training_sample_group=training_sample_group
            )
        return packed, rollout_groups

    def _batch_staleness(
        self, pre_optim_policy_version: int, min_policy_versions
    ) -> int:
        """A pure learner's samples are stamped with the relay/hub checkpoint
        version their remote generator loaded -- a SHARED counter that advances
        per publish, not per local optim step -- so the mixin's default
        (local ``policy_version`` minus sample version) would subtract across
        two unrelated spaces and grow without bound. Measure against the same
        reference the ``max_staleness`` consume gate uses instead: the logged
        number is then directly comparable to ``config.max_staleness``."""
        del pre_optim_policy_version  # local optim-step space; not comparable
        return (
            self._staleness_reference() - min(min_policy_versions)
            if min_policy_versions
            else 0
        )

    async def _refresh_generators(self) -> None:
        """No-op: there is no local generator to refresh (the remote workers
        pull weights from the relay tier, out of band from the train step)."""

    # ------------------------------------------------------------------ #
    # No trainer-side validation (a pure learner runs no generator).
    # ------------------------------------------------------------------ #

    async def _validate_fixed(self, step: int):
        """Validation needs greedy generation, which a pure learner can't do
        (no vLLM here). Skip it: the learning curve is instead the per-window
        mean reward of the remote workers' rollouts, which train() already logs
        as it consumes them (it tracks a greedy held-out eval closely -- corr
        ~0.77, gap ~0.03 across the overnight runs). Returns no metrics so the
        mixin's train() logs val=nan without touching a generator."""
        return []

    # ------------------------------------------------------------------ #
    # Controller-loop hooks shared by both strategies.
    # ------------------------------------------------------------------ #

    async def _train_setup(self) -> None:
        cfg = self.config
        self._buffer = asyncio.Queue(
            maxsize=cfg.buffer_groups or 2 * cfg.async_loop.num_prompts_per_train_step
        )
        self._num_dropped = 0
        self._remote_consumer_task = asyncio.create_task(
            self._consume_remote_rollouts(), name="remote_rollout_consumer"
        )

    async def _train_cleanup(self) -> None:
        task = getattr(self, "_remote_consumer_task", None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
