# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Panocular AI.
#
# decentralized_rl's synchronous RL orchestration base.
#
# Re-homed from the (now removed) ``torchtitan/experiments/rl/trainer.py`` and
# re-based onto the current ``rl/`` framework. Upstream replaced that
# synchronous ``RLTrainer`` with the async off-policy ``Controller``
# (``rl/controller.py``); decentralized_rl's coordination strategies (HeLoCo, DiLoCo,
# pure-learner) instead need a SYNCHRONOUS, windowed outer loop for their
# federated / local-SGD weight averaging, so this module keeps that
# orchestration model while consuming the new ``rl/`` components.
#
# ``RLTrainer`` subclasses ``Controller`` to reuse its actor setup
# (``setup_async``), validation, generator-router generate seam, and teardown,
# but replaces the async 4-loop ``run()`` with synchronous single-step
# primitives (``_collect_training_batch`` / ``_apply_training_batch``) that the
# windowed outer loop in ``RLControllerMixin`` drives.
#
# Trainer-actor injection seam: the inherited ``Controller.setup_async`` spawns
# the trainer actor by resolving the name ``PolicyTrainer`` from the
# ``rl.controller`` module globals at spawn time. The HeLoCo/DiLoCo replicas
# therefore temporarily rebind ``rl.controller.PolicyTrainer`` (their
# monkeypatch seam) around ``super().setup_async(...)`` to spawn a custom
# trainer actor subclass.

import itertools
import logging
import os
import socket
from dataclasses import dataclass

from monarch.actor import ProcMesh
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.controller_metrics import compute_rollout_metrics

# Re-exported so decentralized_rl's config_registry keeps a single import site for the
# loss it used to get from ``rl.trainer``. The synchronous framework's GRPO
# loss now lives in the shared ``rl/losses`` package.
from torchtitan.experiments.rl.losses import GRPOLoss  # noqa: F401
from torchtitan.experiments.rl.rollout import RolloutGroup

logger = logging.getLogger(__name__)

# Below the OS ephemeral range (32768+), so dynamic allocations by unrelated
# processes can never take these; distinct from the launchers' 29500+ rdzv
# ports and the coordinator servers' 295xx/87xx defaults.
_ELASTIC_PORT_BASE = 29800
_mesh_counter = itertools.count()


async def setup_mesh_elastic_env(mesh: ProcMesh) -> None:
    """``setup_torch_elastic_env_async`` with a deterministic, host-unique port.

    Monarch's default probes for a free MASTER_PORT (bind-0 then close; rank
    0's TCPStore binds it for real later) — a check-then-use race: sibling
    replicas on one host probing concurrently can be handed the SAME port,
    and the loser dies with EADDRINUSE deep in vLLM/torch.distributed init.

    Launchers partition a host's replicas by disjoint, contiguous
    CUDA_VISIBLE_DEVICES ranges, and a replica spawns at most one mesh per
    GPU it owns, so (first visible device + this process's mesh counter) is
    unique across every mesh on the host. Without CUDA_VISIBLE_DEVICES there
    is no partition (sole tenant / remote host meshes) — keep Monarch's pick.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        await setup_torch_elastic_env_async(mesh)
        return
    port = _ELASTIC_PORT_BASE + int(visible.split(",")[0]) + next(_mesh_counter)
    await setup_torch_elastic_env_async(
        mesh, master_addr=socket.gethostname(), master_port=port
    )


class RLTrainer(Controller):
    """Synchronous RL orchestration base for the decentralized_rl coordinators.

    Reuses ``Controller.__init__`` (metrics/renderer/sampling/rollouter/recorder),
    ``Controller.setup_async`` (actor spawn + initial weight sync),
    ``Controller.validate``, and ``Controller.close``. Adds synchronous
    single-step primitives the windowed ``RLControllerMixin`` loop drives; the
    async ``Controller.run`` is intentionally unused by decentralized_rl.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Controller.Config):
        """decentralized_rl replicas subclass this; it adds no fields of its own."""

    def _build_sync_pipeline(self) -> None:
        """Build the rollout→sample→batch pipeline components used by the
        synchronous loop. Mirrors what ``Controller.run`` builds inline for the
        async loops, but stores them on ``self`` so the windowed loop can call
        them one step at a time. Must run after ``setup_async`` (needs
        ``self.trainer_dp_degree``)."""
        async_loop = self.config.async_loop
        self._training_sample_builder = async_loop.training_sample_builder.build()
        self._batcher = async_loop.batcher.build(
            num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
            dp_degree=self.trainer_dp_degree,
            pad_id=self.renderer._tokenizer.eos_token_id,
        )
        self._generate_fn = self._make_generate_fn(metrics_prefix="generator")
        self._group_counter = itertools.count()

    async def _rollout_one_group(self, *, sampling, step: int) -> RolloutGroup:
        """Generate + score one rollout group synchronously (the synchronous
        analogue of ``Controller._rollout_loop``'s body). Records the raw group
        for inspection before any downstream drop."""
        async_loop = self.config.async_loop
        group = await self._rollouter.run_group_rollouts(
            generate_fn=self._generate_fn,
            sample=self._rollouter.get_training_sample(),
            group_id=next(self._group_counter),
            group_size=async_loop.num_samples_per_prompt,
            sampling=sampling,
            renderer=self.renderer,
        )
        group.metrics = compute_rollout_metrics(
            prefix="rollout", rollouts=group.rollouts
        )
        self.rollout_recorder.record(is_validation=False, rollout_groups=[group])
        return group

    async def _collect_training_batch(self, step: int):
        """Collect rollout groups and pack exactly one ``TrainingBatch``.

        Feeds each group through ``TrainingSampleBuilder`` and the packing
        ``Batcher`` (main's pipeline) until the batcher yields a batch — that
        happens once ``num_prompts_per_train_step`` trainable groups have
        accumulated. Returns ``(packed_batch, rollout_groups)``; the groups are
        kept for the reward metric in the per-window log line.
        """
        rollout_groups: list[RolloutGroup] = []
        packed = None
        while packed is None:
            group = await self._rollout_one_group(sampling=self._sampling, step=step)
            rollout_groups.append(group)
            training_sample_group = self._training_sample_builder.build_from_group(
                rollout_group=group
            )
            packed = self._batcher.add_training_samples(
                training_sample_group=training_sample_group
            )
        return packed, rollout_groups

    async def _apply_training_batch(self, packed):
        """Run fwd/bwd on every microbatch, then the optimizer step. Returns the
        rank-0 optim result (``.policy_version``, ``.metrics``) plus the last
        microbatch's mean loss for the divergence guard / log line."""
        last_loss = 0.0
        for microbatch in packed.microbatches:
            mb = self._get_rank_0_value(
                await self.trainer.forward_backward.call(
                    microbatch, packed.num_global_valid_tokens
                )
            )
            last_loss = mb.get("loss/mean", mb.get("loss", last_loss))
        optim_output = self._get_rank_0_value(await self.trainer.optim_step.call())
        return optim_output, last_loss
