# Copyright (c) Panocular AI.
#
# Shared controller machinery for the async_rl coordination strategies.
#
# RLControllerMixin is everything the coordinators (heloco / diloco /
# async_inference) have in common, in three layers:
#
#   1. The LlamaRL-style generation/training split (arXiv:2505.24034):
#      _collect_and_build (generator-mesh + CPU work) vs _train_on
#      (trainer-mesh work), always run with one-step-ahead overlap. The
#      overlap is safe with an importance-sampling-corrected loss (the base
#      GRPOLoss's token-level decoupled IS) since it already tolerates the
#      resulting bounded generator/trainer policy staleness -- pipelining
#      doesn't add new staleness-correction machinery, it just lets the
#      existing loss do the job it was built for.
#   2. The window runner (_run_window): one window of sync_every inner steps,
#      always pipelined (overlap collection of step h+1 with training of
#      step h), with divergence detection.
#   3. The outer training loop (train): pre/post validation on a fixed eval
#      set, step- or time-bound windows, and the standard per-window log line
#      that ___benchmark/analyze.py parses. Coordinators plug their sync into
#      the hooks (_window_sync and friends) instead of copying the loop.

import asyncio
import logging
import math
import time

logger = logging.getLogger(__name__)


class RLControllerMixin:
    """The shared controller loop for RLTrainer-based coordinators.

    Requires the host class to be (or subclass) torchtitan's RLTrainer, with a
    `Config` exposing `sync_every: int`, `num_outer_steps: int`,
    `train_seconds: float`, and `replica_id: int`.
    Provides `__init__` (initializes `_policy_version`) and `train`; subclasses
    override the `_window_sync` / `_train_*` hooks for their coordination work.
    """

    def __init__(self, config):
        super().__init__(config)
        self._policy_version = 0

    # ------------------------------------------------------------------ #
    # Generation/training split (the pipelining building blocks).
    # ------------------------------------------------------------------ #

    async def _collect_and_build(self, step: int):
        """Rollout collection + episode building (generator-mesh + CPU work
        only; no trainer-mesh calls). Launched one step ahead of the previous
        step's trainer-mesh work by _run_window's pipeline."""
        await self.trainer.sync_log_step.call(step)
        await self.generator_router.fanout("sync_log_step", step)

        num_groups = self.config.num_groups_per_rollout_batch
        num_tokens_target = self.batcher.num_tokens_target(self.trainer_dp_degree)
        rollout_groups = []
        collected_tokens = 0
        group_offset = 0
        while collected_tokens < num_tokens_target:
            groups, _ = await self._collect_rollouts(
                is_validation=False,
                num_groups=num_groups,
                group_size=self.config.group_size,
                sampling=self._sampling,
                step=step,
                group_offset=group_offset,
            )
            rollout_groups.extend(groups)
            collected_tokens += sum(
                len(r.turns[-1].prompt_token_ids)
                + len(r.turns[-1].completion_token_ids)
                - 1
                for g in groups
                for r in g.rollouts
                if r.turns
            )
            group_offset += num_groups

        episodes, _ = self._build_episodes(rollout_groups)
        microbatches, num_global_valid_tokens, _ = self.batcher.batch(
            episodes, dp_degree=self.trainer_dp_degree
        )
        return rollout_groups, episodes, microbatches, num_global_valid_tokens

    async def _train_on(
        self, rollout_groups, episodes, microbatches, num_global_valid_tokens
    ) -> dict:
        """forward_backward loop + optim_step + generator weight refresh
        (trainer-mesh work)."""
        pre_optim_policy_version = self._policy_version

        last_loss = 0.0
        for microbatch in microbatches:
            mb = self._get_rank_0_value(
                await self.trainer.forward_backward.call(
                    microbatch, num_global_valid_tokens
                )
            )
            last_loss = mb.get("loss/mean", mb.get("loss", last_loss))
        optim_output = self._get_rank_0_value(await self.trainer.optim_step.call())
        self._policy_version = optim_output.policy_version
        await self._refresh_generators()

        rewards = [
            r.reward for g in rollout_groups for r in g.rollouts if r.reward is not None
        ]
        reward_mean = sum(rewards) / len(rewards) if rewards else float("nan")
        # Staleness (in steps) between the trainer policy that produced these
        # rollouts' behavior logprobs and the trainer policy right before this
        # optim_step consumed them. Always >= 0; grows if generation lags
        # training (expected under the always-on one-step pipeline; an
        # IS-corrected loss is designed to absorb this bounded staleness).
        ep_versions = [ep.policy_version for ep in episodes]
        staleness = pre_optim_policy_version - min(ep_versions) if ep_versions else 0
        return {
            "loss": last_loss,
            "reward_mean": reward_mean,
            "policy_version": self._policy_version,
            "num_rollouts": len(rewards),
            "staleness": staleness,
        }

    async def _refresh_generators(self) -> None:
        """Bring this replica's generators up to the just-updated local policy:
        stage the trainer weights, then drain-and-pull every engine. Overridden
        by controllers that decouple the pull from the train step
        (AsyncInferenceReplica stages only; its producers pull per-round)."""
        await self.trainer.push_model_state_dict.call()
        await self.generator_router.pull_model_state_dict(
            policy_version=self._policy_version
        )

    # ------------------------------------------------------------------ #
    # Window runner.
    # ------------------------------------------------------------------ #

    async def _run_window(self, sync_every: int, start_step: int):
        """Run one window of ``sync_every`` inner steps, always
        pipelined: overlap ``_collect_and_build(h+1)`` with ``_train_on(h)``
        (LlamaRL-style one-step-ahead generation/training overlap). Collection
        is never launched on the window's last inner step, so the pipeline
        can't straddle the caller's outer sync boundary.

        Returns ``(window_rewards, last, global_step, diverged)``. The caller
        (train) applies the coordinator's post-window sync via _window_sync
        and stops on ``diverged``.
        """
        window_rewards: list[float] = []
        last = None
        global_step = start_step

        global_step += 1
        pending = asyncio.create_task(self._collect_and_build(global_step))
        try:
            for _h in range(sync_every):
                iter_t0 = time.perf_counter()
                rollout_groups, episodes, microbatches, num_valid = await pending
                pending = None
                if _h != sync_every - 1:
                    global_step += 1
                    pending = asyncio.create_task(self._collect_and_build(global_step))
                last = await self._train_on(
                    rollout_groups, episodes, microbatches, num_valid
                )
                window_rewards.append(last["reward_mean"])
                if not math.isfinite(float(last["loss"])):
                    return window_rewards, last, global_step, True
                await self._after_inner_step(iter_t0)
        finally:
            # Divergence return or a raising _train_on must not leak the
            # in-flight collection task.
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        return window_rewards, last, global_step, False

    # ------------------------------------------------------------------ #
    # Validation on a fixed eval set.
    # ------------------------------------------------------------------ #

    def _reset_validation_set(self) -> None:
        """Rebuild the validation dataset so every pass scores the same fixed
        prompts. Without this, the iterator draws fresh random samples each
        call, making the validation curve reflect prompt difficulty rather
        than learning progress."""
        self._rollouter._validation_dataset = (
            self.config.rollouter.validation_dataset.build()
        )

    async def _validate_fixed(self, step: int):
        """validate() on the fixed eval set. Greedy reward on this set is the
        metric that should trend up; train rollout reward is noisy/exploratory."""
        self._reset_validation_set()
        return await self.validate(step=step)

    def _aggregate_validation(self, metrics) -> dict:
        from torchtitan.experiments.rl.observability import metrics as m

        return m.MetricsProcessor._aggregate_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Coordination hooks (all no-ops for a standalone worker).
    # ------------------------------------------------------------------ #

    async def _train_setup(self) -> None:
        """Runs once, after pre-training validation and before the first
        window (e.g. start rollout producers)."""

    async def _after_inner_step(self, iter_t0: float) -> None:
        """Runs after each inner step of a window (not after a divergence
        stop). ``iter_t0`` is the step's start time (time.perf_counter) —
        subclass-side per-step instrumentation or pacing (e.g. the
        benchmark's slow-worker emulation) hooks in here."""

    async def _window_sync(self, t0: float) -> str | None:
        """The coordinator's window-boundary work, between the window's last
        optim_step and its validation pass (e.g. HeLoCo's push/pull). ``t0``
        is the window's start time (time.perf_counter). May adjust
        ``self._sync_every`` for the next window (DyLU). Returns an extra
        detail line to log after the standard window line, or None."""
        return None

    async def _after_validation(self) -> None:
        """Runs after each window's validation pass (e.g. resume producers)."""

    async def _train_teardown(self) -> None:
        """Runs once, after the last window and before post-training
        validation (e.g. quiesce producers so post-val measures the final
        policy)."""

    async def _train_cleanup(self) -> None:
        """Always runs on the way out of train(), including on divergence or
        error (e.g. cancel producer tasks)."""

    # ------------------------------------------------------------------ #
    # The outer training loop.
    # ------------------------------------------------------------------ #

    async def train(self):
        """Run windows until the step or time bound, validating and emitting
        the standard per-window log line (the format ___benchmark/analyze.py's
        WINDOW_RE parses -- change them together) after each."""
        cfg = self.config
        rid = cfg.replica_id
        self._sync_every = cfg.sync_every

        logger.info("[replica %d] pre-training validation", rid)
        pre_acc = self._aggregate_validation(await self._validate_fixed(0))

        try:
            await self._train_setup()

            global_step = 0
            outer = 0
            deadline = (
                time.monotonic() + cfg.train_seconds
                if cfg.train_seconds > 0
                else float("inf")
            )
            while time.monotonic() < deadline and (
                cfg.num_outer_steps == 0 or outer < cfg.num_outer_steps
            ):
                t0 = time.perf_counter()
                window_rewards, last, global_step, diverged = await self._run_window(
                    self._sync_every, global_step
                )
                if diverged:
                    logger.error("[replica %d] loss diverged; stopping", rid)
                    return

                extra = await self._window_sync(t0)
                val_agg = self._aggregate_validation(
                    await self._validate_fixed(global_step)
                )
                val_reward = float(val_agg.get("validation_reward/_mean", float("nan")))
                await self._after_validation()

                reward_mean = sum(window_rewards) / len(window_rewards)
                logger.info(
                    "[replica %d] window %2d | loss %+.4f | reward %+.3f | "
                    "val %+.3f | rollouts %d | sync_every=%d | staleness=%d | %.1fs",
                    rid,
                    outer,
                    last["loss"],
                    reward_mean,
                    val_reward,
                    last["num_rollouts"],
                    self._sync_every,
                    last.get("staleness", 0),
                    time.perf_counter() - t0,
                )
                if extra:
                    logger.info("[replica %d] %s", rid, extra)
                outer += 1

            await self._train_teardown()
            logger.info("[replica %d] post-training validation", rid)
            post_acc = self._aggregate_validation(
                await self._validate_fixed(global_step)
            )
            logger.info("[replica %d] pre=%s -> post=%s", rid, pre_acc, post_acc)
        finally:
            await self._train_cleanup()
