# Copyright (c) Panocular AI.
#
# Controller for synchronous DiLoCo RL.
#
# Unlike HeLoCoRLReplica (which drives the parameter-server sync manually at
# the window boundary), this controller runs a flat per-step RL loop: the
# DiLoCo sync happens automatically inside ``optim_step``, via the hook
# registered in DiLoCoManagerTrainer.setup_diloco, synchronized across
# replicas through the torchft Manager/Lighthouse quorum.

import logging
from dataclasses import dataclass

from torchtitan.experiments.async_rl.controller import RLControllerMixin
from torchtitan.experiments.async_rl.diloco.actors import DiLoCoManagerTrainer

from torchtitan.experiments.rl import trainer as _rl_trainer_mod
from torchtitan.experiments.rl.trainer import RLTrainer

logger = logging.getLogger(__name__)


class DiLoCoRLReplica(RLControllerMixin, RLTrainer):
    """A single synchronous-DiLoCo RL replica (worker).

    N of these, coordinated by a torchft Lighthouse, run identical flat RL
    loops and sync weights every ``sync_every`` steps via stock DiLoCo.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLTrainer.Config):
        # Launch plumbing -- filled in by train.py from environment variables.
        lighthouse_address: str = ""
        """torchft Lighthouse address (usually from $DILOCO_LIGHTHOUSE_ADDR)."""
        replica_id: int = 0
        """This worker's index (usually from $DILOCO_REPLICA_ID)."""
        num_replicas: int = 2
        """Total workers in the quorum (= min_replica_size for the synchronous barrier)."""

        # DiLoCo hyperparameters.
        sync_every: int = 4
        """Local RL steps between DiLoCo syncs (the inner-loop length H)."""
        num_outer_steps: int = 0
        """Fixed number of sync windows to run (step-bound run). Mutually
        exclusive with train_seconds: set exactly one."""
        train_seconds: float = 0.0
        """Wall-clock training budget in seconds (time-bound run): keep running
        sync_every-step windows until the deadline (checked at window
        boundaries, so a run overshoots by at most one window). Slow or
        heterogeneous workers naturally complete fewer steps in the same time.
        Mutually exclusive with num_outer_steps: set exactly one.
        Replicas share the budget so they stop within one window of each
        other; torchft's fault-tolerant quorum absorbs the skew."""
        outer_lr: float = 0.7
        """Outer Nesterov-SGD learning rate (stock DiLoCo default)."""
        outer_momentum: float = 0.9
        """Outer Nesterov-SGD momentum (stock DiLoCo default)."""
        num_generators: int = 1
        """Number of independent vLLM engines this replica spawns (each of
        generator.parallelism's world size, on its own GPU slice out of the
        replica's CUDA_VISIBLE_DEVICES pool). Requests are spread round-robin
        by the GeneratorRouter; weight refreshes fan out to all engines. The
        replica needs trainer_ws + num_generators * generator_ws visible GPUs
        (launch_diloco.sh's GPUS_PER_REPLICA)."""

        def __post_init__(self):
            RLTrainer.Config.__post_init__(self)
            if self.sync_every < 1:
                raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
            if (self.num_outer_steps > 0) == (self.train_seconds > 0):
                raise ValueError(
                    "Set exactly one of num_outer_steps (step-bound run) or "
                    "train_seconds (time-bound run); got num_outer_steps="
                    f"{self.num_outer_steps}, train_seconds={self.train_seconds}"
                )

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn DiLoCoManagerTrainer (not the base PolicyTrainer) and enter the
        DiLoCo context so subsequent ``optim_step`` calls sync automatically."""
        cfg = self.config
        if not cfg.lighthouse_address:
            raise ValueError(
                "lighthouse_address is required (set $DILOCO_LIGHTHOUSE_ADDR)"
            )

        orig = _rl_trainer_mod.PolicyTrainer
        _rl_trainer_mod.PolicyTrainer = DiLoCoManagerTrainer
        try:
            await super().setup_async(
                trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
            )
        finally:
            _rl_trainer_mod.PolicyTrainer = orig

        await self.trainer.setup_diloco.call(
            lighthouse_address=cfg.lighthouse_address,
            replica_id=cfg.replica_id,
            num_replicas=cfg.num_replicas,
            sync_every=cfg.sync_every,
            outer_lr=cfg.outer_lr,
            outer_momentum=cfg.outer_momentum,
        )
        logger.info(
            "[replica %d] DiLoCo trainer ready (lighthouse=%s)",
            cfg.replica_id,
            cfg.lighthouse_address,
        )

    async def _window_sync(self, t0: float) -> str:
        """The DiLoCo sync for this window already happened inside the
        window's last optim_step (step % sync_every == 0, via the
        inner-optimizer post-step hook); just report the Manager's view as an
        extra detail line (kept off the primary window line so analyze.py's
        WINDOW_RE -- shared verbatim across all controllers -- still matches)."""
        info = self._get_rank_0_value(await self.trainer.diloco_step_info.call())
        return (
            f"diloco_step={info['current_step']} "
            f"participants={info['num_participants']}"
        )

    async def close(self):
        try:
            await self.trainer.close_diloco.call()
        except Exception:
            logger.exception(
                "[replica %d] error closing DiLoCo", self.config.replica_id
            )
        await super().close()
