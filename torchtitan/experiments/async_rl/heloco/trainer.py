# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Panocular AI.
#
# Controller for HeLoCo RL (asynchronous DiLoCo through a parameter server).

import logging
import time
from dataclasses import dataclass

import torch

from torchtitan.experiments.async_rl.controller import RLControllerMixin
from torchtitan.experiments.async_rl.heloco.actors import HeLoCoPolicyTrainer
from torchtitan.experiments.async_rl.heloco.client import HeLoCoRLClient
from torchtitan.experiments.async_rl.heloco.server import param_metadata

from torchtitan.experiments.async_rl.rl_trainer import RLTrainer
from torchtitan.experiments.rl import controller as _rl_controller_mod

logger = logging.getLogger(__name__)


class HeLoCoRLReplica(RLControllerMixin, RLTrainer):
    """A single HeLoCo RL replica (worker).

    Subclasses the base RLTrainer to reuse its actor setup, rollout
    collection, episode/advantage building, batching, forward_backward,
    optim_step, and weight sync; the mixin's train loop runs windows of H
    local RL steps, and _window_sync adds the DiLoCo outer step::

        pull global theta  (client holds it as theta_0, done in setup_async)
        for each window:                        # mixin train loop
            H local RL steps                    # _run_window
            theta_local = trainer.get_full_state_dict_cpu()
            new_global  = client.push(theta_0 - theta_local)  # server outer step
            trainer.load_full_state_dict_cpu(new_global); refresh generators
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLTrainer.Config):
        server_address: str = ""
        """torchft server /sync URL (usually from $DILOCO_SERVER_ADDR)."""
        heartbeat_address: str = ""
        """torchft server /heartbeat URL (usually from $DILOCO_HB_ADDR)."""
        replica_id: int = 0
        """Identifier for logging (usually from $DILOCO_REPLICA_ID)."""
        sync_every: int = 4
        """Local RL steps between parameter-server syncs (DiLoCo inner-loop length)."""
        num_outer_steps: int = 0
        """Fixed number of sync windows to run (step-bound run). Mutually
        exclusive with train_seconds: set exactly one."""
        train_seconds: float = 0.0
        """Wall-clock training budget in seconds (time-bound run): keep running
        sync_every-step windows until the deadline (checked at window
        boundaries, so a run overshoots by at most one window). Slow or
        heterogeneous workers naturally complete fewer steps in the same time.
        Mutually exclusive with num_outer_steps: set exactly one."""
        should_quantize: bool = False
        """fp16 wire transfer (must match the server)."""
        num_generators: int = 1
        """Number of independent vLLM engines this replica spawns (each of
        generator.parallelism's world size, on its own GPU slice out of the
        replica's CUDA_VISIBLE_DEVICES pool). Requests are spread round-robin
        by the GeneratorRouter; weight refreshes at the window boundary fan out
        to all engines. The replica needs trainer_ws + num_generators *
        generator_ws visible GPUs (launch_decentralized.sh's
        GPUS_PER_REPLICA)."""

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

    def __init__(self, config: "HeLoCoRLReplica.Config"):
        super().__init__(config)
        self.client: HeLoCoRLClient | None = None

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn actors (HeLoCoPolicyTrainer) and connect the torchft client.

        Reuses the base setup verbatim but (1) swaps in the HeLoCo trainer
        subclass via a scoped monkeypatch of the symbol setup_async spawns, and
        (2) after setup, builds the torchft client and adopts the server's shared
        theta so all replicas start from the same global weights.
        """
        cfg = self.config
        if not cfg.server_address:
            raise ValueError("server_address is required (set $DILOCO_SERVER_ADDR)")

        # Spawn HeLoCoPolicyTrainer instead of the base PolicyTrainer. The
        # inherited Controller.setup_async resolves the trainer actor class from
        # the rl.controller module globals at spawn time, so scope-patch that
        # symbol for the duration of the base setup.
        orig = _rl_controller_mod.PolicyTrainer
        _rl_controller_mod.PolicyTrainer = HeLoCoPolicyTrainer
        try:
            await super().setup_async(
                trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
            )
        finally:
            _rl_controller_mod.PolicyTrainer = orig

        # Build the torchft client. Parameter names/shapes come from a throwaway
        # meta model built from the SAME model_spec as the server's global model,
        # so both agree on name ordering without a runtime handshake.
        with torch.device("meta"):
            meta_model = cfg.model_spec.model.build()
        names, shapes, dtypes = param_metadata(meta_model)
        del meta_model
        self.client = HeLoCoRLClient(
            cfg.server_address,
            names,
            shapes,
            dtypes,
            heartbeat_address=cfg.heartbeat_address or None,
            should_quantize=cfg.should_quantize,
        )
        self.client.start_heartbeat()

        # Adopt the server's shared theta, then refresh the generator from it.
        global_sd = self.client.pull()
        await self.trainer.load_full_state_dict_cpu.call(global_sd)
        await self.trainer.push_model_state_dict.call()
        await self.generator_router.pull_model_state_dict(policy_version=0)
        logger.info(
            "[replica %d] connected to server; adopted global theta", cfg.replica_id
        )

    async def _window_sync(self, t0: float) -> None:
        """The DiLoCo sync boundary: push theta_local (the server applies its
        outer step to the pseudo-gradient), adopt the returned global theta,
        refresh the generators from it, and apply any DyLU window-length
        recommendation to the next window."""
        theta_local = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call()
        )
        speed = self._sync_every / max(time.perf_counter() - t0, 1e-6)
        new_global = self.client.push(theta_local, speed=speed)
        await self.trainer.load_full_state_dict_cpu.call(new_global)
        await self.trainer.push_model_state_dict.call()
        await self.generator_router.pull_model_state_dict(
            policy_version=self._policy_version
        )
        if self.client.last_dylu_steps > 0:
            self._sync_every = self.client.last_dylu_steps
        return None

    async def close(self):
        if self.client is not None:
            self.client.stop_heartbeat()
        await super().close()
