# Copyright (c) Panocular AI.
#
# Controller for heloco_async_inference: prime-rl-style decoupled generation
# scaled to MULTIPLE trainers. N pure-learner HeLoCo trainer replicas (1 GPU
# each, no local generation) draw ALL training rollouts from a shared,
# hub-hosted rollout queue fed by a pool of remote generator workers, and
# coordinate no-barrier through the HeLoCo parameter server. The hub
# (heloco_async_inference.server) co-hosts the parameter server, the shared
# queue, and a relay tier that republishes the CURRENT global theta for the
# generator pool.
#
# The pure-learner shape (no vLLM on the trainer; consume from a remote
# queue; staleness-bounded batch assembly) is inherited from
# PureLearnerReplica. This class adds only the HeLoCo coordination (build the
# torchft client + adopt global theta in setup_async; push pseudo-gradient +
# adopt new global theta in _window_sync) and the hub-queue rollout source.

import logging
import pickle
import time
from dataclasses import dataclass

import aiohttp
import torch

from torchtitan.experiments.async_rl.heloco.actors import HeLoCoPolicyTrainer
from torchtitan.experiments.async_rl.heloco.client import HeLoCoRLClient
from torchtitan.experiments.async_rl.heloco.server import param_metadata
from torchtitan.experiments.async_rl.pure_learner import PureLearnerReplica

logger = logging.getLogger(__name__)


class SharedRolloutQueueClient:
    """Pops one ``(worker_id, version, groups)`` batch at a time from the
    heloco_async_inference hub's ``SharedRolloutQueueServer`` -- the pop-side
    counterpart to ``async_inference.worker.RolloutQueueClient``'s push
    (unchanged; it already targets this same ``POST /rollouts`` endpoint,
    just pointed at the hub instead of a single trainer's embedded queue).

    A non-blocking poll: :meth:`pop` returns ``None`` immediately if the
    queue is empty or the hub is unreachable, never raises. Multiple trainer
    replicas pop from the SAME hub queue concurrently -- an at-most-once
    claim per call, so no two trainers ever consume the same batch.
    """

    def __init__(self, hub_address: str, *, timeout_s: float = 30.0):
        if not hub_address.strip():
            raise ValueError(
                "hub_address is required (set $HELOCO_ASYNC_INFERENCE_HUB_ADDR)"
            )
        self.hub_address = hub_address.rstrip("/")
        self._timeout_s = timeout_s

    async def pop(self):
        """Returns ``(worker_id, version, groups)`` or ``None`` (empty queue
        or unreachable hub)."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s)
            ) as session:
                async with session.post(f"{self.hub_address}/rollouts/pop") as resp:
                    if resp.status == 204:
                        return None
                    if resp.status != 200:
                        logger.warning(
                            "rollout pop from %s failed (status=%d)",
                            self.hub_address,
                            resp.status,
                        )
                        return None
                    data = await resp.read()
                    return pickle.loads(data)
        except aiohttp.ClientError as exc:
            logger.warning("rollout pop from %s failed: %s", self.hub_address, exc)
            return None


class HeLoCoAsyncInferenceReplica(PureLearnerReplica):
    """A pure-learner HeLoCo trainer whose training rollouts come entirely
    from the hub's shared rollout queue (fed by a remote generator pool).

    Inherits the pure-learner consumer/buffer/staleness machinery and the
    generator-less setup from PureLearnerReplica; adds the HeLoCo outer step:
      - ``setup_async``: spawn only the HeLoCoPolicyTrainer actor, build the
        torchft client, and adopt the server's shared theta.
      - ``_window_sync``: push theta_local (the server applies its outer step
        to the pseudo-gradient), adopt the returned global theta -- no
        generator refresh.

    Staleness is bounded against the shared parameter-server revision, not the
    replica's own ``policy_version``: generator workers stamp each rollout
    batch with the checkpoint version they loaded (the hub publishes checkpoint
    version = server revision + 1), and this replica gates on
    ``self._last_known_revision`` (refreshed every ``_window_sync`` from
    ``self.client.revision + 1``). Server revision is shared and advances once
    per applied push; ``policy_version`` advances per local optim step,
    independently per replica -- so comparing a worker's revision tag against a
    local ``policy_version`` would be meaningless.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PureLearnerReplica.Config):
        server_address: str = ""
        """torchft server /sync URL (usually from $DILOCO_SERVER_ADDR)."""
        heartbeat_address: str = ""
        """torchft server /heartbeat URL (usually from $DILOCO_HB_ADDR)."""
        should_quantize: bool = False
        """fp16/int8 pseudo-gradient wire transfer (must match the server)."""
        rollout_queue_address: str = ""
        """The heloco_async_inference hub's base URL (e.g.
        "http://localhost:8768"), usually from
        $HELOCO_ASYNC_INFERENCE_HUB_ADDR. Required -- launch plumbing, so it's
        checked in setup_async rather than __post_init__ (ConfigManager calls
        the --config function with zero args before overlaying CLI flags, so a
        required-with-empty-default field can't be validated at __post_init__
        time without breaking the CLI path)."""

    def __init__(self, config: "HeLoCoAsyncInferenceReplica.Config"):
        super().__init__(config)
        self.client: HeLoCoRLClient | None = None
        self._queue_client: SharedRolloutQueueClient | None = None
        self._last_known_revision = 0

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn only the HeLoCoPolicyTrainer actor (one learner GPU), connect
        the torchft client, and adopt the server's shared theta."""
        cfg = self.config
        if not cfg.server_address:
            raise ValueError("server_address is required (set $DILOCO_SERVER_ADDR)")
        if not cfg.rollout_queue_address:
            raise ValueError(
                "rollout_queue_address is required "
                "(set $HELOCO_ASYNC_INFERENCE_HUB_ADDR)"
            )

        await self._spawn_trainer_only(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
            policy_trainer_cls=HeLoCoPolicyTrainer,
        )

        # Build the torchft client from a throwaway meta model built from the
        # SAME model_spec as the server's global model, so both agree on
        # parameter name ordering without a runtime handshake.
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

        # Adopt the server's shared theta into the trainer model. No generator
        # to refresh and no TorchStore push -- the hub publishes the global
        # model to the relay for the worker pool.
        global_sd = self.client.pull()
        await self.trainer.load_full_state_dict_cpu.call(global_sd)
        self._queue_client = SharedRolloutQueueClient(cfg.rollout_queue_address)
        # +1: the hub publishes checkpoint versions as server revision + 1
        # (see heloco_async_inference/server.py::_watch_and_publish for why),
        # so this reference must be shifted the same way to stay comparable to
        # the revision workers stamp on their rollout batches.
        self._last_known_revision = self.client.revision + 1
        logger.info(
            "[replica %d] connected to server; adopted global theta "
            "(pure learner, no local generation)",
            cfg.replica_id,
        )

    # ------------------------------------------------------------------ #
    # PureLearnerReplica hooks: rollout source + staleness reference.
    # ------------------------------------------------------------------ #

    async def _next_rollout_batch(self):
        """Pop one batch from the hub's shared queue (None when empty; the
        base consumer then waits queue_poll_interval_s and retries)."""
        return await self._queue_client.pop()

    def _staleness_reference(self) -> int:
        return self._last_known_revision

    # ------------------------------------------------------------------ #
    # HeLoCo outer step.
    # ------------------------------------------------------------------ #

    async def _window_sync(self, t0: float) -> str:
        """The HeLoCo outer step, with NO generator refresh: push theta_local
        (the server applies its outer step to the pseudo-gradient), adopt the
        returned global theta, apply any DyLU recommendation, and refresh this
        replica's revision freshness reference for the next window's staleness
        gate."""
        theta_local = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call()
        )
        speed = self._sync_every / max(time.perf_counter() - t0, 1e-6)
        new_global = self.client.push(theta_local, speed=speed)
        await self.trainer.load_full_state_dict_cpu.call(new_global)
        if self.client.last_dylu_steps > 0:
            self._sync_every = self.client.last_dylu_steps
        self._last_known_revision = self.client.revision + 1
        stats = f"buffer: depth={self._buffer.qsize()} dropped={self._num_dropped}"
        self._num_dropped = 0
        return stats

    async def close(self):
        if self.client is not None:
            self.client.stop_heartbeat()
        await super().close()
