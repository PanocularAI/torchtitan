# Copyright (c) Panocular AI.
#
# Trainer role of the async-inference relay swarm: prime-rl (arXiv:2505.07291).
#
# A PURE-LEARNER trainer (one GPU, no local generation of any kind) whose
# training rollouts come entirely from a pool of REMOTE generator workers on
# separate machines, pushed into an embedded rollout queue. The trainer
# consumes with a max_staleness bound, trains, and publishes its weights to a
# relay tier (SHARDCAST-style) so the workers can pull the next policy version.
# Generation on the same node as the trainer would just be the single-node
# "local" baseline with an async buffer, so there is deliberately no vLLM
# here -- inference is fully delocalized.
#
# The pure-learner shape (no-generator setup, queue consumer, bounded buffer,
# staleness-bounded batch assembly, no trainer-side validation) is inherited
# from PureLearnerReplica. This class adds the embedded rollout queue the
# workers push into and the relay publish of its own weights.

import asyncio
import logging
import pickle
from dataclasses import dataclass

from aiohttp import web

from torchtitan.experiments.async_rl.async_inference.actors import SnapshotPolicyTrainer
from torchtitan.experiments.async_rl.async_inference.relay import (
    build_manifest,
    RelayClient,
    shard_state_dict,
)
from torchtitan.experiments.async_rl.pure_learner import PureLearnerReplica

logger = logging.getLogger(__name__)


class RolloutQueueServer:
    """Bounded queue of ``(worker_id, version, rollout_groups)`` batches
    pushed by remote workers -- the rollout-return path, mirror image of
    relay.py's weight-broadcast direction. Embedded directly in the trainer
    process (its queue lives in the SAME asyncio event loop as the trainer's
    consumer), so a received batch reaches the buffer with zero extra hop off
    the wire. Workers are trusted: no TOPLOC-style verification before
    admission. The pushing side is ``AsyncInferenceWorker``'s
    ``RolloutQueueClient`` (worker.py).

    Backpressure is immediate and non-blocking: a full queue rejects the push
    with 503 rather than holding the HTTP connection open (many concurrent
    workers holding open connections while a slow trainer catches up doesn't
    scale as well as a quick reject). The dropped batch is the worker's
    problem to shrug off and keep generating -- consistent with the trainer's
    own staleness-based dropping philosophy: losing one batch is cheaper than
    stalling generation.
    """

    def __init__(self, maxsize: int = 64):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.num_received = 0
        self.num_rejected = 0

    async def get(self):
        """Consumer side (the trainer): one (worker_id, version, groups) batch."""
        return await self.queue.get()

    def qsize(self) -> int:
        return self.queue.qsize()

    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes([web.post("/rollouts", self._handle_push)])
        return app

    async def _handle_push(self, request: web.Request) -> web.Response:
        data = await request.read()
        try:
            worker_id, version, groups = pickle.loads(data)
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            # TypeError/ValueError cover a payload that unpickles cleanly but
            # isn't a (worker_id, version, groups) 3-tuple (e.g. a stray
            # object from a mismatched worker/trainer version).
            return web.Response(status=400, text=f"malformed rollout payload: {exc}")
        try:
            self.queue.put_nowait((worker_id, version, groups))
        except asyncio.QueueFull:
            self.num_rejected += 1
            return web.Response(status=503, text="rollout queue full; consumer stalled")
        self.num_received += 1
        return web.Response(status=204)


async def run_rollout_queue_server(
    host: str = "0.0.0.0", port: int = 8767, maxsize: int = 64
):
    """Start an embedded rollout-queue server; returns ``(server, runner)`` --
    caller keeps the runner alive and calls ``.cleanup()`` to stop, and reads
    batches directly off ``server`` via ``await server.get()``."""
    server = RolloutQueueServer(maxsize=maxsize)
    runner = web.AppRunner(server.app())
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    logger.info(
        "rollout queue server listening on %s:%d (maxsize=%d)", host, port, maxsize
    )
    return server, runner


class AsyncInferenceReplica(PureLearnerReplica):
    """prime-rl: one pure-learner trainer + a pool of remote generator workers.

    Inherits the pure-learner consumer/buffer/staleness machinery and the
    generator-less setup from PureLearnerReplica. Adds:
      - an embedded ``RolloutQueueServer`` the remote workers push rollouts
        into (drained by the inherited ``_consume_remote_rollouts`` into the
        staleness-bounded buffer);
      - relay publishing of THIS trainer's own weights (its weights are the
        policy -- there is a single trainer, no parameter server): an initial
        publish at setup so the workers can bootstrap, then a publish every
        ``publish_every`` windows.

    Staleness is bounded against ``self._checkpoint_version`` (the latest
    version this trainer published): workers stamp each rollout batch with the
    checkpoint version they loaded off the relay, so the two live in the same
    version space.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PureLearnerReplica.Config):
        relay_addresses: str = ""
        """Comma-separated relay server base URLs (e.g.
        "http://localhost:8765,http://localhost:8766"). Required -- every
        published checkpoint version is uploaded to all of them. Launch
        plumbing (usually from $ASYNC_INFERENCE_RELAY_ADDRS via train.py), so
        it's checked in setup_async rather than __post_init__ (ConfigManager
        calls the --config function with zero args before overlaying CLI
        flags, so a required-with-empty-default field can't be validated at
        __post_init__ time without breaking the CLI path)."""
        num_shards: int = 4
        """Shards per published checkpoint (SHARDCAST-style checkpoint-
        transfer sharding; unrelated to model/optimizer sharding)."""
        publish_every: int = 1
        """Windows between relay publishes (1 = publish every window)."""
        rollout_queue_host: str = "0.0.0.0"
        """Host the embedded rollout-queue server listens on for worker pushes."""
        rollout_queue_port: int = 8767
        """Port the embedded rollout-queue server listens on. Workers need
        this reachable as $ASYNC_INFERENCE_TRAINER_ROLLOUT_ADDR (e.g.
        "http://<this host>:8767")."""
        rollout_queue_maxsize: int = 64
        """Batches buffered before a worker's push is rejected with 503
        (RolloutQueueServer backpressure)."""

        def __post_init__(self):
            PureLearnerReplica.Config.__post_init__(self)
            if self.num_shards < 1:
                raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
            if self.publish_every < 1:
                raise ValueError(
                    f"publish_every must be >= 1, got {self.publish_every}"
                )
            if self.rollout_queue_maxsize < 1:
                raise ValueError(
                    "rollout_queue_maxsize must be >= 1, got "
                    f"{self.rollout_queue_maxsize}"
                )

    def __init__(self, config: "AsyncInferenceReplica.Config"):
        super().__init__(config)
        self._relay_client: RelayClient | None = None
        self._checkpoint_version = 0
        self._window_count = 0
        self._rollout_queue_server = None
        self._rollout_queue_runner = None

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn only the SnapshotPolicyTrainer actor (one learner GPU), build
        the relay client, start the embedded rollout-queue server workers push
        into, and publish the initial checkpoint so the worker pool can
        bootstrap before the first window."""
        cfg = self.config
        if not cfg.relay_addresses:
            raise ValueError(
                "relay_addresses is required (set $ASYNC_INFERENCE_RELAY_ADDRS)"
            )
        self._relay_client = RelayClient(
            [url.strip() for url in cfg.relay_addresses.split(",") if url.strip()]
        )

        await self._spawn_trainer_only(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
            policy_trainer_cls=SnapshotPolicyTrainer,
        )

        (
            self._rollout_queue_server,
            self._rollout_queue_runner,
        ) = await run_rollout_queue_server(
            host=cfg.rollout_queue_host,
            port=cfg.rollout_queue_port,
            maxsize=cfg.rollout_queue_maxsize,
        )

        # Publish the initial (HF-initialized) weights so the remote workers
        # have a checkpoint to load and start generating from before the first
        # window -- otherwise the trainer would starve at window 0 (nothing to
        # consume) waiting for rollouts the workers can't produce yet.
        note = await self._publish_checkpoint()
        logger.info("[replica %d] %s (initial)", cfg.replica_id, note)

    # ------------------------------------------------------------------ #
    # PureLearnerReplica hooks: rollout source + staleness reference.
    # ------------------------------------------------------------------ #

    async def _next_rollout_batch(self):
        """Block on the embedded queue for the next worker-pushed batch (never
        None -- asyncio.Queue.get waits until one arrives)."""
        return await self._rollout_queue_server.get()

    def _staleness_reference(self) -> int:
        return self._checkpoint_version

    # ------------------------------------------------------------------ #
    # Relay publish.
    # ------------------------------------------------------------------ #

    async def _publish_checkpoint(self) -> str:
        theta = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call()
        )
        self._checkpoint_version += 1
        shards = shard_state_dict(theta, self.config.num_shards)
        manifest = build_manifest(self._checkpoint_version, shards)
        await self._relay_client.publish(self._checkpoint_version, shards, manifest)
        total_bytes = sum(manifest.shard_sizes)
        return (
            f"relay: published v{self._checkpoint_version} "
            f"({manifest.num_shards} shards, {total_bytes}B)"
        )

    # ------------------------------------------------------------------ #
    # Coordination boundary.
    # ------------------------------------------------------------------ #

    async def _window_sync(self, t0: float) -> str:
        """Window boundary: publish the new weights to the relay tier on
        publish_every boundaries so the worker pool advances to the next
        policy version. No generator refresh and no producer pausing -- there
        is no local generation."""
        del t0
        stats = f"buffer: depth={self._buffer.qsize()} dropped={self._num_dropped}"
        self._num_dropped = 0
        self._window_count += 1
        if self._window_count % self.config.publish_every == 0:
            relay_note = await self._publish_checkpoint()
            stats = f"{stats} | {relay_note}"
        return stats

    async def close(self):
        if self._rollout_queue_runner is not None:
            await self._rollout_queue_runner.cleanup()
        await super().close()
