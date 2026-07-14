# Copyright (c) Panocular AI.
#
# Trainer role of the async-inference relay swarm: prime-rl (arXiv:2505.07291).
#
# A PURE-LEARNER trainer (one GPU, no local generation of any kind) whose
# training rollouts come entirely from a pool of REMOTE generator workers on
# separate machines, pushed into a standalone rollout-queue process
# (async_inference.rollout_queue) this trainer pops from. The trainer
# consumes with a max_staleness bound, trains, and publishes its weights to a
# relay tier (SHARDCAST-style) so the workers can pull the next policy version.
# Generation on the same node as the trainer would just be the single-node
# "local" baseline with an async buffer, so there is deliberately no vLLM
# here -- inference is fully delocalized.
#
# The pure-learner shape (no-generator setup, queue consumer, bounded buffer,
# staleness-bounded batch assembly, no trainer-side validation) is inherited
# from PureLearnerReplica. This class adds the queue pop client and the relay
# publish of its own weights.

import asyncio
import logging
from dataclasses import dataclass

import torch

from torchtitan.experiments.async_rl.async_inference.actors import SnapshotPolicyTrainer
from torchtitan.experiments.async_rl.async_inference.relay import (
    build_manifest,
    RelayClient,
    shard_state_dict,
)
from torchtitan.experiments.async_rl.async_inference.rollout_queue import (
    RolloutQueuePopClient,
)
from torchtitan.experiments.async_rl.pure_learner import PureLearnerReplica

logger = logging.getLogger(__name__)


class AsyncInferenceReplica(PureLearnerReplica):
    """prime-rl: one pure-learner trainer + a pool of remote generator workers.

    Inherits the pure-learner consumer/buffer/staleness machinery and the
    generator-less setup from PureLearnerReplica. Adds:
      - a pop client on the standalone rollout-queue process the remote
        workers push into (drained by the inherited
        ``_consume_remote_rollouts`` into the staleness-bounded buffer);
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
        rollout_queue_address: str = ""
        """Base URL of the standalone rollout-queue process
        (async_inference.rollout_queue) this trainer pops from, e.g.
        "http://localhost:8767". Required -- launch plumbing (usually from
        $ROLLOUT_QUEUE_ADDR via train.py), so it's checked in setup_async
        rather than __post_init__ (same reasoning as relay_addresses)."""

        def __post_init__(self):
            PureLearnerReplica.Config.__post_init__(self)
            if self.num_shards < 1:
                raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
            if self.publish_every < 1:
                raise ValueError(
                    f"publish_every must be >= 1, got {self.publish_every}"
                )

    def __init__(self, config: "AsyncInferenceReplica.Config"):
        super().__init__(config)
        self._relay_client: RelayClient | None = None
        self._checkpoint_version = 0
        self._window_count = 0
        self._publish_task: asyncio.Task | None = None
        self._queue_client: RolloutQueuePopClient | None = None

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn only the SnapshotPolicyTrainer actor (one learner GPU), build
        the relay client and the rollout-queue pop client, and publish the
        initial checkpoint so the worker pool can bootstrap before the first
        window."""
        cfg = self.config
        if not cfg.relay_addresses:
            raise ValueError(
                "relay_addresses is required (set $ASYNC_INFERENCE_RELAY_ADDRS)"
            )
        if not cfg.rollout_queue_address:
            raise ValueError(
                "rollout_queue_address is required (set $ROLLOUT_QUEUE_ADDR)"
            )
        self._relay_client = RelayClient(
            [url.strip() for url in cfg.relay_addresses.split(",") if url.strip()]
        )
        self._queue_client = RolloutQueuePopClient(cfg.rollout_queue_address)

        await self._spawn_trainer_only(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
            policy_trainer_cls=SnapshotPolicyTrainer,
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
        """Pop one batch from the standalone queue (None when empty; the base
        consumer then waits queue_poll_interval_s and retries)."""
        return await self._queue_client.pop()

    def _staleness_reference(self) -> int:
        return self._checkpoint_version

    # ------------------------------------------------------------------ #
    # Relay publish.
    # ------------------------------------------------------------------ #

    async def _publish_checkpoint(self) -> str:
        theta = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call()
        )
        # bf16 over the wire: halves every publish and every worker download.
        # Safe because relay checkpoints are a dead end -- workers use them
        # only as the generation behavior policy (the engine runs bf16 anyway)
        # and never train on or push back these weights, so unlike the
        # trainer<->server DiLoCo channel the cast can't compound. The IS-
        # corrected loss stays consistent: the recorded behavior logprobs come
        # from the bf16 model that actually generated.
        theta = {k: v.to(dtype=torch.bfloat16) for k, v in theta.items()}
        self._checkpoint_version += 1
        # Shard in a thread: torch.save of ~GB blobs would otherwise block
        # the event loop (and with it the embedded rollout queue).
        shards = await asyncio.to_thread(
            shard_state_dict, theta, self.config.num_shards
        )
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
        policy version. The publish runs as a BACKGROUND task -- snapshotting,
        sharding, and POSTing ~GBs must not stall the next window (the actor
        mailbox serializes the snapshot against optim steps, so the background
        task still reads a consistent theta). If the previous publish is
        somehow still in flight, this boundary's publish is skipped -- workers
        just keep the last version a little longer, which max_staleness
        already tolerates. No generator refresh and no producer pausing --
        there is no local generation."""
        del t0
        stats = f"buffer: depth={self._buffer.qsize()} dropped={self._num_dropped}"
        self._num_dropped = 0
        self._window_count += 1
        if self._window_count % self.config.publish_every == 0:
            prev = self._publish_task
            if prev is not None and not prev.done():
                stats = f"{stats} | relay: publish skipped (previous in flight)"
            else:
                if prev is not None and (exc := prev.exception()) is not None:
                    logger.warning(
                        "[replica %d] previous relay publish failed: %s",
                        self.config.replica_id,
                        exc,
                    )
                self._publish_task = asyncio.create_task(self._publish_checkpoint())
                stats = f"{stats} | relay: publish started (background)"
        return stats

    async def close(self):
        task = self._publish_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await super().close()
