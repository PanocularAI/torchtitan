# Copyright (c) Panocular AI.
#
# Standalone "hub" process for the heloco_async_inference strategy: the
# HeLoCo parameter server PLUS everything the async-inference side of the
# swarm needs to run without a per-trainer relay/queue -- a relay tier
# (checkpoint distribution) and a rollout-return queue SHARED by every
# trainer replica, all colocated on the parameter-server machine.
#
# Three pieces, one process:
#   - The HeLoCo/AsyncDiLoCo parameter server itself (reused verbatim from
#     heloco.server: build_global_model/build_server/param_metadata) -- its
#     own HTTP server (torchft's, Python http.server) runs in background
#     threads, exactly as in heloco/server.py.
#   - One aiohttp app combining a RelayServer (reused from
#     async_inference.relay, unmodified wire protocol) with the new
#     SharedRolloutQueueServer (below) on ONE port, so N trainer replicas
#     and M inference workers all point at the same hub address instead of
#     each trainer running its own embedded relay client / queue server.
#   - A background publish loop that watches the parameter server's
#     revision (server.status()["revision"], its existing public API) and,
#     on every publish_every_revisions commits, shards + publishes the
#     CURRENT global model directly into the co-located RelayServer (an
#     in-process call, no HTTP hop) -- the SHARDCAST checkpoint tier reads
#     directly off the HeLoCo global weights instead of any one trainer's
#     local copy.

import argparse
import asyncio
import logging
import pickle

import torch
from aiohttp import web

from torchtitan.experiments.async_rl.async_inference.relay import (
    build_manifest,
    RelayServer,
    shard_state_dict,
)

from torchtitan.experiments.async_rl.heloco.server import (
    build_global_model,
    build_server,
    param_metadata,
)

logger = logging.getLogger(__name__)


class SharedRolloutQueueServer:
    """Bounded queue of ``(worker_id, version, rollout_groups)`` batches,
    shared by every trainer replica in the swarm -- the multi-trainer
    generalization of ``async_inference.trainer.RolloutQueueServer``'s
    single-embedded-consumer queue.

    Push (``POST /rollouts``, pickled ``(worker_id, version, groups)``) is
    the IDENTICAL wire contract that class already serves, so
    ``async_inference.worker.AsyncInferenceWorker``'s existing
    ``RolloutQueueClient`` needs no changes to target this hub instead of a
    single trainer's embedded queue -- just point its
    ``trainer_rollout_address`` here.

    Pop (``POST /rollouts/pop``) is new: any trainer claims and removes
    exactly one batch per call (at-most-once -- a lost batch is cheaper than
    a trainer that stalls waiting on a wedged one, the same dropping
    philosophy the rest of this swarm uses). Safe under concurrent trainers
    despite no explicit lock: aiohttp's single-threaded event loop serializes
    handler bodies between awaits, and ``asyncio.Queue.get_nowait`` has no
    await in it, so two concurrent pops can't claim the same batch (same
    reasoning as RelayServer's own no-lock thread-safety note).
    """

    def __init__(self, maxsize: int = 256):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.num_received = 0
        self.num_rejected = 0
        self.num_popped = 0

    def qsize(self) -> int:
        return self.queue.qsize()

    def routes(self) -> list:
        return [
            web.post("/rollouts", self._handle_push),
            web.post("/rollouts/pop", self._handle_pop),
        ]

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
            return web.Response(status=400, text=f"malformed rollout payload: {exc}")
        try:
            self.queue.put_nowait((worker_id, version, groups))
        except asyncio.QueueFull:
            self.num_rejected += 1
            return web.Response(status=503, text="rollout queue full; consumer stalled")
        self.num_received += 1
        return web.Response(status=204)

    async def _handle_pop(self, request: web.Request) -> web.Response:
        del request
        try:
            batch = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return web.Response(status=204)
        self.num_popped += 1
        return web.Response(
            body=pickle.dumps(batch), content_type="application/octet-stream"
        )


async def _consistent_snapshot(
    model: torch.nn.Module,
    param_names: list[str],
    get_revision,
    *,
    max_retries: int = 5,
) -> tuple[dict[str, torch.Tensor], int]:
    """Best-effort torn-read-free CPU snapshot of ``model``'s parameters.

    The HeLoCo outer optimizer mutates parameters in place from a background
    thread with no lock exposed to this process; re-checking
    ``get_revision()`` before and after the copy catches (and retries) the
    rare case where a commit landed mid-snapshot. After ``max_retries``
    unstable attempts, publishes the last snapshot anyway (a stale-by-one-
    revision checkpoint is harmless here; the next watch-loop tick corrects
    it) rather than blocking the hub indefinitely.
    """
    revision = get_revision()
    for _ in range(max_retries):
        state_dict = {
            name: p.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, p in model.named_parameters()
            if name in param_names
        }
        new_revision = get_revision()
        if new_revision == revision:
            return state_dict, revision
        revision = new_revision
        await asyncio.sleep(0)
    return state_dict, revision


async def _watch_and_publish(
    server,
    model: torch.nn.Module,
    param_names: list[str],
    relay: RelayServer,
    *,
    num_shards: int,
    publish_every_revisions: int,
    poll_interval_s: float,
) -> None:
    """Publish the global model to the co-located relay every
    ``publish_every_revisions`` outer-step commits (immediately at startup
    too, server revision 0's initial weights, so workers have something to
    bootstrap from before the first outer step lands). Runs forever.

    Published checkpoint versions are ``server revision + 1``, never the raw
    revision: ``AsyncInferenceWorker`` starts at version 0 and
    ``RelayClient.fetch_latest`` requires ``manifest.version > min_version``
    (strict), so a checkpoint published under version 0 could never be
    fetched by a freshly-started worker -- a constant +1 shift avoids that
    bootstrap deadlock. The same shift is applied wherever a revision needs
    to be compared as a checkpoint version (see
    ``HeLoCoAsyncInferenceReplica._last_known_revision``), so relative
    staleness math is unaffected."""
    last_published_revision = -1
    while True:
        revision = server.status()["revision"]
        if revision - last_published_revision >= publish_every_revisions:
            state_dict, snap_revision = await _consistent_snapshot(
                model, param_names, lambda: server.status()["revision"]
            )
            checkpoint_version = snap_revision + 1
            shards = shard_state_dict(state_dict, num_shards)
            manifest = build_manifest(checkpoint_version, shards)
            relay.publish_manifest(checkpoint_version, manifest)
            for idx, data in enumerate(shards):
                relay.publish_shard(checkpoint_version, idx, data)
            logger.info(
                "published checkpoint v%d (server revision %d) to the "
                "co-located relay (%d shards)",
                checkpoint_version,
                snap_revision,
                len(shards),
            )
            last_published_revision = snap_revision
        await asyncio.sleep(poll_interval_s)


def build_hub_app(
    relay: RelayServer, rollout_queue: SharedRolloutQueueServer
) -> web.Application:
    """One aiohttp app serving the relay's checkpoint-distribution routes and
    the shared rollout queue's push/pop routes on a single port -- disjoint
    path prefixes (``/publish``, ``/manifest``, ``/shard`` vs ``/rollouts``),
    so both compose onto ``relay.app()`` with no path rewriting needed and no
    client-side changes to either RelayClient or RolloutQueueClient."""
    app = relay.app()
    app.add_routes(rollout_queue.routes())
    return app


async def _serve(
    server,
    model: torch.nn.Module,
    param_names: list[str],
    *,
    host: str,
    port: int,
    advertise_host: str,
    retain_last: int,
    rollout_queue_maxsize: int,
    num_shards: int,
    publish_every_revisions: int,
    publish_poll_interval_s: float,
) -> None:
    relay = RelayServer(retain_last=retain_last)
    rollout_queue = SharedRolloutQueueServer(maxsize=rollout_queue_maxsize)
    app = build_hub_app(relay, rollout_queue)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    # `host` is the bind interface (often "0.0.0.0"); remote trainers/workers
    # need a real, connectable hostname/IP, so this is printed separately --
    # mirrors torchft's own AsyncDiLoCoServer.address()/advertise_host split.
    print(f"HELOCO_ASYNC_INFERENCE_HUB_ADDR=http://{advertise_host}:{port}", flush=True)
    logger.info(
        "hub (relay + shared rollout queue) listening on %s:%d, advertised as %s (retain_last=%d)",
        host,
        port,
        advertise_host,
        retain_last,
    )
    try:
        await _watch_and_publish(
            server,
            model,
            param_names,
            relay,
            num_shards=num_shards,
            publish_every_revisions=publish_every_revisions,
            poll_interval_s=publish_poll_interval_s,
        )
    finally:
        await runner.cleanup()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="heloco_async_inference hub (param server + relay + shared rollout queue)"
    )
    parser.add_argument(
        "--outer_method", choices=["heloco", "diloco"], default="heloco"
    )
    parser.add_argument("--lr", type=float, default=0.7)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--nesterov_period",
        type=int,
        default=2,
        help="diloco only: pushes between momentum corrections (>= #workers)",
    )
    parser.add_argument("--port", type=int, default=29520, help="HeLoCo /sync port")
    parser.add_argument("--dylu_H", type=int, default=0)
    parser.add_argument("--grace_period", type=float, default=0.0)
    parser.add_argument(
        "--heartbeat_timeout",
        type=float,
        default=15.0,
        help="seconds without a heartbeat before a worker is dropped",
    )
    parser.add_argument("--should_quantize", action="store_true")
    parser.add_argument("--queue_host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--queue_port",
        type=int,
        default=8768,
        help="port for the combined relay + shared-rollout-queue hub",
    )
    parser.add_argument(
        "--queue_advertise_host",
        type=str,
        default=None,
        help="hostname/IP trainers and workers use to reach the hub (default: "
        "$TORCHFT_PS_ADVERTISE_HOST if set, else this machine's hostname -- "
        "NOT --queue_host, which is only the local bind interface)",
    )
    parser.add_argument(
        "--retain_last",
        type=int,
        default=5,
        help="checkpoint versions kept in the relay before eviction",
    )
    parser.add_argument("--rollout_queue_maxsize", type=int, default=256)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="shards per published checkpoint (SHARDCAST-style)",
    )
    parser.add_argument(
        "--publish_every_revisions",
        type=int,
        default=1,
        help="outer-step commits between relay publishes",
    )
    parser.add_argument("--publish_poll_interval_s", type=float, default=1.0)
    # Config selects the model_spec / hf_assets_path (same registry the
    # replicas use), e.g. --module async_rl --config rl_heloco_async_inference_qwen3_0_6b
    parser.add_argument("--module", type=str, default="async_rl")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    from torchtitan.config.manager import ConfigManager

    replica_cfg = ConfigManager().parse_args(
        ["--module", args.module, "--config", args.config]
    )
    model = build_global_model(replica_cfg.model_spec, replica_cfg.hf_assets_path)
    param_names, _, _ = param_metadata(model)

    server = build_server(
        model,
        outer_method=args.outer_method,
        lr=args.lr,
        momentum=args.momentum,
        nesterov_period=args.nesterov_period,
        port=args.port,
        dylu_H=args.dylu_H,
        grace_period=args.grace_period,
        heartbeat_timeout=args.heartbeat_timeout,
        should_quantize=args.should_quantize,
    )

    # Replicas read these from the environment (see launch_heloco_async_inference.sh).
    print(f"DILOCO_SERVER_ADDR={server.address()}", flush=True)
    print(f"DILOCO_HB_ADDR={server.heartbeat_address()}", flush=True)
    logger.info("%s RL server serving; ctrl-c to stop", args.outer_method)

    from torchft.parameter_server import _resolve_advertise_host

    try:
        asyncio.run(
            _serve(
                server,
                model,
                param_names,
                host=args.queue_host,
                port=args.queue_port,
                advertise_host=_resolve_advertise_host(args.queue_advertise_host),
                retain_last=args.retain_last,
                rollout_queue_maxsize=args.rollout_queue_maxsize,
                num_shards=args.num_shards,
                publish_every_revisions=args.publish_every_revisions,
                publish_poll_interval_s=args.publish_poll_interval_s,
            )
        )
    except KeyboardInterrupt:
        logger.info("hub shutting down")


if __name__ == "__main__":
    main()
