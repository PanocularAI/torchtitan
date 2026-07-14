# Copyright (c) Panocular AI.
#
# Parameter-server process for the prime_heloco strategy: the
# HeLoCo parameter server plus a publish loop that pushes each new global
# checkpoint to the SEPARATE relay process over HTTP.
#
# The strategy's coordination plane is three CPU processes (split so the
# multi-GB checkpoint traffic, the rollout traffic, and the HeLoCo merge math
# never share a Python process -- and its GIL/event loop; a colocated publish
# could stall rollout pops for 30s+):
#   - THIS process: the HeLoCo/AsyncDiLoCo parameter server (reused verbatim
#     from heloco.server; torchft's own HTTP server runs in background
#     threads) + a watch loop on server.status()["revision"] that, every
#     publish_every_revisions commits, snapshots the global model (bf16),
#     shards it, and publishes to the relay via RelayClient.
#   - prime.relay: checkpoint distribution to the generator pool.
#   - prime.rollout_queue: the shared rollout queue every trainer
#     replica pops from and every generator worker pushes into.

import argparse
import asyncio
import logging

import aiohttp
import torch

from torchtitan.experiments.async_rl.prime.relay import (
    build_manifest,
    RelayClient,
    shard_state_dict,
)

from torchtitan.experiments.async_rl.heloco.server import (
    build_global_model,
    build_server,
    param_metadata,
)

logger = logging.getLogger(__name__)


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
        # bf16: halves every relay publish and worker download. Safe for this
        # channel only -- workers use the checkpoint purely as the generation
        # behavior policy (their engines run bf16) and never train on or push
        # back these weights, so the cast can't compound; the server's own
        # fp32 global model is untouched.
        state_dict = {
            name: p.detach().to(device="cpu", dtype=torch.bfloat16).clone()
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
    relay_client: RelayClient,
    *,
    num_shards: int,
    publish_every_revisions: int,
    poll_interval_s: float,
) -> None:
    """Publish the global model to the relay/queue process (over HTTP, via
    ``relay_client``) every ``publish_every_revisions`` outer-step commits
    (immediately at startup too, server revision 0's initial weights, so
    workers have something to bootstrap from before the first outer step
    lands). Runs forever; a failed publish is retried on the next revision
    tick rather than crashing the parameter server.

    Published checkpoint versions are ``server revision + 1``, never the raw
    revision: ``PrimeWorker`` starts at version 0 and
    ``RelayClient.fetch_latest`` requires ``manifest.version > min_version``
    (strict), so a checkpoint published under version 0 could never be
    fetched by a freshly-started worker -- a constant +1 shift avoids that
    bootstrap deadlock. The same shift is applied wherever a revision needs
    to be compared as a checkpoint version (see
    ``PrimeHeLoCoReplica._last_known_revision``), so relative
    staleness math is unaffected."""
    last_published_revision = -1
    while True:
        revision = server.status()["revision"]
        if revision - last_published_revision >= publish_every_revisions:
            state_dict, snap_revision = await _consistent_snapshot(
                model, param_names, lambda: server.status()["revision"]
            )
            checkpoint_version = snap_revision + 1
            # Shard in a thread: torch.save of ~GB blobs would otherwise
            # block this loop's revision polling.
            shards = await asyncio.to_thread(shard_state_dict, state_dict, num_shards)
            manifest = build_manifest(checkpoint_version, shards)
            try:
                await relay_client.publish(checkpoint_version, shards, manifest)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "publish of checkpoint v%d failed (%s); retrying on the "
                    "next revision tick",
                    checkpoint_version,
                    exc,
                )
                await asyncio.sleep(poll_interval_s)
                continue
            logger.info(
                "published checkpoint v%d (server revision %d) to the relay "
                "(%d shards, %dB)",
                checkpoint_version,
                snap_revision,
                len(shards),
                sum(manifest.shard_sizes),
            )
            last_published_revision = snap_revision
        await asyncio.sleep(poll_interval_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="prime_heloco hub (param server + relay + shared rollout queue)"
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
    parser.add_argument(
        "--relay_addr",
        type=str,
        required=True,
        help="base URL of the relay process (prime.relay) this "
        "parameter server publishes checkpoints to, e.g. http://host:8768",
    )
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
    # replicas use), e.g. --module async_rl --config rl_prime_heloco_qwen3_0_6b
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

    # Replicas read these from the environment (see launch_prime_heloco.sh).
    print(f"DILOCO_SERVER_ADDR={server.address()}", flush=True)
    print(f"DILOCO_HB_ADDR={server.heartbeat_address()}", flush=True)
    logger.info(
        "%s RL server serving; publishing checkpoints to %s; ctrl-c to stop",
        args.outer_method,
        args.relay_addr,
    )

    try:
        asyncio.run(
            _watch_and_publish(
                server,
                model,
                param_names,
                RelayClient([args.relay_addr]),
                num_shards=args.num_shards,
                publish_every_revisions=args.publish_every_revisions,
                poll_interval_s=args.publish_poll_interval_s,
            )
        )
    except KeyboardInterrupt:
        logger.info("parameter server shutting down")


if __name__ == "__main__":
    main()
