# Copyright (c) Panocular AI.
#
# Standalone parameter-server process for HeLoCo RL.
#
# Builds the authoritative *unsharded, fp32, CPU* global model, wraps it in
# torchft's HeLoCo parameter server (the system's outer optimizer: direction-
# aware, heterogeneity-aware staleness correction; ``--outer_method diloco``
# swaps in plain async DiLoCo, same wire protocol), prints the addresses
# replicas connect to, and serves until killed.
#
# It also provides ``param_metadata`` (the single source of truth for parameter
# name ordering) and the server/optimizer factories with the outer-method flag.

import argparse
import logging
import os
import time

import torch
from torch import nn

from torchft.async_diloco import AsyncDiLoCoServer, DelayedNesterovOptimizer
from torchft.heloco import HeLoCoOptimizer, HeLoCoServer

logger: logging.Logger = logging.getLogger(__name__)


def param_metadata(
    model: nn.Module,
) -> tuple[list[str], dict[str, torch.Size], dict[str, torch.dtype]]:
    """Return ``(names, shapes, dtypes)`` in ``named_parameters()`` order.

    This is the single source of truth for parameter ordering. The client is
    constructed from the SAME ordering so the flat coalesced wire buffers on
    both ends stay aligned -- the most likely silent bug in the transfer.
    The server stores fp32, so dtypes are reported as fp32.
    """
    names: list[str] = []
    shapes: dict[str, torch.Size] = {}
    dtypes: dict[str, torch.dtype] = {}
    for name, p in model.named_parameters():
        names.append(name)
        shapes[name] = p.shape
        dtypes[name] = torch.float32
    return names, shapes, dtypes


def build_server(
    model: nn.Module,
    *,
    outer_method: str = "heloco",
    lr: float = 0.7,
    momentum: float = 0.9,
    nesterov_period: int = 2,
    port: int = 0,
    dylu_H: int = 0,
    grace_period: float = 0.0,
    heartbeat_timeout: float = 15.0,
    should_quantize: bool = False,
):
    """Build the parameter server around an unsharded CPU model.

    Both outer optimizers share the same wire protocol, so workers are identical
    regardless of choice:
      - ``heloco``: HeLoCo — direction-aware, heterogeneity-aware staleness
        correction of each pseudo-gradient.
      - ``diloco``: plain async DiLoCo with Delayed-Nesterov momentum
        (``nesterov_period`` controls how often the momentum correction applies;
        set >= number of workers).
    """
    model = model.to(device="cpu", dtype=torch.float32)
    if outer_method == "heloco":
        outer = HeLoCoOptimizer(model.parameters(), lr=lr, momentum=momentum)
        server_cls = HeLoCoServer
    elif outer_method == "diloco":
        outer = DelayedNesterovOptimizer(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov_period=nesterov_period,
        )
        server_cls = AsyncDiLoCoServer
    else:
        raise ValueError(f"unknown outer_method {outer_method!r} (heloco|diloco)")

    return server_cls(
        model,
        outer,
        port=port,
        dylu_H=dylu_H,
        grace_period=grace_period,
        heartbeat_timeout=heartbeat_timeout,
        should_quantize=should_quantize,
    )


def build_global_model(model_spec, hf_assets_path: str) -> nn.Module:
    """Build the authoritative *unsharded fp32 CPU* global model.

    Mirrors the base trainer's build path (meta -> to_empty -> init_weights ->
    optional HF load) but WITHOUT parallelism, so the server holds one whole
    replicated copy. The named_parameters() order matches both the orchestrator's
    metadata model and the trainer's get_model_state_dict FQNs (all built from
    the same model_spec), which is what keeps the wire transfer aligned.

    Replicas adopt these weights on their first pull, so this is the shared
    starting point for the whole swarm.
    """
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict,
        StateDictOptions,
    )

    with torch.device("meta"):
        model = model_spec.model.build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    if model_spec.state_dict_adapter is not None:
        import os

        import torch.distributed.checkpoint as dcp

        adapter = model_spec.state_dict_adapter(model_spec.model, hf_assets_path)
        if os.path.isdir(hf_assets_path):
            try:
                storage_reader = adapter.get_hf_storage_reader(hf_assets_path)
                hf_sd = adapter.to_hf(model.state_dict())
                dcp.load(hf_sd, storage_reader=storage_reader)
                titan_sd = adapter.from_hf(hf_sd)
                set_model_state_dict(
                    model, titan_sd, options=StateDictOptions(strict=False)
                )
                logger.info(
                    "global model: loaded HF checkpoint from %s", hf_assets_path
                )
            except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "global model: no usable HF checkpoint at %s (%s); "
                    "using init_weights (fine for debug)",
                    hf_assets_path,
                    exc,
                )
        else:
            logger.warning(
                "global model: hf_assets_path %s not a dir; using init_weights",
                hf_assets_path,
            )

    return model.to(dtype=torch.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="HeLoCo RL parameter server")
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
    parser.add_argument("--port", type=int, default=29520)
    parser.add_argument("--dylu_H", type=int, default=0)
    parser.add_argument("--grace_period", type=float, default=0.0)
    parser.add_argument(
        "--heartbeat_timeout",
        type=float,
        default=15.0,
        help="seconds without a heartbeat before a worker is dropped",
    )
    parser.add_argument("--should_quantize", action="store_true")
    # Config selects the model_spec / hf_assets_path (same registry the
    # replicas use), e.g. --module async_rl --config rl_heloco_qwen3_0_6b.
    # --hf_assets_path overrides the preset's default checkpoint dir so the
    # global model loads the exact checkpoint the trainers load (launchers
    # that fetch a HF repo point every role at the fetched dir).
    parser.add_argument("--module", type=str, default="async_rl")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hf_assets_path", type=str, default=None)
    args = parser.parse_args()

    from torchtitan.config.manager import ConfigManager

    cfg_args = ["--module", args.module, "--config", args.config]
    if args.hf_assets_path:
        # Env (not just the CLI overlay): the rl_*_hf presets resolve the
        # ARCHITECTURE from RL_HF_ASSETS_PATH at config-fn time; the flag
        # overlay below only retargets where the checkpoint loads from.
        os.environ["RL_HF_ASSETS_PATH"] = args.hf_assets_path
        cfg_args.append(f"--hf_assets_path={args.hf_assets_path}")
    replica_cfg = ConfigManager().parse_args(cfg_args)
    model = build_global_model(replica_cfg.model_spec, replica_cfg.hf_assets_path)

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

    # Replicas read these from the environment (see launch_decentralized.sh).
    print(f"DILOCO_SERVER_ADDR={server.address()}", flush=True)
    print(f"DILOCO_HB_ADDR={server.heartbeat_address()}", flush=True)
    logger.info("%s RL server serving; ctrl-c to stop", args.outer_method)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("server shutting down")


if __name__ == "__main__":
    main()
