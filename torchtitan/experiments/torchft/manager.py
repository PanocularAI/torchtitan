# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.distributed.distributed_c10d import ReduceOp

from torchtitan.config import Configurable
from torchtitan.tools.logging import logger

if importlib.util.find_spec("torchft") is not None:
    import torchft
    from torchft.checkpointing.pg_transport import PGTransport

    if TYPE_CHECKING:
        from torchft import local_sgd

    has_torchft = True
else:
    has_torchft = False


class TorchFTManager(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """
        Enable TorchFT integration. When TorchFT is enabled, HSDP will be used.
        And --fault_tolerance.data_parallel_replicate_degree should be 1 and
        --fault_tolerance.group_size will be used to control the maximum
        replicate group size as the replicate group size is dynamic.
        Note that this is still an experimental feature.
        """

        process_group: str = "gloo"
        """
        The process group to use for fault tolerance. Currently, only "gloo" and "nccl" are supported.
        """

        process_group_timeout_ms: int = 10000
        """
        The process group will abort if operations don't succeed within this duration.
        Note: This currently only works with gloo process group.
        """

        replica_id: int = 0
        """The TorchFT replica ID of this run."""

        group_size: int = 0
        """
        The number of TorchFT replicate groups. This number will be used for
        dataloader to split the dataset across the replicate groups and FSDP
        dimension
        """

        min_replica_size: int = 1
        """The minimum number of FT replica for each step."""

        semi_sync_method: str | None = None
        """
        The algorithm to use for semi-sync training. Currently, only "local_sgd" and "diloco" from
        torchft are supported
        (https://github.com/pytorch/torchft/blob/360c5c534bdeac959507e9d238ba9f3902d3fda9/torchft/local_sgd.py#L41)
        """

        manager_hostname: str | None = None
        """
        The hostname to advertise to the TorchFT lighthouse server if rank == 0.
        """

        use_pg_checkpoint_transport: bool = False
        """
        Whether to use the process group for checkpoint transport.
        """

        rank0_synchronization_only: bool = False
        """
        Whether inter-replica synchronization occurs only among rank 0. This allows training and healing in
        heterogeneous configurations (i.e., varying sharding degree). Note that this will increase rank 0's
        memory footprint and introduce potential network contention during inter-replica synchronization.
        This feature is experimental.
        """

        copy_pseudogradients_to_cpu: bool = False
        """
        Whether to copy the pseudogradients to the CPU before the outer step when process_group is set to "gloo".
        This flag is introduced as a workaround due to a bug when using a process group with the Gloo backend on
        AMD GPU tensors. Ignored when process_group is not "gloo" or rank0_synchronization_only is "true".
        """

    def __init__(
        self,
        config: Config,
    ) -> None:
        if not config.enable:
            self._manager = None
            return

        if not has_torchft:
            raise ImportError("torchft is not installed. Please install it.")

        self._rank0_synchronization_only = config.rank0_synchronization_only
        if self._rank0_synchronization_only:
            assert config.num_fragments == 1, "num_fragments > 1 not supported with rank 0 synchronization only"

        process_group_timeout = timedelta(milliseconds=config.process_group_timeout_ms)
        if config.process_group == "gloo":
            pg = torchft.ProcessGroupGloo(timeout=process_group_timeout)
            if config.use_pg_checkpoint_transport:
                pg_checkpoint_transport = PGTransport(pg, timeout=process_group_timeout, device="cpu")
        elif config.process_group == "nccl":
            pg = torchft.ProcessGroupNCCL(timeout=process_group_timeout)
            if config.use_pg_checkpoint_transport:
                pg_checkpoint_transport = PGTransport(pg, timeout=process_group_timeout, device="cuda")
        elif config.process_group == "mccl":
            import torchcomms
            from torchft.torchcomms import ProcessGroupTorchComms

            comm = torchcomms.new_comm(
                "mccl",
                device=torch.device("cuda"),
                name="mccl_ft",
                timeout=process_group_timeout,
                enable_reconfigure=True,
            )
            pg = ProcessGroupTorchComms(comm, timeout=process_group_timeout)
        else:
            raise ValueError(f"Unsupported process group: {config.process_group}")

        # If the training method is specific, then the quorum should be synchronous
        self.use_async_quorum = config.semi_sync_method is None

        init_sync = not config.rank0_synchronization_only
        checkpoint_transport = pg_checkpoint_transport if config.use_pg_checkpoint_transport else None
        self._manager = torchft.Manager(
            pg=pg,
            min_replica_size=config.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=self.use_async_quorum,
            replica_id=f"torchtitan_ft_{config.replica_id}",
            hostname=config.manager_hostname,
            init_sync=init_sync,
            checkpoint_transport=checkpoint_transport,
            rank0_synchronization_only=config.rank0_synchronization_only,
            copy_pseudogradients_to_cpu=config.copy_pseudogradients_to_cpu
        )
        self.group_size = config.group_size
        self.replica_id = config.replica_id

        if self.use_async_quorum:
            self.replicate_pg = torchft.process_group.ManagedProcessGroup(self._manager)
            self.replicate_pg.register("dp_replicate")

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def manager(self) -> "torchft.Manager":
        assert self._manager is not None
        return self._manager

    @property
    def rank0_synchronization_only(self) -> bool:
        if not hasattr(self, "_rank0_synchronization_only"):
            return False
        return self._rank0_synchronization_only

    def get_dp_info(self, dp_degree: int, dp_rank: int) -> tuple[int, int]:
        if self.enabled:
            return dp_degree * self.group_size, dp_degree * self.replica_id + dp_rank
        else:
            return dp_degree, dp_rank

    def maybe_set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
        if self.enabled and self.use_async_quorum:

            def all_reduce_hook(output):
                dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)

            def apply_set_all_reduce_hook(m):
                if isinstance(m, FSDPModule):
                    m.set_all_reduce_hook(all_reduce_hook)

            for model_part in model_parts:
                model_part.apply(apply_set_all_reduce_hook)

    @property
    def loss_sync_pg(
        self,
    ) -> "torchft.process_group.ManagedProcessGroup" | None:
        if self.enabled and self.use_async_quorum:
            return self.replicate_pg
        else:
            # skip loss sync when using semi-sync training
            return None


def maybe_semi_sync_training(
    ft_config: "TorchFTManager.Config",
    ft_manager: TorchFTManager,
    model: torch.nn.Module,
    n_layers: int,
    optimizer: torch.optim.Optimizer,
    fragment_fn: Callable[..., list[nn.Module]] | None = None,
) -> AbstractContextManager["local_sgd.DiLoCo" | "local_sgd.LocalSGD" | None]:
    """
    If TorchFT is enabled and the config is set, use semi_sync_method
    """
    from torchtitan.experiments.torchft.config import (
        FaultTolerance as ExtendedTorchFTConfig,
    )

    extend_ft_config = cast(ExtendedTorchFTConfig, ft_config)
    semi_sync_method = extend_ft_config.semi_sync_method
    if extend_ft_config.enable and semi_sync_method is not None:
        from torchft import local_sgd

        assert (
            ft_manager._manager is not None
        ), "TorchFTManager must be enabled to use semi-sync training."
        logger.info(
            f"using fragment function to split model: {fragment_fn is not None}"
        )
        if semi_sync_method.lower() == "diloco":
            if fragment_fn:
                model_parts = fragment_fn(model, extend_ft_config, n_layers)
            else:
                model_parts = [model]

            # Create the outer optimizer based on the inner optimizer parameters.
            outer_optimizers = []
            for model in model_parts:
                params = [p for p in model.parameters() if p.requires_grad]
                outer_optimizer = torch.optim.SGD(
                    params, lr=0.7, momentum=0.9, nesterov=True
                )
                outer_optimizers.append(outer_optimizer)

            return local_sgd.DiLoCo(
                manager=ft_manager._manager,
                model_fragments=model_parts,
                inner_optimizer=optimizer,
                outer_optimizer=outer_optimizers,
                sync_every=extend_ft_config.sync_steps,
                should_quantize=extend_ft_config.should_quantize,
                fragment_sync_delay=extend_ft_config.fragment_sync_delay,
                fragment_update_alpha=extend_ft_config.fragment_update_alpha,
            )
        elif semi_sync_method.lower() == "local_sgd":
            return local_sgd.LocalSGD(
                manager=ft_manager._manager,
                model=model,
                optimizer=optimizer,
                sync_every=extend_ft_config.sync_steps,
            )
        elif semi_sync_method.lower() == "heloco":
            # The parameter-server member of the family: each worker POSTs
            # its pseudo-gradient to torchft's HeLoCoServer over HTTP (the
            # decentralized_rl parameter_server process) and pulls back the
            # look-ahead global params -- no cross-worker collective. The FT
            # manager (and its lighthouse) stays required regardless: the
            # trainer derives its dataloader shard from it. The server's
            # URLs are runtime addresses launchers export from the PS
            # coordinator's stdout, hence env rather than config fields.
            import os

            from torchft.async_diloco import AsyncDiLoCo

            server_address = os.environ.get("DILOCO_SERVER_ADDR", "").strip()
            if not server_address:
                raise RuntimeError(
                    "semi_sync_method='heloco' needs the parameter server's "
                    "/sync URL in $DILOCO_SERVER_ADDR (launchers export it "
                    "from the run's PS coordinator); got an empty value. Use "
                    "semi_sync_method='diloco' for lighthouse-coordinated "
                    "training with no parameter server."
                )
            heartbeat = os.environ.get("DILOCO_HB_ADDR", "").strip() or None
            logger.info(
                f"heloco worker: syncing to {server_address} every "
                f"{extend_ft_config.sync_steps} steps "
                f"(heartbeat: {heartbeat or 'disabled'})"
            )
            return AsyncDiLoCo(
                server_address=server_address,
                # torchtitan builds models through FSDP2 (DTensor params even
                # at world size 1); torchft is DTensor-agnostic, so hand it
                # the plain-tensor view of this model (see _PlainParamsView).
                model=_PlainParamsView(model),
                inner_optimizer=optimizer,
                sync_every=extend_ft_config.sync_steps,
                fragment_update_alpha=extend_ft_config.fragment_update_alpha,
                heartbeat_address=heartbeat,
                should_quantize=extend_ft_config.should_quantize,
            )
        else:
            raise ValueError(
                f"Unknown training method: {semi_sync_method}, only 'diloco', 'local_sgd' and 'heloco' are supported."
            )
    return nullcontext()


class _PlainParamsView:
    """FSDP2 -> torchft adapter: a model's parameters as PLAIN tensors.

    torchft's AsyncDiLoCo does ordinary tensor arithmetic on parameters
    (``copy_``, subtraction, ``torch.cat``), which rejects the DTensors
    torchtitan's FSDP2 build yields. It reaches the model through exactly one
    method -- ``named_parameters()`` -- so unwrapping there is the entire
    adaptation. ``to_local()`` returns a view SHARING STORAGE with the
    DTensor, so AsyncDiLoCo's in-place adoption of pulled global params
    writes straight into the real model.

    Valid only while each parameter's local tensor IS the whole parameter
    (one-device or fully replicated mesh). For a sharded parameter,
    ``to_local()`` is just this rank's slice, and pushing it as though it
    were the whole tensor would corrupt the sync silently -- that case raises.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def named_parameters(self):
        from torch.distributed.tensor import DTensor

        for name, p in self._model.named_parameters():
            if not isinstance(p, DTensor):
                yield name, p
                continue
            local = p.to_local()
            if tuple(local.shape) != tuple(p.shape):
                raise RuntimeError(
                    f"parameter {name!r} is SHARDED across the replica "
                    f"(local {tuple(local.shape)} vs global "
                    f"{tuple(p.shape)}), which semi_sync_method='heloco' "
                    "cannot sync: the parameter server would receive this "
                    "rank's shard as if it were the whole tensor. Use "
                    "single-device replicas (scale out with more replicas), "
                    "or diloco/local_sgd, whose lighthouse path handles "
                    "sharded models."
                )
            yield name, local

