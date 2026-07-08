# Copyright (c) Panocular AI.
#
# On-GPU trainer actor with a full-parameter CPU snapshot/restore pair, used
# by AsyncInferenceReplica to read the whole model out for relay publishing
# (see trainer.py's setup_async). Not specific to any sync protocol --
# torchtitan.experiments.async_rl.heloco.actors.HeLoCoPolicyTrainer implements
# an analogous snapshot/restore pair for its own parameter-server exchange,
# kept as a separate class since the two exchanges' shapes differ enough not
# to share code.

import logging

import torch
from monarch.actor import endpoint
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer

logger = logging.getLogger(__name__)


class SnapshotPolicyTrainer(PolicyTrainer):
    """PolicyTrainer + full-parameter snapshot/restore endpoints.

    The base forward_backward/optim_step are reused unchanged (stock token-level
    GRPO loss); only the full-state-dict exchange endpoints are added::

        theta = trainer.get_full_state_dict_cpu()   # native names, CPU, fp32
        trainer.load_full_state_dict_cpu(theta)      # theta -> model
    """

    def _param_name_set(self) -> set[str]:
        return {name for name, _ in self.model.named_parameters()}

    @endpoint
    async def get_full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Return full unsharded *parameters* as native-named fp32 CPU tensors.

        Mirrors ``push_model_state_dict``'s unshard but returns to the controller
        and filters out buffers. ``cpu_offload=True`` keeps gathered tensors off
        the GPU; fp32 for a numerically stable exchange format.
        """
        sd = get_model_state_dict(
            self.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        param_names = self._param_name_set()
        return {
            k: v.detach().to(device="cpu", dtype=torch.float32)
            for k, v in sd.items()
            if k in param_names
        }

    @endpoint
    async def load_full_state_dict_cpu(
        self, global_sd: dict[str, torch.Tensor]
    ) -> None:
        """Load full theta (native-named, fp32, CPU) back into the FSDP/TP model.

        ``broadcast_from_rank0=True`` distributes from the rank that received the
        weights to the rest of the trainer mesh. ``strict=False`` because we
        send parameters only (buffers aren't part of the exchange). After the
        discontinuous theta jump we clear the inner optimizer state (stale
        momentum would bias the next window) but leave ``policy_version``
        monotone so staleness tracking stays correct.
        """
        train_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        sd = {k: v.to(dtype=train_dtype) for k, v in global_sd.items()}

        set_model_state_dict(
            self.model,
            model_state_dict=sd,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )

        self.optimizers.zero_grad(set_to_none=True)
        for opt in self.optimizers:
            opt.state.clear()
        logger.info(
            "SnapshotPolicyTrainer loaded full theta (policy_version=%d)",
            self.policy_version,
        )
