# Copyright (c) Panocular AI.
#
# On-GPU trainer actor for HeLoCo RL.

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


class HeLoCoPolicyTrainer(PolicyTrainer):
    """PolicyTrainer + full-parameter snapshot/restore for the HeLoCo sync.

    The base forward_backward/optim_step are reused unchanged (stock token-level
    GRPO loss); only the parameter-server exchange endpoints are added. The
    controller drives one window like::

        theta_local = trainer.get_full_state_dict_cpu()   # native names, CPU, fp32
        new_global  = client.push(theta_local, speed)     # server outer step
        trainer.load_full_state_dict_cpu(new_global)      # theta <- global
        trainer.push_model_state_dict()                   # refresh generator
        generator.pull_model_state_dict(v)
    """

    def _param_name_set(self) -> set[str]:
        return {name for name, _ in self.model.named_parameters()}

    @endpoint
    async def get_full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Return full unsharded *parameters* as native-named fp32 CPU tensors.

        Mirrors ``push_model_state_dict``'s unshard but returns to the controller
        and filters out buffers. ``cpu_offload=True`` keeps gathered tensors off
        the GPU; fp32 because the pseudo-gradient subtraction and outer optimizer
        run in fp32 on the server.
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
        """Load global theta (native-named, fp32, CPU) back into the FSDP/TP model.

        ``broadcast_from_rank0=True`` distributes from the rank that received the
        global weights to the rest of the trainer mesh. ``strict=False`` because
        we send parameters only (buffers are not part of the DiLoCo exchange).
        After the discontinuous theta jump we clear the inner optimizer state
        (stale momentum would bias the next window) but leave ``policy_version``
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
            "HeLoCoPolicyTrainer loaded global theta (policy_version=%d)",
            self.policy_version,
        )
