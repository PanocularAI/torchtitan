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

from torch.distributed.tensor import distribute_tensor, DTensor

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

        Every rank receives the full dict (the controller `.call()`s the whole
        trainer mesh), so each param is distributed onto its own mesh/placement
        here rather than via ``full_state_dict=True, broadcast_from_rank0=True``:
        that path discovers shardings by walking ``named_children()``, which the
        HF transformers backend overrides to the titan-alias view
        (tok_embeddings/layers/...) whose FQNs never match these canonical
        names — full CPU tensors would then reach DTensor params unconverted
        and fail the copy. Keying off the model's own state dict is
        backend-neutral. ``strict=False`` because we send parameters only
        (buffers aren't part of the exchange). After the discontinuous theta
        jump we clear the inner optimizer state (stale momentum would bias the
        next window) but leave ``policy_version`` monotone so staleness
        tracking stays correct.
        """
        train_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        local_sd = get_model_state_dict(self.model)
        sd = {}
        for k, v in global_sd.items():
            ref = local_sd.get(k)
            if ref is None:
                continue
            v = v.to(dtype=train_dtype)
            if isinstance(ref, DTensor):
                v = distribute_tensor(
                    v.to(device=ref.device), ref.device_mesh, ref.placements
                )
            sd[k] = v

        set_model_state_dict(
            self.model,
            model_state_dict=sd,
            options=StateDictOptions(strict=False),
        )

        self.optimizers.zero_grad(set_to_none=True)
        for opt in self.optimizers:
            opt.state.clear()
        logger.info(
            "SnapshotPolicyTrainer loaded full theta (policy_version=%d)",
            self.policy_version,
        )
