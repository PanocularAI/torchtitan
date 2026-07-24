# Copyright (c) Panocular AI.
#
# On-GPU trainer actor for HeLoCo RL.

import logging

import torch
from monarch.actor import endpoint
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpoint_utils import canonical_fqn
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

    @endpoint
    async def get_full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Return full unsharded *parameters* as native-named fp32 CPU tensors.

        Mirrors ``push_model_state_dict``'s unshard but returns to the controller
        and filters out buffers. ``cpu_offload=True`` keeps gathered tensors off
        the GPU; fp32 because the pseudo-gradient subtraction and outer optimizer
        run in fp32 on the server.
        """
        # Iterate named_parameters() (the same source param_metadata keys the
        # wire buffers on) and unshard each DTensor with full_tensor(). Do NOT
        # use DCP get_model_state_dict: it triggers per-module state_dict hooks
        # (e.g. FusedQKVLinear's wq/wk/wv split, now default) whose synthetic
        # keys have no real submodule, so DCP's FQN resolution raises
        # AttributeError. named_parameters() yields the real (fused) params, so
        # names stay aligned with the server/client ordering.
        # canonical_fqn strips compile/AC/FSDP wrapper segments (e.g. _orig_mod)
        # so keys match the uncompiled meta-model names the server/client are
        # built from (see param_metadata).
        sd = {}
        for name, param in self.model.named_parameters():
            tensor = param.detach()
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            sd[canonical_fqn(name)] = tensor.to(device="cpu", dtype=torch.float32)
        return sd

    @endpoint
    async def load_full_state_dict_cpu(
        self, global_sd: dict[str, torch.Tensor]
    ) -> None:
        """Load global theta (native-named, fp32, CPU) back into the FSDP/TP model.

        Every rank receives the full dict (the controller `.call()`s the whole
        trainer mesh), so each param is distributed onto its own mesh/placement
        here rather than via ``full_state_dict=True, broadcast_from_rank0=True``:
        that path discovers shardings by walking ``named_children()``, which the
        HF transformers backend overrides to the titan-alias view
        (tok_embeddings/layers/...) whose FQNs never match these canonical
        names — full CPU tensors would then reach DTensor params unconverted
        and fail the copy. Keying off the model's own state dict is
        backend-neutral. ``strict=False`` because we send parameters only
        (buffers are not part of the DiLoCo exchange). After the discontinuous
        theta jump we clear the inner optimizer state (stale momentum would
        bias the next window) but leave ``policy_version`` monotone so
        staleness tracking stays correct.
        """
        train_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        # Copy the incoming global theta straight into named_parameters() (same
        # keying as param_metadata / get_full_state_dict_cpu), redistributing
        # each full CPU tensor onto its param's mesh/placement. Avoids DCP
        # set_model_state_dict, which trips on FusedQKVLinear's hook-synthesized
        # wq/wk/wv keys (no real submodule -> FQN AttributeError). Parameters
        # only (buffers are not part of the DiLoCo exchange).
        local_params = {
            canonical_fqn(n): p for n, p in self.model.named_parameters()
        }
        with torch.no_grad():
            for k, v in global_sd.items():
                ref = local_params.get(k)
                if ref is None:
                    continue
                v = v.to(dtype=train_dtype, device=ref.device)
                if isinstance(ref, DTensor):
                    v = distribute_tensor(v, ref.device_mesh, ref.placements)
                ref.copy_(v)

        self.optimizers.zero_grad(set_to_none=True)
        for opt in self.optimizers:
            opt.state.clear()
        logger.info(
            "HeLoCoPolicyTrainer loaded global theta (policy_version=%d)",
            self.policy_version,
        )
