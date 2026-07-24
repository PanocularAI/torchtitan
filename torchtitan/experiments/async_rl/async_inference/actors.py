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

    @endpoint
    async def get_full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Return the full unsharded model state dict as fp32 CPU tensors.

        Uses ``self.model.state_dict()`` — the SAME source ``push_model_state_dict``
        stages for the generators — so the relay-published weights carry the exact
        keys the worker's vLLM engine consumes on pull. In particular FusedQKVLinear's
        state_dict hooks split the fused ``wqkv`` back into ``wq``/``wk``/``wv``,
        the layout the generator load path expects; a ``named_parameters()`` dump
        would instead emit the fused key and the worker would load mismatched
        attention weights. DCP ``get_model_state_dict`` can't be used here — it
        trips on FusedQKVLinear's synthetic keys during FQN resolution — so unshard
        each DTensor with ``full_tensor()`` directly (relay shards are torch.save'd,
        so full CPU tensors are required)."""
        sd = {}
        for name, tensor in self.model.state_dict().items():
            tensor = tensor.detach()
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            sd[name] = tensor.to(device="cpu", dtype=torch.float32)
        return sd

    @endpoint
    async def load_full_state_dict_cpu(
        self, global_sd: dict[str, torch.Tensor]
    ) -> None:
        """Load a full state dict (state_dict()-keyed, fp32, CPU) back into the
        FSDP/TP model — the inverse of ``get_full_state_dict_cpu``.

        Distributes each incoming full tensor onto the placement of the model's
        current state_dict entry, then loads via the model's own
        ``load_state_dict`` (whose FusedQKVLinear hooks re-fuse wq/wk/wv), which
        is fused-QKV-safe unlike DCP ``set_model_state_dict``. ``strict=False``
        tolerates any absent buffers. After the discontinuous theta jump we clear
        the inner optimizer state (stale momentum would bias the next window) but
        leave ``policy_version`` monotone so staleness tracking stays correct.
        """
        train_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        ref_sd = self.model.state_dict()
        to_load = {}
        for k, v in global_sd.items():
            ref = ref_sd.get(k)
            if ref is None:
                continue
            v = v.to(dtype=train_dtype, device=ref.device)
            if isinstance(ref, DTensor):
                v = distribute_tensor(v, ref.device_mesh, ref.placements)
            to_load[k] = v
        self.model.load_state_dict(to_load, strict=False)

        self.optimizers.zero_grad(set_to_none=True)
        for opt in self.optimizers:
            opt.state.clear()
        logger.info(
            "SnapshotPolicyTrainer loaded full theta (policy_version=%d)",
            self.policy_version,
        )
