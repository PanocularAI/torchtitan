# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# A transformers AttentionInterface entry backed by torchtitan's own
# VarlenAttention, so an HF-wrapped model trains on packed RL batches with the
# SAME flash-varlen kernel as the native qwen3/llama3 RL trainer path (per-
# document isolation + RoPE restarts driven by cu_seqlens from `positions`).
# The name is deliberately absent from transformers' mask registry, so
# create_causal_mask returns None and the kernel owns causality.

import torch

from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata

VARLEN_TT_IMPL = "varlen_torchtitan"

_varlen_singleton: VarlenAttention | None = None


def _varlen() -> VarlenAttention:
    # Lazy: the ctor probes CUDA capability (FA3 activation on Hopper).
    global _varlen_singleton
    if _varlen_singleton is None:
        _varlen_singleton = VarlenAttention(VarlenAttention.Config())
    return _varlen_singleton


def titan_varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    **kwargs,
):
    assert cu_seq_lens_q is not None, (
        "varlen_torchtitan needs cu_seqlens: call the model via "
        "HFRLModel.forward(tokens, attention_masks=<VarlenMetadata>, positions=...)"
    )
    # transformers hands q/k/v as [B, H, S, D]; VarlenAttention expects [B, S, H, D].
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    masks = VarlenMetadata(
        cu_seq_q=cu_seq_lens_q,
        cu_seq_k=cu_seq_lens_k,
        max_q=max_length_q,
        max_k=max_length_k,
    )
    out = _varlen()(
        q,
        k,
        v,
        attention_masks=masks,
        scale=scaling,
        enable_gqa=q.shape[2] != k.shape[2],
    )
    # [B, S, H, D]; callers reshape to [B, S, H*D]. No attention weights.
    return out, None
