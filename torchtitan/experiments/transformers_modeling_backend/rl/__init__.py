# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# RL glue for the HF transformers modeling backend: a ModelSpec factory whose
# result satisfies the async-RL contract (varlen packed attention, HF weight
# loading via a near-identity state-dict adapter, titan-shaped layers view) so
# the RL trainer, the vLLM generator wrapper, and torchft consume it exactly
# like a native model spec. Mirrors the pretraining glue pattern: compose at
# the ModelSpec seam, no RL-consumer edits.

from transformers import AutoConfig

from torchtitan.protocols.model_spec import ModelSpec

from .. import (
    HFTransformerModel,
    parallelize_hf_transformers,
    pipeline_hf_transformers,
    TitanDenseModelConfig,
)
from ..state_dict_adapter import HFTransformersStateDictAdapter
from .attention import VARLEN_TT_IMPL


class _RegistryResolveConfig:
    """Minimal update_from_config carrier: hf_model rides on the model Config
    itself (set below), and training/parallelism/debug are optional there."""


def model_registry(
    flavor: str,
    *,
    attn_backend: str = "varlen",
    hf_assets_path: str = "",
    max_seq_len: int = 2048,
) -> ModelSpec:
    """RL-capable ModelSpec for an HF-architecture policy.

    The architecture comes from `hf_assets_path` (a local HF checkpoint dir
    with config.json + safetensors — the same dir the trainer initial-loads
    from and vLLM reads tokenizer assets from). `flavor` is informational.
    """
    if attn_backend != "varlen":
        raise ValueError(
            f"hf backend RL supports attn_backend='varlen' only, got {attn_backend!r}"
        )
    if not hf_assets_path:
        raise ValueError(
            "hf backend needs hf_assets_path (local dir with config.json)"
        )

    hf_cfg = AutoConfig.from_pretrained(hf_assets_path)
    # Mirror the checkpoint's dims so _titan_injected_model_args re-application
    # inside update_from_config is a no-op instead of clobbering real values.
    cfg = HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(
            dim=hf_cfg.hidden_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_heads=hf_cfg.num_attention_heads,
            n_kv_heads=getattr(hf_cfg, "num_key_value_heads", None)
            or hf_cfg.num_attention_heads,
            vocab_size=hf_cfg.vocab_size,
            norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
            rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
            max_seq_len=max_seq_len,
        ),
        attn_implementation=VARLEN_TT_IMPL,
    )
    cfg.hf_model = hf_assets_path  # AutoConfig source for every resolve site
    cfg._titan_injected_model_args["hf_model"] = hf_assets_path
    # FSDP cannot shard one weight into two groups; both RL roles host the
    # same untied model and the adapter aliases embeddings into lm_head at
    # load, so step-0 logprobs match the tied checkpoint.
    cfg._titan_injected_model_args["tie_word_embeddings"] = False
    # Resolve fully at registry time (architectures, vocab, dims) so the
    # generator's config parser and the layers shim are correct before any
    # trainer-side update_from_config runs.
    cfg.update_from_config(config=_RegistryResolveConfig())

    return ModelSpec(
        name="hf_transformers_rl",
        flavor=flavor,
        model=cfg,
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        post_optimizer_build_fn=None,
        state_dict_adapter=HFTransformersStateDictAdapter,
    )
