# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
from dataclasses import dataclass

from transformers import PretrainedConfig

from torchtitan.protocols.model_spec import ModelSpec

# transformers >= 5 turned PretrainedConfig into a dataclass with positional
# fields; torchtitan's Configurable.__init_subclass__ rejects any inherited
# non-kw-only field when HFTransformerModel.Config (which subclasses both
# Configurable.Config and PretrainedConfig) is defined below.
#
# Field.kw_only is only consulted when @dataclass GENERATES an __init__ for a
# class -- PretrainedConfig's own __init__ was already compiled by
# transformers at its import time, so flipping the flag here can't change how
# a bare PretrainedConfig(...) is constructed (directly, or via AutoConfig for
# any HF architecture class). It only affects dataclasses that inherit these
# fields and get *their* __init__ generated after this point -- exactly
# HFTransformerModel.Config, imported next. So this flip is permanent and
# global by construction, not a scoped patch: there is no window where it
# needs to be, or safely can be, reverted.
for _f in dataclasses.fields(PretrainedConfig):
    _f.kw_only = True
del _f

from .model import HFTransformerModel  # noqa: E402

from .parallelize import parallelize_hf_transformers  # noqa: E402
from .pipeline import pipeline_hf_transformers  # noqa: E402

__all__ = [
    "HFTransformerModel",
]


@dataclass
class TitanDenseModelConfig:
    """Arguments for the base TorchTitan model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"


flavors = {
    "debugmodel": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    # full = the hf_model repo's REAL dims: without inject_titan_dims=False the
    # default TitanDenseModelConfig dims would be re-applied over the repo's
    # config.json (e.g. n_layers=32 onto a 28-layer checkpoint) and the model
    # build IndexErrors on the derived layer_types.
    "full": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(),
        inject_titan_dims=False,
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="transformers_modeling_backend",
        flavor=flavor,
        model=flavors[flavor],
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
