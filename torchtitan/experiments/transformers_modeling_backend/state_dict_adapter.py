# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class HFTransformersStateDictAdapter(StateDictAdapter):
    """HF <-> native adapter for HFTransformerModel.

    The wrapped modules ARE transformers modules, so native keys equal the HF
    checkpoint keys plus one leading "model." (HFTransformerModel.model is the
    <X>ForCausalLM): "model.model.layers.0.self_attn.q_proj.weight" native vs
    "model.layers.0.self_attn.q_proj.weight" in safetensors. Near-identity —
    no fused-QKV or expert regrouping like the hand-written native adapters.

    Tied embeddings: checkpoints with tie_word_embeddings=true ship no
    lm_head.weight shard. We train untied (FSDP can't shard one weight into
    two groups), so from_hf aliases embed_tokens into lm_head at load; to_hf
    drops lm_head so dcp.load never asks the reader for a missing shard.
    Tie state is read from the CHECKPOINT's config.json (not the model config,
    which the RL registry forces untied).
    """

    def __init__(self, model_config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self._ckpt_ties_embeddings = False
        if hf_assets_path:
            cfg_path = os.path.join(hf_assets_path, "config.json")
            try:
                with open(cfg_path) as f:
                    self._ckpt_ties_embeddings = bool(
                        json.load(f).get("tie_word_embeddings", False)
                    )
            except FileNotFoundError:
                pass

    def to_hf(self, state_dict: dict) -> dict:
        hf_sd = {}
        for key, value in state_dict.items():
            hf_key = key.removeprefix("model.")
            if hf_key == "lm_head.weight" and self._ckpt_ties_embeddings:
                continue  # no shard in a tied checkpoint
            hf_sd[hf_key] = value
        return hf_sd

    def from_hf(self, hf_state_dict: dict) -> dict:
        if (
            self._ckpt_ties_embeddings
            and "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
        ):
            hf_state_dict = dict(hf_state_dict)
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.embed_tokens.weight"
            ]
        return {f"model.{key}": value for key, value in hf_state_dict.items()}
