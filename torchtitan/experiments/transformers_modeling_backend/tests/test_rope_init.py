# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RoPE's inv_freq must survive the meta build -> to_empty -> init_weights path.

The buffer is computed in the rotary module's __init__, so on the meta device it
is empty and to_empty() leaves it uninitialized; init_weights has to recompute
it, or the model trains with no positional information.
"""

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from torchtitan.experiments.transformers_modeling_backend.config_registry import (
    transformers_modeling_backend_debugmodel,
)


def test_inv_freq_is_recomputed_after_to_empty(tmp_path):
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp_path)
    job = transformers_modeling_backend_debugmodel(seq_len=256)
    job.hf_model = str(tmp_path)
    job.model.update_from_config(config=job)
    with torch.device("meta"):
        model = job.model.build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cpu"))

    rotary = next(m for m in model.modules() if isinstance(m, Qwen3RotaryEmbedding))
    expected = Qwen3RotaryEmbedding(rotary.config).inv_freq  # what HF computes
    torch.testing.assert_close(rotary.inv_freq, expected)
    torch.testing.assert_close(rotary.original_inv_freq, expected)
