# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, CommConfig, TrainingConfig
from torchtitan.experiments.ft.checkpoint import FTCheckpointManager
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler

from . import model_registry


def _llama3_ft_debugmodel_with_outer_optimizer(
    outer_optimizer: str,
    outer_lr: float,
    outer_momentum: float,
    outer_cos_ok: float = 0.2,
    outer_correction_strength: float = 0.5,
    outer_eps: float = 1e-8,
) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        optimizer=FTOptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=100,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(),
        checkpoint=FTCheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        comm=CommConfig(train_timeout_seconds=15),
        fault_tolerance=FaultTolerance(
            enable=True,
            semi_sync_method="diloco",
            process_group="nccl",
            process_group_timeout_ms=10000,
            sync_steps=10,
            num_fragments=2,

            # Added for custom DiLoCo outer optimizer
            # Options:
            #   "nesterov" -> original TorchTitan DiLoCo outer optimizer
            #   "mla"      -> MLA baseline
            #   "heloco"   -> HeLoCo outer optimizer
            outer_optimizer=outer_optimizer,
            outer_lr=outer_lr,
            outer_momentum=outer_momentum,

            # Used only when outer_optimizer="heloco"
            outer_cos_ok=outer_cos_ok,
            outer_correction_strength=outer_correction_strength,
            outer_eps=outer_eps,
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def llama3_ft_debugmodel_nesterov() -> FaultTolerantTrainer.Config:
    """
    Original TorchTitan DiLoCo baseline.
    Outer optimizer: Nesterov SGD.
    """
    return _llama3_ft_debugmodel_with_outer_optimizer(
        outer_optimizer="nesterov",
        outer_lr=0.7,
        outer_momentum=0.9,
    )


def llama3_ft_debugmodel_mla() -> FaultTolerantTrainer.Config:
    """
    MLA baseline.
    Outer optimizer: Momentum Look-Ahead.
    """
    return _llama3_ft_debugmodel_with_outer_optimizer(
        outer_optimizer="mla",
        outer_lr=0.7,
        outer_momentum=0.9,
    )


def llama3_ft_debugmodel_heloco() -> FaultTolerantTrainer.Config:
    """
    HeLoCo method.
    Outer optimizer: tensor-wise direction-corrected outer optimizer.
    """
    return _llama3_ft_debugmodel_with_outer_optimizer(
        outer_optimizer="heloco",
        outer_lr=0.7,
        outer_momentum=0.9,
        outer_cos_ok=0.2,
        outer_correction_strength=0.5,
        outer_eps=1e-8,
    )


def llama3_ft_debugmodel() -> FaultTolerantTrainer.Config:
    """
    Default debug config.

    During HeLoCo development, this points to HeLoCo.
    Change this to llama3_ft_debugmodel_nesterov() if you want original
    TorchTitan behavior as default.
    """
    return llama3_ft_debugmodel_heloco()






# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# from torchtitan.components.lr_scheduler import LRSchedulersContainer
# from torchtitan.components.metrics import MetricsProcessor
# from torchtitan.components.validate import Validator
# from torchtitan.config import ActivationCheckpointConfig, CommConfig, TrainingConfig
# from torchtitan.experiments.ft.checkpoint import FTCheckpointManager
# from torchtitan.experiments.ft.config.job_config import FaultTolerance
# from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
# from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
# from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
# from torchtitan.tools.profiler import Profiler

# from . import model_registry


# def llama3_ft_debugmodel() -> FaultTolerantTrainer.Config:
#     return FaultTolerantTrainer.Config(
#         hf_assets_path="./tests/assets/tokenizer",
#         profiler=Profiler.Config(
#             enable_profiling=True,
#             profile_freq=10,
#             profiler_active=10,
#             profiler_warmup=0,
#         ),
#         metrics=MetricsProcessor.Config(log_freq=1),
#         model_spec=model_registry("debugmodel"),
#         optimizer=FTOptimizersContainer.Config(lr=8e-4),
#         lr_scheduler=LRSchedulersContainer.Config(
#             warmup_steps=2,
#             decay_ratio=0.8,
#             decay_type="linear",
#             min_lr_factor=0.0,
#         ),
#         training=TrainingConfig(
#             local_batch_size=8,
#             seq_len=2048,
#             steps=100,
#         ),
#         dataloader=HuggingFaceTextDataLoader.Config(),
#         checkpoint=FTCheckpointManager.Config(
#             interval=10,
#             last_save_model_only=False,
#         ),
#         activation_checkpoint=ActivationCheckpointConfig(
#             mode="selective",
#         ),
#         comm=CommConfig(train_timeout_seconds=15),
#         fault_tolerance=FaultTolerance(
#             enable=True,
#             semi_sync_method="diloco",
#             process_group="nccl",
#             process_group_timeout_ms=10000,
#             sync_steps=10,
#             num_fragments=2,
#         ),
#         validator=Validator.Config(
#             freq=5,
#             steps=10,
#         ),
#     )
