# torchtitan/experiments/ft/diloco/outer_optimizer.py

import torch

from torchtitan.experiments.ft.diloco.heloco_optimizer import (
    HeLoCoOuterOptimizer,
    MomentumLookAhead,
)


def build_outer_optimizer(params, ft_config):
    """
    Build the DiLoCo outer optimizer.

    Supported options:
        nesterov : original TorchTitan DiLoCo outer optimizer
        mla      : momentum look-ahead style outer optimizer
        heloco   : tensor-wise direction-corrected outer optimizer
    """

    name = getattr(ft_config, "outer_optimizer", "nesterov").lower()

    lr = getattr(ft_config, "outer_lr", 0.7)
    momentum = getattr(ft_config, "outer_momentum", 0.9)

    if name == "nesterov":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )

    if name == "mla":
        return MomentumLookAhead(
            params,
            lr=lr,
            momentum=momentum,
        )

    if name == "heloco":
        return HeLoCoOuterOptimizer(
            params,
            lr=lr,
            momentum=momentum,
            cos_ok=getattr(ft_config, "outer_cos_ok", 0.2),
            correction_strength=getattr(
                ft_config,
                "outer_correction_strength",
                0.5,
            ),
            eps=getattr(ft_config, "outer_eps", 1e-8),
        )

    raise ValueError(f"Unsupported DiLoCo outer optimizer: {name}")