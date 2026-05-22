# torchtitan/experiments/ft/diloco/heloco_optimizer.py

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class MomentumLookAhead(Optimizer):
    """
    MLA-style DiLoCo outer optimizer.

    Expected input:
        p.grad = DiLoCo pseudo-gradient

    This is useful as a clean baseline against HeLoCo.

    Update:
        m <- momentum * m + (1 - momentum) * grad
        p <- p - lr * (grad + momentum * m)
    """

    def __init__(self, params, lr: float = 0.7, momentum: float = 0.9):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
        }
        super().__init__(params, defaults)

        self.diagnostics = {
            "last_cos": math.nan,
            "last_alpha": 0.0,
            "last_active_rate": 0.0,
            "last_correction_ratio": 0.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if grad.is_sparse:
                    raise RuntimeError("MomentumLookAhead does not support sparse gradients.")

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["momentum_buffer"]

                m.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                # MLA outer update.
                p.add_(grad + momentum * m, alpha=-lr)

        return loss


class HeLoCoOuterOptimizer(Optimizer):
    """
    HeLoCo outer optimizer for DiLoCo.

    Expected input:
        p.grad = DiLoCo pseudo-gradient from TorchFT DiLoCo.

    Main idea:
        For each tensor, compare the incoming pseudo-gradient with the
        current outer momentum buffer.

        If the pseudo-gradient is well aligned with momentum, keep it.

        If it is poorly aligned, softly rotate its direction toward the
        momentum direction while preserving its original norm.

    This version is intentionally simple and has only one main correction
    hyperparameter: correction_strength.
    """

    def __init__(
        self,
        params,
        lr: float = 0.7,
        momentum: float = 0.9,
        cos_ok: float = 0.2,
        correction_strength: float = 0.5,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not -1.0 <= cos_ok <= 1.0:
            raise ValueError(f"Invalid cos_ok: {cos_ok}")
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError(f"Invalid correction_strength: {correction_strength}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "cos_ok": cos_ok,
            "correction_strength": correction_strength,
            "eps": eps,
        }

        super().__init__(params, defaults)

        self.diagnostics = {
            "last_cos": math.nan,
            "last_alpha": math.nan,
            "last_active_rate": math.nan,
            "last_correction_ratio": math.nan,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        cos_values = []
        alpha_values = []
        correction_ratios = []

        active_count = 0
        total_count = 0

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            cos_ok = group["cos_ok"]
            correction_strength = group["correction_strength"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if grad.is_sparse:
                    raise RuntimeError("HeLoCoOuterOptimizer does not support sparse gradients.")

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["momentum_buffer"]

                total_count += 1

                grad_f = grad.float()
                m_f = m.float()

                grad_norm = torch.linalg.vector_norm(grad_f).item()
                m_norm = torch.linalg.vector_norm(m_f).item()

                # First step or tiny tensor: no reliable momentum direction yet.
                if grad_norm < eps or m_norm < eps:
                    cos = 1.0
                    alpha = 0.0
                    corrected_grad = grad
                else:
                    dot = torch.sum(grad_f * m_f).item()
                    cos = dot / (grad_norm * m_norm + eps)
                    cos = max(-1.0, min(1.0, float(cos)))

                    if cos >= cos_ok:
                        # Already aligned enough.
                        alpha = 0.0
                        corrected_grad = grad
                    else:
                        # Poor alignment.
                        #
                        # gap = 0 when cos == cos_ok
                        # gap = 1 when cos == -1
                        #
                        # alpha controls how much we rotate toward momentum.
                        gap = (cos_ok - cos) / (cos_ok + 1.0 + eps)
                        gap = max(0.0, min(1.0, float(gap)))

                        alpha = correction_strength * gap
                        alpha = max(0.0, min(1.0, float(alpha)))

                        grad_dir = grad / (grad_norm + eps)
                        mom_dir = m / (m_norm + eps)

                        mixed_dir = (1.0 - alpha) * grad_dir + alpha * mom_dir
                        mixed_norm = torch.linalg.vector_norm(mixed_dir.float()).item()

                        if mixed_norm < eps:
                            corrected_grad = grad
                        else:
                            # Preserve original pseudo-gradient norm.
                            corrected_grad = mixed_dir / (mixed_norm + eps) * grad_norm

                if grad_norm < eps:
                    correction_ratio = 0.0
                else:
                    correction_ratio = (
                        torch.linalg.vector_norm((corrected_grad - grad).float()).item()
                        / (grad_norm + eps)
                    )

                if alpha > 0.0:
                    active_count += 1

                # Update outer momentum using corrected pseudo-gradient.
                m.mul_(momentum).add_(corrected_grad, alpha=1.0 - momentum)

                # MLA-style outer update using corrected pseudo-gradient.
                p.add_(corrected_grad + momentum * m, alpha=-lr)

                cos_values.append(float(cos))
                alpha_values.append(float(alpha))
                correction_ratios.append(float(correction_ratio))

        if total_count > 0:
            self.diagnostics = {
                "last_cos": sum(cos_values) / len(cos_values),
                "last_alpha": sum(alpha_values) / len(alpha_values),
                "last_active_rate": active_count / total_count,
                "last_correction_ratio": sum(correction_ratios) / len(correction_ratios),
            }

        return loss