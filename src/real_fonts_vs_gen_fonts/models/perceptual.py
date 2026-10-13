"""Perceptual loss networks — pluggable backend for diffusion training.

Each class wraps a frozen pretrained network and exposes:
    forward(x_pred, x_real) -> scalar Tensor

Inputs must be float32 in [0, 1] with shape (B, C, H, W).
Callers are responsible for 1ch → 3ch conversion before calling forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LPIPSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import piq
        self.net = piq.LPIPS()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x_pred: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        return self.net(x_pred, x_real).mean()


class _VGG16Loss(nn.Module):
    """VGG16 feature matching loss using relu2_2 activations."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:9]  # up to relu2_2
        for p in self.features.parameters():
            p.requires_grad_(False)

    def forward(self, x_pred: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.features(x_pred), self.features(x_real))


_REGISTRY: dict[str, type[nn.Module]] = {
    "lpips": _LPIPSLoss,
    "vgg16": _VGG16Loss,
}


def build_perceptual_loss(name: str) -> nn.Module:
    """Factory that instantiates and freezes a perceptual loss network by name.

    To add a new backend, register a class in ``_REGISTRY``.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown perceptual network '{name}'. Choose from: {sorted(_REGISTRY)}")
    return _REGISTRY[name]().eval()
