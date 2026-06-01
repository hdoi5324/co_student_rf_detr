"""Color-only view augmentation for Co-Student (geometry shared across branches)."""

from __future__ import annotations

import torch


def color_jitter_batch(tensors: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply mild color jitter on normalized CHW tensors (B, C, H, W)."""
    if strength <= 0:
        return tensors

    out = tensors
    b = tensors.shape[0]
    device = tensors.device
    brightness = 1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * strength
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * strength
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean
    out = out * brightness

    saturation = 1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * strength
    gray = out.mean(dim=1, keepdim=True)
    out = gray + (out - gray) * saturation

    noise = torch.randn_like(out) * (0.02 * strength)
    return out + noise
