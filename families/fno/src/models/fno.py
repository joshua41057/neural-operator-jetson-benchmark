from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import SpectralConv1d, SpectralConv2d


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral(x) + self.pointwise(x)


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral(x) + self.pointwise(x)


class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        projection_channels: int,
        n_layers: int,
        modes: int,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.modes = modes
        self.padding = padding

        self.lift = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_channels, modes) for _ in range(n_layers)])
        self.proj1 = nn.Conv1d(hidden_channels, projection_channels, kernel_size=1)
        self.proj2 = nn.Conv1d(projection_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x = self.lift(x)
        x = x.permute(0, 2, 1)  # [B, C, L]
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
        for block in self.blocks:
            x = block(x)
            x = F.gelu(x)
        if self.padding > 0:
            x = x[..., :-self.padding]
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        x = x.permute(0, 2, 1)
        return x


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        projection_channels: int,
        n_layers: int,
        modes1: int,
        modes2: int,
        padding: Sequence[int] | int = 0,
    ):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.padding = tuple(int(p) for p in padding)

        self.lift = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, modes1, modes2) for _ in range(n_layers)
        ])
        self.proj1 = nn.Conv2d(hidden_channels, projection_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(projection_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        pad_h, pad_w = self.padding
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        for block in self.blocks:
            x = block(x)
            x = F.gelu(x)
        if pad_h > 0 or pad_w > 0:
            x = x[..., :x.shape[-2] - pad_h, :x.shape[-1] - pad_w]
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        x = x.permute(0, 2, 3, 1)
        return x


def build_fno_model(cfg, input_channels: int, output_channels: int) -> nn.Module:
    spatial_dim = int(cfg.model.spatial_dim)
    if spatial_dim == 1:
        return FNO1d(
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_channels=int(cfg.model.hidden_channels),
            projection_channels=int(cfg.model.projection_channels),
            n_layers=int(cfg.model.n_layers),
            modes=int(cfg.model.modes[0]),
            padding=int(cfg.model.padding[0]),
        )
    if spatial_dim == 2:
        return FNO2d(
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_channels=int(cfg.model.hidden_channels),
            projection_channels=int(cfg.model.projection_channels),
            n_layers=int(cfg.model.n_layers),
            modes1=int(cfg.model.modes[0]),
            modes2=int(cfg.model.modes[1]),
            padding=tuple(cfg.model.padding),
        )
    raise ValueError(f'Unsupported spatial_dim={cfg.model.spatial_dim}')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
