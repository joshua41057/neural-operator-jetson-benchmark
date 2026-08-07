from __future__ import annotations

import torch


def add_coords_1d(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, L, C]
    returns: [N, L, C+1]
    """
    if x.ndim != 3:
        raise ValueError(f'Expected x shape [N, L, C], got {tuple(x.shape)}')
    n, l, _ = x.shape
    coords = torch.linspace(0.0, 1.0, steps=l, dtype=x.dtype, device=x.device)
    coords = coords.view(1, l, 1).expand(n, -1, -1)
    return torch.cat([x, coords], dim=-1)


def add_coords_2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, H, W, C]
    returns: [N, H, W, C+2]
    """
    if x.ndim != 4:
        raise ValueError(f'Expected x shape [N, H, W, C], got {tuple(x.shape)}')
    n, h, w, _ = x.shape
    yy = torch.linspace(0.0, 1.0, steps=h, dtype=x.dtype, device=x.device)
    xx = torch.linspace(0.0, 1.0, steps=w, dtype=x.dtype, device=x.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    return torch.cat([x, grid], dim=-1)