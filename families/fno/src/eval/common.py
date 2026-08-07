from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.fno import build_fno_model
from src.utils.normalizer import UnitGaussianNormalizer


class RawInputInferenceWrapper(nn.Module):
    """
    Wrapper that accepts raw physical input fields and returns de-normalized predictions.

    1D input shape: [B, L]
    2D input shape: [B, H, W]
    """

    def __init__(
        self,
        model: nn.Module,
        x_normalizer: UnitGaussianNormalizer,
        y_normalizer: UnitGaussianNormalizer,
        add_coords: bool,
        spatial_dim: int,
    ):
        super().__init__()
        self.model = model
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.add_coords = bool(add_coords)
        self.spatial_dim = int(spatial_dim)

    def _coords_1d(self, length: int, x: torch.Tensor):
        c = torch.linspace(0.0, 1.0, steps=length, device=x.device, dtype=x.dtype)
        return c.view(1, length, 1).expand(x.shape[0], -1, -1)

    def _coords_2d(self, height: int, width: int, x: torch.Tensor):
        ys = torch.linspace(0.0, 1.0, steps=height, device=x.device, dtype=x.dtype)
        xs = torch.linspace(0.0, 1.0, steps=width, device=x.device, dtype=x.dtype)
        yy = ys.view(1, height, 1, 1).expand(x.shape[0], height, width, 1)
        xx = xs.view(1, 1, width, 1).expand(x.shape[0], height, width, 1)
        return torch.cat([yy, xx], dim=-1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            x = x_raw.unsqueeze(-1)
            x = self.x_normalizer.encode(x)
            if self.add_coords:
                x = torch.cat([x, self._coords_1d(x.shape[1], x)], dim=-1)
            y = self.model(x)
            y = self.y_normalizer.decode(y)
            return y.squeeze(-1)
        elif self.spatial_dim == 2:
            x = x_raw.unsqueeze(-1)
            x = self.x_normalizer.encode(x)
            if self.add_coords:
                x = torch.cat([x, self._coords_2d(x.shape[1], x.shape[2], x)], dim=-1)
            y = self.model(x)
            y = self.y_normalizer.decode(y)
            return y.squeeze(-1)
        else:
            raise RuntimeError('Unsupported spatial dimension')


def load_model_and_normalizers(checkpoint_path: str, map_location='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    cfg = ckpt['config']

    if isinstance(cfg, dict):
        class Obj(dict):
            def __getattr__(self, k):
                v = self[k]
                if isinstance(v, dict):
                    v = Obj(v)
                    self[k] = v
                return v
        cfg = Obj(cfg)

    add_coords = bool(cfg['data']['add_coords'] if isinstance(cfg, dict) else cfg.data.add_coords)
    spatial_dim = int(cfg['model']['spatial_dim'] if isinstance(cfg, dict) else cfg.model.spatial_dim)

    x_mean = ckpt['x_normalizer']['mean']
    x_std = ckpt['x_normalizer']['std']
    y_mean = ckpt['y_normalizer']['mean']
    y_std = ckpt['y_normalizer']['std']
    x_norm = UnitGaussianNormalizer(x_mean, x_std, eps=float(ckpt['x_normalizer'].get('eps', 1e-6)))
    y_norm = UnitGaussianNormalizer(y_mean, y_std, eps=float(ckpt['y_normalizer'].get('eps', 1e-6)))

    input_channels = 1 + (1 if spatial_dim == 1 and add_coords else 0) + (2 if spatial_dim == 2 and add_coords else 0)
    output_channels = 1
    model = build_fno_model(cfg, input_channels=input_channels, output_channels=output_channels)
    model.load_state_dict(ckpt['model_state'])
    wrapper = RawInputInferenceWrapper(model, x_norm, y_norm, add_coords=add_coords, spatial_dim=spatial_dim)
    return ckpt, cfg, model, x_norm, y_norm, wrapper
