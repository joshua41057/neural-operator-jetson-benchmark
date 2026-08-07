from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


class UnitGaussianNormalizer(nn.Module):
    """
    Dataset-level normalizer using train-split statistics.
    Statistics are computed over all dimensions except the last channel dim.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(f'mean/std must be 1D channel vectors, got mean={mean.shape}, std={std.shape}')
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float())
        self.eps = float(eps)

    @classmethod
    def from_data(cls, x: torch.Tensor, eps: float = 1e-6) -> 'UnitGaussianNormalizer':
        if x.ndim < 2:
            raise ValueError(f'Expected x with shape [N, ..., C], got {tuple(x.shape)}')
        x_flat = x.reshape(-1, x.shape[-1])
        mean = x_flat.mean(dim=0)
        std = x_flat.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(mean=mean, std=std, eps=eps)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean': self.mean.detach().cpu(),
            'std': self.std.detach().cpu(),
            'eps': self.eps,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> 'UnitGaussianNormalizer':
        return cls(
            mean=state['mean'].float(),
            std=state['std'].float(),
            eps=float(state.get('eps', 1e-6)),
        )

    def extra_repr(self) -> str:
        return f'channels={self.mean.numel()}, eps={self.eps}'


@dataclass
class IdentityNormalizer:
    """
    Convenience object with the same interface as UnitGaussianNormalizer.
    Useful for coordinates or already-normalized data.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device: torch.device | str):
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'identity'}

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> 'IdentityNormalizer':
        return cls()