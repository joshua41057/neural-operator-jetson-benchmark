from __future__ import annotations

from typing import Dict

import torch


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f'Expected tensor with batch dimension and at least 1 feature dim, got shape={tuple(x.shape)}')
    return x.reshape(x.shape[0], -1)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Returns per-sample relative L2 error with shape [B].
    """
    pred_f = _flatten_per_sample(pred)
    target_f = _flatten_per_sample(target)
    num = torch.linalg.norm(pred_f - target_f, dim=1)
    den = torch.linalg.norm(target_f, dim=1).clamp_min(eps)
    return num / den


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = _flatten_per_sample(pred)
    target_f = _flatten_per_sample(target)
    return torch.mean((pred_f - target_f) ** 2, dim=1)


def per_sample_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = _flatten_per_sample(pred)
    target_f = _flatten_per_sample(target)
    return torch.mean(torch.abs(pred_f - target_f), dim=1)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Returns batch-level averages computed from per-sample metrics.
    This is safe to use for progress bars, but epoch summaries should still
    aggregate per-sample values across the full epoch.
    """
    rel = relative_l2_error(pred, target)
    mse = per_sample_mse(pred, target)
    mae = per_sample_mae(pred, target)
    return {
        'rel_l2_mean': float(rel.mean().item()),
        'rel_l2_std': float(rel.std(unbiased=False).item()),
        'mse': float(mse.mean().item()),
        'mae': float(mae.mean().item()),
    }