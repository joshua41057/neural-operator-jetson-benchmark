from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.metrics import batch_metrics, per_sample_mae, per_sample_mse, relative_l2_error


@dataclass
class EpochResult:
    loss: float
    rel_l2_mean: float
    rel_l2_std: float
    rel_l2_median: float
    mse: float
    mae: float
    n_samples: int


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return relative_l2_error(pred, target).mean()


def _finalize_epoch_stats(
    losses: List[float],
    rel_all: List[torch.Tensor],
    mse_all: List[torch.Tensor],
    mae_all: List[torch.Tensor],
) -> EpochResult:
    if len(rel_all) == 0:
        return EpochResult(
            loss=0.0,
            rel_l2_mean=0.0,
            rel_l2_std=0.0,
            rel_l2_median=0.0,
            mse=0.0,
            mae=0.0,
            n_samples=0,
        )

    rel = torch.cat(rel_all, dim=0)
    mse = torch.cat(mse_all, dim=0)
    mae = torch.cat(mae_all, dim=0)

    return EpochResult(
        loss=float(sum(losses) / max(len(losses), 1)),
        rel_l2_mean=float(rel.mean().item()),
        rel_l2_std=float(rel.std(unbiased=False).item()),
        rel_l2_median=float(rel.median().item()),
        mse=float(mse.mean().item()),
        mae=float(mae.mean().item()),
        n_samples=int(rel.numel()),
    )


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    y_normalizer=None,
    amp_enabled: bool = False,
) -> EpochResult:
    model.eval()
    losses: List[float] = []
    rel_all: List[torch.Tensor] = []
    mse_all: List[torch.Tensor] = []
    mae_all: List[torch.Tensor] = []

    use_amp = bool(amp_enabled) and device.type == 'cuda'

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            pred = model(x)
            loss = relative_l2_loss(pred, y)

        losses.append(float(loss.item()))

        pred_eval = y_normalizer.decode(pred) if y_normalizer is not None else pred
        y_eval = y_normalizer.decode(y) if y_normalizer is not None else y

        rel_all.append(relative_l2_error(pred_eval, y_eval).detach().cpu())
        mse_all.append(per_sample_mse(pred_eval, y_eval).detach().cpu())
        mae_all.append(per_sample_mae(pred_eval, y_eval).detach().cpu())

    return _finalize_epoch_stats(losses, rel_all, mse_all, mae_all)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_normalizer=None,
    amp_enabled: bool = False,
    grad_clip: float | None = None,
) -> EpochResult:
    model.train()

    use_amp = bool(amp_enabled) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    losses: List[float] = []
    rel_all: List[torch.Tensor] = []
    mse_all: List[torch.Tensor] = []
    mae_all: List[torch.Tensor] = []

    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            pred = model(x)
            loss = relative_l2_loss(pred, y)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))

        with torch.no_grad():
            pred_eval = y_normalizer.decode(pred) if y_normalizer is not None else pred
            y_eval = y_normalizer.decode(y) if y_normalizer is not None else y

            rel_all.append(relative_l2_error(pred_eval, y_eval).detach().cpu())
            mse_all.append(per_sample_mse(pred_eval, y_eval).detach().cpu())
            mae_all.append(per_sample_mae(pred_eval, y_eval).detach().cpu())

        pbar.set_postfix(loss=f'{loss.item():.4e}')

    return _finalize_epoch_stats(losses, rel_all, mse_all, mae_all)