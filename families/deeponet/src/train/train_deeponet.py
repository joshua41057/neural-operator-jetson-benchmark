from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.datasets import build_dataset_bundle
from src.models.deeponet import build_deeponet_model, count_parameters
from src.train.engine import evaluate_epoch, train_one_epoch
from src.utils.config import load_config, to_plain_dict
from src.utils.device import get_device
from src.utils.io import append_csv_row, ensure_dir, save_json
from src.utils.seed import set_seed


def build_optimizer(cfg, model):
    opt_name = str(cfg.train.optimizer).lower()
    lr = float(cfg.train.lr)
    wd = float(cfg.train.weight_decay)

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def build_scheduler(cfg, optimizer):
    name = str(cfg.train.scheduler).lower()

    if name == "none":
        return None

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=float(cfg.train.min_lr),
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.train.step_size),
            gamma=float(cfg.train.gamma),
        )

    raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg,
    summary: Dict[str, Any],
    x_normalizer,
    y_normalizer,
    best_metric_so_far: float,
):
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": to_plain_dict(cfg),
        "summary": summary,
        "x_normalizer": x_normalizer.to_dict(),
        "y_normalizer": y_normalizer.to_dict(),
        "best_metric_so_far": float(best_metric_so_far),
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.experiment.seed = int(args.seed)

    seed = int(cfg.experiment.seed)
    deterministic = bool(cfg.experiment.deterministic)

    set_seed(seed, deterministic=deterministic)
    device = get_device(args.device)

    bundle = build_dataset_bundle(cfg)

    input_channels = bundle.input_shape[-1]
    output_channels = bundle.output_shape[-1]

    model = build_deeponet_model(
        cfg,
        input_channels=input_channels,
        output_channels=output_channels,
    ).to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    run_dir = ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name / f"seed{seed}")
    history_path = run_dir / "train_history.csv"
    save_json(run_dir / "resolved_config.json", to_plain_dict(cfg))

    y_norm = bundle.y_normalizer.to(device)
    x_norm = bundle.x_normalizer.to(device)

    summary = {
        "config_path": str(cfg._config_path),
        "experiment_name": str(cfg.experiment.name),
        "family": "DeepONet",
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "dataset": bundle.dataset_name,
        "spatial_dim": bundle.spatial_dim,
        "input_shape": list(bundle.input_shape),
        "output_shape": list(bundle.output_shape),
        "parameter_count": int(count_parameters(model)),
        "n_train": int(bundle.n_train),
        "n_val": int(bundle.n_val),
        "n_test": int(bundle.n_test),
        "sensor_resolution": cfg.model.sensor_resolution,
        "basis_dim": int(cfg.model.basis_dim),
        "branch_hidden_dim": int(cfg.model.branch_hidden_dim),
        "branch_layers": int(cfg.model.branch_layers),
        "trunk_hidden_dim": int(cfg.model.trunk_hidden_dim),
        "trunk_layers": int(cfg.model.trunk_layers),
        "use_coord_features": bool(cfg.model.use_coord_features),
        "chunk_size": int(cfg.model.chunk_size),
    }

    print("=== Run summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    best_metric = math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_res = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            device=device,
            y_normalizer=y_norm,
            amp_enabled=bool(cfg.train.amp),
            grad_clip=float(cfg.train.grad_clip) if cfg.train.grad_clip is not None else None,
        )

        val_res = evaluate_epoch(
            model=model,
            loader=bundle.val_loader,
            device=device,
            y_normalizer=y_norm,
            amp_enabled=bool(cfg.train.amp),
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_res.loss,
            "train_rel_l2_mean": train_res.rel_l2_mean,
            "train_rel_l2_std": train_res.rel_l2_std,
            "train_rel_l2_median": train_res.rel_l2_median,
            "train_mse": train_res.mse,
            "train_mae": train_res.mae,
            "train_n": train_res.n_samples,
            "val_loss": val_res.loss,
            "val_rel_l2_mean": val_res.rel_l2_mean,
            "val_rel_l2_std": val_res.rel_l2_std,
            "val_rel_l2_median": val_res.rel_l2_median,
            "val_mse": val_res.mse,
            "val_mae": val_res.mae,
            "val_n": val_res.n_samples,
        }
        append_csv_row(history_path, row)

        print(
            f"Epoch {epoch:04d} | "
            f"lr={current_lr:.3e} | "
            f"train_rel_l2={train_res.rel_l2_mean:.4e}±{train_res.rel_l2_std:.2e} | "
            f"val_rel_l2={val_res.rel_l2_mean:.4e}±{val_res.rel_l2_std:.2e}"
        )

        metric = float(val_res.rel_l2_mean)

        save_checkpoint(
            path=run_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            summary=summary,
            x_normalizer=bundle.x_normalizer,
            y_normalizer=bundle.y_normalizer,
            best_metric_so_far=min(best_metric, metric),
        )

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                path=run_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                summary=summary,
                x_normalizer=bundle.x_normalizer,
                y_normalizer=bundle.y_normalizer,
                best_metric_so_far=best_metric,
            )
        else:
            patience_counter += 1

        if patience_counter >= int(cfg.train.early_stopping_patience):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch={best_epoch}, best val rel-L2={best_metric:.4e})"
            )
            break

    best_ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.to(device)

    test_res = evaluate_epoch(
        model=model,
        loader=bundle.test_loader,
        device=device,
        y_normalizer=y_norm,
        amp_enabled=bool(cfg.train.amp),
    )

    summary.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_rel_l2": float(best_metric),
            "test_rel_l2_mean": float(test_res.rel_l2_mean),
            "test_rel_l2_std": float(test_res.rel_l2_std),
            "test_rel_l2_median": float(test_res.rel_l2_median),
            "test_mse": float(test_res.mse),
            "test_mae": float(test_res.mae),
            "test_n": int(test_res.n_samples),
        }
    )

    save_json(run_dir / "summary.json", summary)

    print("=== Final summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
