#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path("configs/deeponet")
RES_ROOT = ROOT / "resolution"
ROOT.mkdir(parents=True, exist_ok=True)
RES_ROOT.mkdir(parents=True, exist_ok=True)


COMMON_TRAIN = """\
train:
  batch_size: {batch_size}
  epochs: {epochs}
  optimizer: adamw
  lr: {lr}
  min_lr: 1.0e-6
  weight_decay: {weight_decay}
  scheduler: cosine
  step_size: 100
  gamma: 0.5
  amp: false
  grad_clip: 1.0
  early_stopping_patience: {patience}
"""


def write(path: Path, s: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")
    print(f"wrote {path}")


def base_config(
    *,
    name: str,
    dataset: str,
    spatial_dim: int,
    resolution: int,
    scale: str,
    seed: int = 0,
):
    if dataset == "burgers":
        data_block = f"""\
data:
  dataset: burgers
  raw_path: data/raw/burgers_data_R10.mat
  split_path: splits/burgers_split.json
  resolution: {resolution}
  normalize_x: true
  normalize_y: true
  num_workers: 2
"""
        batch_size = 16
        epochs = 500
        lr = "1.0e-3"
        weight_decay = "1.0e-5"
        patience = 80

        if scale == "small":
            sensor = max(64, resolution // 16)
            basis = 64
            bh = 128
            th = 128
            bl = 3
            tl = 3
        elif scale == "base":
            sensor = max(128, resolution // 8)
            basis = 128
            bh = 256
            th = 256
            bl = 4
            tl = 4
        elif scale == "large":
            sensor = max(256, resolution // 4)
            basis = 256
            bh = 512
            th = 512
            bl = 5
            tl = 5
        else:
            raise ValueError(scale)

        chunk_size = 0

    elif dataset == "darcy":
        data_block = f"""\
data:
  dataset: darcy
  train_path: data/raw/piececonst_r421_N1024_smooth1.mat
  test_path: data/raw/piececonst_r421_N1024_smooth2.mat
  split_path: splits/darcy_smooth1_split.json
  resolution: {resolution}
  normalize_x: true
  normalize_y: true
  num_workers: 2
"""
        batch_size = 4
        epochs = 700
        lr = "7.5e-4"
        weight_decay = "1.0e-5"
        patience = 100

        if scale == "small":
            sensor = max(17, resolution // 6)
            basis = 64
            bh = 256
            th = 128
            bl = 3
            tl = 3
        elif scale == "base":
            sensor = max(25, resolution // 4)
            basis = 128
            bh = 512
            th = 256
            bl = 4
            tl = 4
        elif scale == "large":
            sensor = max(35, resolution // 3)
            basis = 256
            bh = 768
            th = 384
            bl = 5
            tl = 5
        else:
            raise ValueError(scale)

        # Training memory safety for high-res 2D trunk evaluation.
        # For deployment, this still produces full-field output.
        chunk_size = 32768

    else:
        raise ValueError(dataset)

    model_block = f"""\
model:
  family: deeponet
  spatial_dim: {spatial_dim}
  sensor_resolution: {sensor}
  basis_dim: {basis}
  branch_hidden_dim: {bh}
  branch_layers: {bl}
  trunk_hidden_dim: {th}
  trunk_layers: {tl}
  activation: gelu
  dropout: 0.0
  use_coord_features: true
  chunk_size: {chunk_size}
"""

    train_block = COMMON_TRAIN.format(
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    cfg = f"""\
experiment:
  name: {name}
  seed: {seed}
  deterministic: true
  output_dir: checkpoints

{data_block}
{model_block}
{train_block}
"""
    return cfg


def main():
    # Controlled model-scale configs.
    for dataset, spatial_dim, res in [
        ("burgers", 1, 2048),
        ("darcy", 2, 141),
    ]:
        for scale in ["small", "base", "large"]:
            name = f"{dataset}_deeponet_{scale}"
            write(ROOT / f"{name}.yaml", base_config(
                name=name,
                dataset=dataset,
                spatial_dim=spatial_dim,
                resolution=res,
                scale=scale,
            ))

    # Resolution-scaling configs.
    for res in [512, 1024, 2048, 4096]:
        name = f"burgers_deeponet_base_r{res}"
        write(RES_ROOT / f"{name}.yaml", base_config(
            name=name,
            dataset="burgers",
            spatial_dim=1,
            resolution=res,
            scale="base",
        ))

    for res in [85, 141, 211, 281]:
        name = f"darcy_deeponet_base_r{res}"
        write(RES_ROOT / f"{name}.yaml", base_config(
            name=name,
            dataset="darcy",
            spatial_dim=2,
            resolution=res,
            scale="base",
        ))


if __name__ == "__main__":
    main()
