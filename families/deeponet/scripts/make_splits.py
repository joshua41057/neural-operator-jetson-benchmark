#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='splits')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--burgers-n', type=int, default=2048)
    parser.add_argument('--darcy-trainval-n', type=int, default=1024)
    parser.add_argument('--burgers-train-frac', type=float, default=0.8)
    parser.add_argument('--burgers-val-frac', type=float, default=0.1)
    parser.add_argument('--darcy-val-frac', type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Burgers: fixed train/val/test split
    burgers_idx = np.arange(args.burgers_n, dtype=np.int64)
    rng.shuffle(burgers_idx)

    n_train = int(round(args.burgers_n * args.burgers_train_frac))
    n_val = int(round(args.burgers_n * args.burgers_val_frac))
    n_train = min(n_train, args.burgers_n - 2)
    n_val = min(n_val, args.burgers_n - n_train - 1)

    burgers_split = {
        'train': burgers_idx[:n_train].tolist(),
        'val': burgers_idx[n_train:n_train + n_val].tolist(),
        'test': burgers_idx[n_train + n_val:].tolist(),
    }
    save_json(out_dir / 'burgers_split.json', burgers_split)

    # Darcy: fixed train/val split for smooth1; smooth2 remains full held-out test
    darcy_idx = np.arange(args.darcy_trainval_n, dtype=np.int64)
    rng.shuffle(darcy_idx)

    n_val_darcy = int(round(args.darcy_trainval_n * args.darcy_val_frac))
    n_val_darcy = max(1, min(n_val_darcy, args.darcy_trainval_n - 1))
    n_train_darcy = args.darcy_trainval_n - n_val_darcy

    darcy_split = {
        'train': darcy_idx[:n_train_darcy].tolist(),
        'val': darcy_idx[n_train_darcy:].tolist(),
    }
    save_json(out_dir / 'darcy_smooth1_split.json', darcy_split)

    print(f'Wrote {out_dir / "burgers_split.json"}')
    print(f'Wrote {out_dir / "darcy_smooth1_split.json"}')


if __name__ == '__main__':
    main()