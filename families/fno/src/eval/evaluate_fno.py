from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.datasets import build_dataset_bundle
from src.eval.common import load_model_and_normalizers
from src.train.engine import evaluate_epoch
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt, cfg, model, x_norm, y_norm, wrapper = load_model_and_normalizers(args.checkpoint, map_location='cpu')
    device = get_device(args.device)
    model = model.to(device)
    y_norm = y_norm.to(device)

    cfg_path = ckpt['config'].get('_config_path', None)
    if cfg_path is not None:
        cfg = load_config(cfg_path)
    else:
        from src.utils.config import AttrDict
        cfg = AttrDict(ckpt['config'])

    bundle = build_dataset_bundle(cfg)
    loader = {
        'train': bundle.train_loader,
        'val': bundle.val_loader,
        'test': bundle.test_loader,
    }[args.split]

    metrics = evaluate_epoch(model, loader, device=device, y_normalizer=y_norm)
    out = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'loss': metrics.loss,
        'rel_l2_mean': metrics.rel_l2_mean,
        'mse': metrics.mse,
        'mae': metrics.mae,
    }

    out_path = Path('results') / f"eval_{Path(args.checkpoint).stem}_{args.split}.json"
    save_json(out_path, out)
    print(out)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
