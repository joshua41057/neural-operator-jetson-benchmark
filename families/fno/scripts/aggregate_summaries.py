#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

try:
    import numpy as np
except Exception:
    np = None


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='Example: checkpoints/burgers_fno_base')
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith('seed')])
    if not seed_dirs:
        raise RuntimeError(f'No seed directories found under {exp_dir}')

    rows = []
    for sd in seed_dirs:
        summary_path = sd / 'summary.json'
        if not summary_path.exists():
            print(f'Skipping missing summary: {summary_path}')
            continue
        rows.append(load_json(summary_path))

    if not rows:
        raise RuntimeError(f'No summary.json files found under {exp_dir}')

    keys_numeric = [
        'best_val_rel_l2',
        'test_rel_l2_mean',
        'test_rel_l2_std',
        'test_rel_l2_median',
        'test_mse',
        'test_mae',
        'parameter_count',
        'best_epoch',
    ]

    agg = {
        'experiment_name': rows[0]['experiment_name'],
        'num_seeds': len(rows),
        'seeds': [r['seed'] for r in rows],
        'per_seed': rows,
    }

    for key in keys_numeric:
        vals = [float(r[key]) for r in rows if key in r]
        if not vals:
            continue
        agg[f'{key}_mean'] = mean(vals)
        agg[f'{key}_std'] = pstdev(vals) if len(vals) > 1 else 0.0
        if np is not None:
            arr = np.asarray(vals, dtype=float)
            agg[f'{key}_min'] = float(arr.min())
            agg[f'{key}_max'] = float(arr.max())

    out_path = exp_dir / 'aggregate_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)

    print(f'Wrote {out_path}')
    print(json.dumps({
        'experiment_name': agg['experiment_name'],
        'num_seeds': agg['num_seeds'],
        'test_rel_l2_mean_mean': agg.get('test_rel_l2_mean_mean', None),
        'test_rel_l2_mean_std': agg.get('test_rel_l2_mean_std', None),
        'test_mse_mean': agg.get('test_mse_mean_mean', None),
        'test_mae_mean': agg.get('test_mae_mean_mean', None),
    }, indent=2))


if __name__ == '__main__':
    main()