#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def mean(xs):
    return float(statistics.mean(xs)) if xs else None


def stdev(xs):
    return float(statistics.stdev(xs)) if len(xs) >= 2 else 0.0


def aggregate_experiment(exp_dir: Path):
    summaries = []
    for seed_dir in sorted(exp_dir.glob("seed*")):
        sp = seed_dir / "summary.json"
        if sp.exists:
            summaries.append(load_json(sp))

    if not summaries:
        return None

    keys = [
        "best_val_rel_l2",
        "test_rel_l2_mean",
        "test_rel_l2_std",
        "test_rel_l2_median",
        "test_mse",
        "test_mae",
        "parameter_count",
        "best_epoch",
    ]

    agg = {
        "experiment_name": exp_dir.name,
        "family": "DeepONet",
        "n_seeds": len(summaries),
        "seeds": [int(s["seed"]) for s in summaries],
        "dataset": summaries[0].get("dataset"),
        "spatial_dim": summaries[0].get("spatial_dim"),
        "input_shape": summaries[0].get("input_shape"),
        "output_shape": summaries[0].get("output_shape"),
        "sensor_resolution": summaries[0].get("sensor_resolution"),
        "basis_dim": summaries[0].get("basis_dim"),
        "branch_hidden_dim": summaries[0].get("branch_hidden_dim"),
        "branch_layers": summaries[0].get("branch_layers"),
        "trunk_hidden_dim": summaries[0].get("trunk_hidden_dim"),
        "trunk_layers": summaries[0].get("trunk_layers"),
        "use_coord_features": summaries[0].get("use_coord_features"),
        "chunk_size": summaries[0].get("chunk_size"),
    }

    for k in keys:
        vals = [float(s[k]) for s in summaries if k in s and s[k] is not None]
        if vals:
            agg[f"{k}_mean"] = mean(vals)
            agg[f"{k}_std"] = stdev(vals)
            agg[f"{k}_min"] = float(min(vals))
            agg[f"{k}_max"] = float(max(vals))

    # Select best seed by lowest test rel-L2.
    best = min(summaries, key=lambda s: float(s["test_rel_l2_mean"]))
    agg["best_seed_by_test_rel_l2"] = int(best["seed"])
    agg["best_seed_test_rel_l2"] = float(best["test_rel_l2_mean"])
    agg["best_seed_val_rel_l2"] = float(best["best_val_rel_l2"])

    save_json(exp_dir / "aggregate_summary.json", agg)
    return agg


def main():
    root = Path("checkpoints")
    rows = []
    for exp_dir in sorted(root.glob("*deeponet*")):
        if not exp_dir.is_dir():
            continue
        agg = aggregate_experiment(exp_dir)
        if agg is not None:
            rows.append(agg)
            print(f"aggregated {exp_dir}")

    print(f"Aggregated {len(rows)} DeepONet experiments")


if __name__ == "__main__":
    main()
