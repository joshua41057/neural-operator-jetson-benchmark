# scripts/select_fno_jetson_seeds.py
from __future__ import annotations

import json
import csv
from pathlib import Path

ROOT = Path("checkpoints")
OUT = Path("results/fno_seed_selection.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []

for exp_dir in sorted(ROOT.iterdir()):
    if not exp_dir.is_dir():
        continue
    if exp_dir.name == "legacy":
        continue

    seed_summaries = []
    for seed_dir in sorted(exp_dir.glob("seed*")):
        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        seed_summaries.append({
            "experiment_name": exp_dir.name,
            "seed": int(s["seed"]),
            "test_rel_l2_mean": float(s["test_rel_l2_mean"]),
            "parameter_count": int(s["parameter_count"]),
            "input_shape": s["input_shape"],
            "output_shape": s["output_shape"],
            "dataset": s["dataset"],
            "spatial_dim": int(s["spatial_dim"]),
        })

    if not seed_summaries:
        continue

    seed_summaries = sorted(seed_summaries, key=lambda x: (x["test_rel_l2_mean"], x["seed"]))
    median_idx = len(seed_summaries) // 2
    chosen = seed_summaries[median_idx]

    rows.append({
        "experiment_name": chosen["experiment_name"],
        "selected_seed": chosen["seed"],
        "selection_rule": "median_test_rel_l2",
        "selected_test_rel_l2_mean": chosen["test_rel_l2_mean"],
        "parameter_count": chosen["parameter_count"],
        "dataset": chosen["dataset"],
        "spatial_dim": chosen["spatial_dim"],
        "input_shape": json.dumps(chosen["input_shape"]),
        "output_shape": json.dumps(chosen["output_shape"]),
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT} with {len(rows)} rows")