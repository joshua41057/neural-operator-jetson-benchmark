#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = [
    "burgers_deeponet_small",
    "burgers_deeponet_base",
    "burgers_deeponet_large",
    "darcy_deeponet_small",
    "darcy_deeponet_base",
    "darcy_deeponet_large",
    "burgers_deeponet_base_r512",
    "burgers_deeponet_base_r1024",
    "burgers_deeponet_base_r2048",
    "burgers_deeponet_base_r4096",
    "darcy_deeponet_base_r85",
    "darcy_deeponet_base_r141",
    "darcy_deeponet_base_r211",
    "darcy_deeponet_base_r281",
]


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    out = Path("results/deeponet_seed_selection.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        exp_dir = Path("checkpoints") / exp
        candidates = []
        for seed_dir in sorted(exp_dir.glob("seed*")):
            sp = seed_dir / "summary.json"
            if not sp.exists():
                continue
            s = load_json(sp)
            candidates.append(s)

        if not candidates:
            print(f"WARNING: no candidates for {exp}")
            continue

        # Use validation for selection to avoid selecting on test.
        selected = min(candidates, key=lambda s: float(s["best_val_rel_l2"]))

        rows.append(
            {
                "experiment_name": exp,
                "selected_seed": int(selected["seed"]),
                "selection_rule": "lowest_best_val_rel_l2",
                "dataset": selected.get("dataset"),
                "spatial_dim": selected.get("spatial_dim"),
                "input_shape": selected.get("input_shape"),
                "output_shape": selected.get("output_shape"),
                "parameter_count": selected.get("parameter_count"),
                "best_val_rel_l2": selected.get("best_val_rel_l2"),
                "test_rel_l2_mean": selected.get("test_rel_l2_mean"),
                "test_rel_l2_std": selected.get("test_rel_l2_std"),
                "sensor_resolution": selected.get("sensor_resolution"),
                "basis_dim": selected.get("basis_dim"),
                "branch_hidden_dim": selected.get("branch_hidden_dim"),
                "branch_layers": selected.get("branch_layers"),
                "trunk_hidden_dim": selected.get("trunk_hidden_dim"),
                "trunk_layers": selected.get("trunk_layers"),
            }
        )

    if not rows:
        raise RuntimeError("No DeepONet rows selected")

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
