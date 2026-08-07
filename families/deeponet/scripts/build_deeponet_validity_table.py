#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def model_group(exp: str) -> str:
    if "_small" in exp:
        return "small"
    if "_large" in exp:
        return "large"
    return "base"


def resolution_string(dataset: str, input_shape: str) -> str:
    res = json.loads(input_shape)
    if dataset == "burgers":
        return str(res[0])
    return f"{res[0]}x{res[1]}"


def main():
    manifest = Path("manifests/deeponet_jetson_manifest.csv")
    out = Path("results/artifacts/deeponet_validity_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = json.load(open(r["summary_path"], "r", encoding="utf-8"))
            rows.append({
                "experiment_name": r["experiment_name"],
                "task": r["dataset"],
                "model_group": model_group(r["experiment_name"]),
                "resolution": resolution_string(r["dataset"], r["input_shape"]),
                "selected_seed": r["selected_seed"],
                "params": r["parameter_count"],
                "best_val_rel_l2": r["best_val_rel_l2"],
                "test_rel_l2_mean": r["test_rel_l2_mean"],
                "test_rel_l2_std": r["test_rel_l2_std"],
                "sensor_resolution": r["sensor_resolution"],
                "basis_dim": r["basis_dim"],
                "branch_hidden_dim": r["branch_hidden_dim"],
                "branch_layers": r["branch_layers"],
                "trunk_hidden_dim": r["trunk_hidden_dim"],
                "trunk_layers": r["trunk_layers"],
                "checkpoint": r["checkpoint_path"],
                "torchscript": r["torchscript_path"],
                "input_bank": r["input_bank_path"],
            })

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
