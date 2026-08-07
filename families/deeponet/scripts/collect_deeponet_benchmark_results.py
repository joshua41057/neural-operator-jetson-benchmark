#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_manifest():
    m = {}
    for r in csv.DictReader(open("manifests/deeponet_jetson_manifest.csv", "r", encoding="utf-8")):
        m[r["experiment_name"]] = r
    return m


def infer_exp_from_checkpoint(path: str) -> str:
    name = Path(path).name
    # remove _seedX_best.pt suffix
    parts = name.split("_seed")
    return parts[0]


def main():
    manifest = load_manifest()
    results_dir = Path("results/jetson_deeponet_fp32")
    out = Path("results/artifacts/deeponet_fp32_deployment_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(results_dir.glob("*.json")):
        d = json.load(open(p, "r", encoding="utf-8"))
        exp = infer_exp_from_checkpoint(d["checkpoint"])
        mr = manifest.get(exp, {})

        row = {
            "experiment_name": exp,
            "task": mr.get("dataset", ""),
            "spatial_dim": mr.get("spatial_dim", ""),
            "input_shape": mr.get("input_shape", ""),
            "output_shape": mr.get("output_shape", ""),
            "parameter_count": mr.get("parameter_count", ""),
            "selected_seed": mr.get("selected_seed", ""),
            "test_rel_l2_mean": mr.get("test_rel_l2_mean", ""),
            "test_rel_l2_std": mr.get("test_rel_l2_std", ""),
            "sensor_resolution": mr.get("sensor_resolution", ""),
            "basis_dim": mr.get("basis_dim", ""),
            "branch_hidden_dim": mr.get("branch_hidden_dim", ""),
            "branch_layers": mr.get("branch_layers", ""),
            "trunk_hidden_dim": mr.get("trunk_hidden_dim", ""),
            "trunk_layers": mr.get("trunk_layers", ""),
            "backend": d.get("mode"),
            "precision": d.get("precision"),
            "batch_size": d.get("batch_size"),
            "num_warmup": d.get("num_warmup"),
            "num_iters": d.get("num_iters"),
            "mean_ms": d.get("mean_ms"),
            "median_ms": d.get("median_ms"),
            "p95_ms": d.get("p95_ms"),
            "p99_ms": d.get("p99_ms"),
            "std_ms": d.get("std_ms"),
            "min_ms": d.get("min_ms"),
            "max_ms": d.get("max_ms"),
            "peak_cuda_allocated_mb": d.get("peak_cuda_allocated_mb"),
            "input_bank": d.get("input_bank"),
            "checkpoint": d.get("checkpoint"),
            "torchscript": d.get("torchscript"),
            "result_json": str(p),
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"No JSON results found in {results_dir}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
