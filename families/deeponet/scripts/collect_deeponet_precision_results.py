#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def main():
    results_dir = Path("results/jetson_deeponet_precision")
    out = Path("results/artifacts/deeponet_precision_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(results_dir.glob("*.json")):
        d = json.load(open(p, "r", encoding="utf-8"))
        rows.append({
            "result_tag": d.get("result_tag", p.stem),
            "status": d.get("status", "success"),
            "failure_class": d.get("failure_class", ""),
            "mode": d.get("mode", d.get("backend", "")),
            "precision": d.get("precision"),
            "checkpoint": d.get("checkpoint"),
            "torchscript": d.get("torchscript"),
            "input_bank": d.get("input_bank"),
            "batch_size": d.get("batch_size", ""),
            "num_warmup": d.get("num_warmup", ""),
            "num_iters": d.get("num_iters", ""),
            "mean_ms": d.get("mean_ms", ""),
            "median_ms": d.get("median_ms", ""),
            "p95_ms": d.get("p95_ms", ""),
            "p99_ms": d.get("p99_ms", ""),
            "peak_cuda_allocated_mb": d.get("peak_cuda_allocated_mb", ""),
            "json": str(p),
        })

    if not rows:
        raise SystemExit(f"No JSON files found in {results_dir}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
