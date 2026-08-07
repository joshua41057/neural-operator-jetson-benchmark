from __future__ import annotations

import csv
import json
from pathlib import Path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    results_dir = Path("results/jetson_fno_precision_fft_safe")
    out_csv = Path("results/fno_precision_fft_safe_summary.csv")

    rows = []

    for p in sorted(results_dir.glob("*.json")):
        d = read_json(p)

        rows.append({
            "tag": p.stem,
            "status": d.get("status"),
            "dataset": d.get("dataset"),
            "resolution": d.get("resolution"),
            "parameter_count": d.get("parameter_count"),
            "precision_mode": d.get("precision_mode"),
            "mode": d.get("mode", "eager"),
            "input_bank": d.get("input_bank"),
            "input_shape": d.get("input_shape"),
            "input_dtype": d.get("input_dtype"),
            "median_ms": d.get("median_ms"),
            "mean_ms": d.get("mean_ms"),
            "p95_ms": d.get("p95_ms"),
            "p99_ms": d.get("p99_ms"),
            "repeat_median_ms_median": d.get("repeat_median_ms_median"),
            "repeat_median_ms_std": d.get("repeat_median_ms_std"),
            "peak_cuda_alloc_mb": d.get("peak_cuda_alloc_mb"),
            "peak_cuda_reserved_mb": d.get("peak_cuda_reserved_mb"),
            "error_type": d.get("error_type"),
            "error_message": d.get("error_message", "")[:240],
        })

    if not rows:
        raise SystemExit(f"No JSON files found in {results_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv} with {len(rows)} rows")


if __name__ == "__main__":
    main()