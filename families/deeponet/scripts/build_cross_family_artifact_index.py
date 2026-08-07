#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CATEGORIES = [
    ("validity", "results/artifacts/*validity*.csv", "predictive validity / held-out relative L2"),
    ("fp32_latency", "results/**/*fp32*.json", "short-run FP32 latency / memory harness output"),
    ("precision", "results/**/*precision*.json", "precision-mode execution output"),
    ("precision_summary", "results/artifacts/*precision*.csv", "precision feasibility or numerical perturbation summary"),
    ("energy_json", "results/**/*energy*.json", "long-run energy harness output"),
    ("tegrastats", "results/**/*tegrastats*.log", "raw board-level telemetry"),
    ("nsys_report", "results/**/*.nsys-rep", "Nsight Systems report"),
    ("nsys_sqlite", "results/**/*.sqlite", "Nsight Systems SQLite export"),
    ("nsys_stats", "results/**/*nsys*/*.csv", "Nsight Systems extracted stats"),
    ("ncu_report", "results/**/*.ncu-rep", "Nsight Compute report"),
    ("ncu_raw", "results/**/*ncu*/*raw.csv", "Nsight Compute raw metric export"),
    ("ncu_summary", "results/artifacts/*ncu*.csv", "Nsight Compute summary or inventory"),
]


def collect(root: Path, family: str) -> list[dict[str, object]]:
    rows = []
    for category, pattern, meaning in CATEGORIES:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                rows.append(
                    {
                        "family": family,
                        "category": category,
                        "path": str(p),
                        "bytes": p.stat().st_size,
                        "meaning": meaning,
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fno-root", required=True)
    ap.add_argument("--deeponet-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fno_root = Path(args.fno_root).expanduser().resolve()
    deeponet_root = Path(args.deeponet_root).expanduser().resolve()

    rows = []
    rows.extend(collect(fno_root, "FNO"))
    rows.extend(collect(deeponet_root, "DeepONet"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "category", "path", "bytes", "meaning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()