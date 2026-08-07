#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path("results/profiles/deeponet_ncu_detailed")
EXPDIR = Path("results/profiles/deeponet_ncu_detailed_exports")

CASES = [
    "burgers_base_r2048_ts_fp32",
    "darcy_base_r141_ts_fp32",
    "darcy_base_r281_ts_fp32",
    "darcy_base_r281_ts_fp16_native",
    "darcy_large_r141_ts_fp32",
]

KEYS = [
    "gpu__time_duration",
    "sm__throughput",
    "gpu__compute_memory_throughput",
    "l1tex__throughput",
    "lts__throughput",
    "lts__t_sector_hit_rate",
    "lts__t_request_hit_rate",
    "launch__occupancy",
    "sm__warps_active",
    "smsp__warps",
    "eligible",
]

bad = False

for case in CASES:
    rep = ROOT / f"{case}.ncu-rep"
    log = ROOT / f"{case}.log"
    raw = EXPDIR / f"{case}_raw.csv"
    details = EXPDIR / f"{case}_details.csv"
    per_kernel = EXPDIR / f"{case}_per_kernel.csv"

    print(f"\n=== {case} ===")

    for p in [rep, log, raw, details, per_kernel]:
        if not p.exists():
            print(f"ERROR missing: {p}")
            bad = True
            continue
        print(f"{p}: {p.stat().st_size} bytes")

    if log.exists():
        text = log.read_text(errors="ignore")
        if "No kernels were profiled" in text:
            print("ERROR: No kernels were profiled")
            bad = True
        if "--nvtx-include" in text or "NVTX include" in text:
            print("ERROR: stale NVTX-filtered run detected")
            bad = True

    if details.exists():
        text = details.read_text(errors="ignore")
        present = [k for k in KEYS if k in text]
        print("metric keys present:", ", ".join(present) if present else "NONE")
        if "gpu__time_duration" not in text:
            print("ERROR: missing gpu__time_duration in details export")
            bad = True
        if ("l1tex__" not in text) and ("lts__" not in text):
            print("WARNING: no L1/L2/LTS metric names found in details export")

    if per_kernel.exists() and per_kernel.stat().st_size < 1000:
        print("ERROR: per-kernel export is too small")
        bad = True

if bad:
    raise SystemExit(1)

print("\nNCU detailed exports look usable.")
