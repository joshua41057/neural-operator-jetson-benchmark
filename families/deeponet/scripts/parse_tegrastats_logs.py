#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean


RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
RE_SWAP = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
RE_GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")
RE_EMC = re.compile(r"EMC_FREQ\s+(\d+)%")
RE_VDD_IN = re.compile(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?")
RE_TEMP = re.compile(r"([A-Za-z0-9_]+)@([0-9.]+)C")


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def safe_mean(xs: list[float]) -> float:
    return mean(xs) if xs else float("nan")


def safe_max(xs: list[float]) -> float:
    return max(xs) if xs else float("nan")


def parse_one(path: Path) -> dict[str, object]:
    ram_used = []
    ram_total = []
    swap_used = []
    swap_total = []
    gr3d = []
    emc = []
    vdd_now_mw = []
    vdd_avg_mw = []
    gpu_temp = []
    cpu_temp = []
    all_temp = []

    n_lines = 0

    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        n_lines += 1

        m = RE_RAM.search(line)
        if m:
            ram_used.append(float(m.group(1)))
            ram_total.append(float(m.group(2)))

        m = RE_SWAP.search(line)
        if m:
            swap_used.append(float(m.group(1)))
            swap_total.append(float(m.group(2)))

        m = RE_GR3D.search(line)
        if m:
            gr3d.append(float(m.group(1)))

        m = RE_EMC.search(line)
        if m:
            emc.append(float(m.group(1)))

        m = RE_VDD_IN.search(line)
        if m:
            vdd_now_mw.append(float(m.group(1)))
            if m.group(2) is not None:
                vdd_avg_mw.append(float(m.group(2)))

        for name, val in RE_TEMP.findall(line):
            temp = float(val)
            all_temp.append(temp)
            lname = name.lower()
            if "gpu" in lname:
                gpu_temp.append(temp)
            if "cpu" in lname:
                cpu_temp.append(temp)

    avg_power_w = safe_mean(vdd_now_mw) / 1000.0
    p95_power_w = pct(vdd_now_mw, 0.95) / 1000.0 if vdd_now_mw else float("nan")
    max_power_w = safe_max(vdd_now_mw) / 1000.0

    return {
        "log_path": str(path),
        "case_id": path.name.replace("_tegrastats_raw.log", "").replace("_tegrastats.log", ""),
        "n_samples": n_lines,
        "avg_vddin_w": avg_power_w,
        "p95_vddin_w": p95_power_w,
        "max_vddin_w": max_power_w,
        "avg_ram_used_mb": safe_mean(ram_used),
        "p95_ram_used_mb": pct(ram_used, 0.95) if ram_used else float("nan"),
        "max_ram_used_mb": safe_max(ram_used),
        "ram_total_mb": safe_max(ram_total),
        "avg_swap_used_mb": safe_mean(swap_used),
        "max_swap_used_mb": safe_max(swap_used),
        "swap_total_mb": safe_max(swap_total),
        "avg_gr3d_pct": safe_mean(gr3d),
        "p95_gr3d_pct": pct(gr3d, 0.95) if gr3d else float("nan"),
        "max_gr3d_pct": safe_max(gr3d),
        "avg_emc_pct": safe_mean(emc),
        "p95_emc_pct": pct(emc, 0.95) if emc else float("nan"),
        "max_emc_pct": safe_max(emc),
        "avg_gpu_temp_c": safe_mean(gpu_temp),
        "max_gpu_temp_c": safe_max(gpu_temp),
        "avg_cpu_temp_c": safe_mean(cpu_temp),
        "max_cpu_temp_c": safe_max(cpu_temp),
        "max_any_temp_c": safe_max(all_temp),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repository root.")
    ap.add_argument("--glob", action="append", required=True, help="Glob pattern relative to root. Can be repeated.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    logs: list[Path] = []
    for pattern in args.glob:
        logs.extend(root.glob(pattern))

    logs = sorted(set(p for p in logs if p.is_file()))

    rows = [parse_one(p) for p in logs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "log_path",
        "n_samples",
        "avg_vddin_w",
        "p95_vddin_w",
        "max_vddin_w",
        "avg_ram_used_mb",
        "p95_ram_used_mb",
        "max_ram_used_mb",
        "ram_total_mb",
        "avg_swap_used_mb",
        "max_swap_used_mb",
        "swap_total_mb",
        "avg_gr3d_pct",
        "p95_gr3d_pct",
        "max_gr3d_pct",
        "avg_emc_pct",
        "p95_emc_pct",
        "max_emc_pct",
        "avg_gpu_temp_c",
        "max_gpu_temp_c",
        "avg_cpu_temp_c",
        "max_cpu_temp_c",
        "max_any_temp_c",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Parsed {len(rows)} tegrastats logs")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()