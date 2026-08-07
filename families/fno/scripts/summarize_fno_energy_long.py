from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median


RESULTS_DIR = Path("results/jetson_fno_energy_long")
OUT_CSV = Path("results/artifacts/fno_energy_long_summary.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def parse_float_list(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "min": None,
        }
    s = sorted(vals)
    return {
        "mean": mean(vals),
        "median": median(vals),
        "p95": s[int(0.95 * (len(s) - 1))],
        "max": max(vals),
        "min": min(vals),
    }


def parse_tegrastats(path: Path) -> dict:
    # Typical fields:
    # RAM 2184/7620MB ...
    # GR3D_FREQ 33%
    # cpu@48.0C ... gpu@47.5C ... tj@48.25C
    # VDD_IN 9655mW/9500mW
    ram_re = re.compile(r"RAM\s+(\d+)/(\d+)MB")
    gr3d_re = re.compile(r"GR3D_FREQ\s+(\d+)%")
    vdd_re = re.compile(r"VDD_IN\s+(\d+)mW/(\d+)mW")
    gpu_temp_re = re.compile(r"gpu@([0-9.]+)C")
    tj_temp_re = re.compile(r"tj@([0-9.]+)C")

    ram_used = []
    gr3d = []
    vdd_inst = []
    vdd_avg_field = []
    gpu_temp = []
    tj_temp = []

    if not path.exists():
        return {
            "tegrastats_exists": False,
            "tegrastats_samples": 0,
        }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ram_re.search(line)
            if m:
                ram_used.append(int(m.group(1)))

            g = gr3d_re.search(line)
            if g:
                gr3d.append(int(g.group(1)))

            p = vdd_re.search(line)
            if p:
                vdd_inst.append(int(p.group(1)))
                vdd_avg_field.append(int(p.group(2)))

            gt = gpu_temp_re.search(line)
            if gt:
                gpu_temp.append(float(gt.group(1)))

            tt = tj_temp_re.search(line)
            if tt:
                tj_temp.append(float(tt.group(1)))

    vdd_stats = parse_float_list(vdd_inst)
    ram_stats = parse_float_list(ram_used)
    gr3d_stats = parse_float_list(gr3d)
    gpu_temp_stats = parse_float_list(gpu_temp)
    tj_temp_stats = parse_float_list(tj_temp)

    return {
        "tegrastats_exists": True,
        "tegrastats_samples": len(vdd_inst),
        "ram_used_mean_mb": ram_stats["mean"],
        "ram_used_median_mb": ram_stats["median"],
        "ram_used_peak_mb": ram_stats["max"],
        "vdd_in_mean_mw": vdd_stats["mean"],
        "vdd_in_median_mw": vdd_stats["median"],
        "vdd_in_p95_mw": vdd_stats["p95"],
        "vdd_in_peak_mw": vdd_stats["max"],
        "gr3d_mean_pct": gr3d_stats["mean"],
        "gr3d_median_pct": gr3d_stats["median"],
        "gr3d_p95_pct": gr3d_stats["p95"],
        "gpu_temp_start_c": gpu_temp[0] if gpu_temp else None,
        "gpu_temp_end_c": gpu_temp[-1] if gpu_temp else None,
        "gpu_temp_peak_c": gpu_temp_stats["max"],
        "tj_temp_start_c": tj_temp[0] if tj_temp else None,
        "tj_temp_end_c": tj_temp[-1] if tj_temp else None,
        "tj_temp_peak_c": tj_temp_stats["max"],
    }


def main():
    rows = []

    for jp in sorted(RESULTS_DIR.glob("*.json")):
        tag = jp.stem
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)

        tegra = parse_tegrastats(RESULTS_DIR / f"{tag}_tegrastats.log")

        avg_power_w = None
        median_power_w = None
        p95_power_w = None
        peak_power_w = None
        energy_total_j = None
        energy_per_inf_j = None

        if tegra.get("vdd_in_mean_mw") is not None:
            avg_power_w = tegra["vdd_in_mean_mw"] / 1000.0
            energy_total_j = avg_power_w * float(d.get("measure_seconds_actual", 0.0))
            iters = d.get("measure_iters", None)
            if iters:
                energy_per_inf_j = energy_total_j / float(iters)

        if tegra.get("vdd_in_median_mw") is not None:
            median_power_w = tegra["vdd_in_median_mw"] / 1000.0
        if tegra.get("vdd_in_p95_mw") is not None:
            p95_power_w = tegra["vdd_in_p95_mw"] / 1000.0
        if tegra.get("vdd_in_peak_mw") is not None:
            peak_power_w = tegra["vdd_in_peak_mw"] / 1000.0

        rows.append({
            "tag": tag,
            "status": d.get("status"),
            "dataset": d.get("dataset"),
            "resolution": d.get("resolution"),
            "parameter_count": d.get("parameter_count"),
            "mode": d.get("mode"),
            "precision_mode": d.get("precision_mode"),
            "input_bank": d.get("input_bank"),
            "input_shape": d.get("input_shape"),
            "measure_seconds_actual": d.get("measure_seconds_actual"),
            "measure_iters": d.get("measure_iters"),
            "throughput_inf_s": d.get("throughput_inf_s"),
            "median_ms": d.get("median_ms"),
            "mean_ms": d.get("mean_ms"),
            "p95_ms": d.get("p95_ms"),
            "p99_ms": d.get("p99_ms"),
            "peak_cuda_alloc_mb": d.get("peak_cuda_alloc_mb"),
            "peak_cuda_reserved_mb": d.get("peak_cuda_reserved_mb"),
            "tegrastats_samples": tegra.get("tegrastats_samples"),
            "ram_used_mean_mb": tegra.get("ram_used_mean_mb"),
            "ram_used_peak_mb": tegra.get("ram_used_peak_mb"),
            "avg_power_w": avg_power_w,
            "median_power_w": median_power_w,
            "p95_power_w": p95_power_w,
            "peak_power_w": peak_power_w,
            "energy_total_j": energy_total_j,
            "energy_per_inf_j": energy_per_inf_j,
            "gr3d_mean_pct": tegra.get("gr3d_mean_pct"),
            "gr3d_p95_pct": tegra.get("gr3d_p95_pct"),
            "gpu_temp_start_c": tegra.get("gpu_temp_start_c"),
            "gpu_temp_end_c": tegra.get("gpu_temp_end_c"),
            "gpu_temp_peak_c": tegra.get("gpu_temp_peak_c"),
            "tj_temp_start_c": tegra.get("tj_temp_start_c"),
            "tj_temp_end_c": tegra.get("tj_temp_end_c"),
            "tj_temp_peak_c": tegra.get("tj_temp_peak_c"),
        })

    if not rows:
        raise SystemExit(f"No JSON files found in {RESULTS_DIR}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
