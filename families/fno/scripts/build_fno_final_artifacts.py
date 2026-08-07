from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

OUT_DIR = Path("results/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] missing {path}")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        print(f"[WARN] no rows for {path}")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} with {len(rows)} rows")


def as_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def normalize_tag(row):
    return row.get("tag", "")


def make_main_deployability_table():
    rows = read_csv(Path("results/fno_all_results_summary.csv"))
    if not rows:
        rows = read_csv(Path("results/jetson_fno_summary.csv"))

    selected = []
    for r in rows:
        tag = normalize_tag(r)
        if tag.startswith("smoke_"):
            continue

        dataset = r.get("dataset", "")
        mode = r.get("mode", "")
        precision = r.get("precision", r.get("precision_mode", ""))
        resolution = r.get("resolution", "")

        # Main table should be focused, not the entire appendix.
        keep = False
        if mode == "torchscript" and precision in ["fp32", "fp32_strict", ""]:
            if "base_seed" in tag or "_small_" in tag or "_large_" in tag or "_r85_" in tag or "_r141_" in tag or "_r211_" in tag or "_r281_" in tag or "_r421_" in tag:
                keep = True
        if "frontier" in r.get("results_dir", ""):
            keep = True

        if keep:
            selected.append({
                "tag": tag,
                "dataset": dataset,
                "resolution": resolution,
                "mode": mode,
                "precision": precision,
                "batch_size": r.get("batch_size", ""),
                "mean_ms": r.get("mean_ms", ""),
                "median_ms": r.get("median_ms", ""),
                "p95_ms": r.get("p95_ms", ""),
                "p99_ms": r.get("p99_ms", ""),
                "peak_ram_mb": r.get("peak_ram_mb", ""),
                "avg_power_w_short_run": r.get("avg_power_w", ""),
                "energy_per_inf_j_short_run": r.get("energy_per_inf_j", ""),
                "source": r.get("source", ""),
                "input_bank": r.get("input_bank", ""),
            })

    write_csv(OUT_DIR / "paper_fno_main_deployability_table.csv", selected)


def make_resolution_scaling_table():
    rows = read_csv(Path("results/fno_all_results_summary.csv"))
    if not rows:
        rows = read_csv(Path("results/jetson_fno_summary.csv"))

    selected = []
    for r in rows:
        tag = normalize_tag(r)
        if r.get("mode") != "torchscript":
            continue
        if r.get("precision", "") not in ["fp32", "fp32_strict"]:
            continue
        if "_r85_" in tag or "_r141_" in tag or "_r211_" in tag or "_r281_" in tag or "_r421_" in tag or "_r512_" in tag or "_r1024_" in tag or "_r2048_" in tag or "_r4096_" in tag or "_r8192_" in tag:
            selected.append({
                "tag": tag,
                "dataset": r.get("dataset", ""),
                "resolution": r.get("resolution", ""),
                "median_ms": r.get("median_ms", ""),
                "p95_ms": r.get("p95_ms", ""),
                "peak_ram_mb": r.get("peak_ram_mb", ""),
                "avg_power_w_short_run": r.get("avg_power_w", ""),
                "energy_per_inf_j_short_run": r.get("energy_per_inf_j", ""),
                "input_bank": r.get("input_bank", ""),
            })

    write_csv(OUT_DIR / "paper_fno_resolution_scaling_table.csv", selected)


def make_backend_speedup_table():
    rows = read_csv(Path("results/fno_all_results_summary.csv"))
    if not rows:
        rows = read_csv(Path("results/jetson_fno_summary.csv"))

    by_base = {}
    for r in rows:
        tag = normalize_tag(r)
        if tag.startswith("smoke_"):
            continue
        if r.get("precision", "") not in ["fp32", "fp32_strict"]:
            continue

        base = tag
        base = base.replace("_eager_fp32", "")
        base = base.replace("_torchscript_fp32", "")
        base = base.replace("_eager_fp32_strict", "")
        base = base.replace("_torchscript_fp32_strict", "")

        by_base.setdefault(base, {})[r.get("mode", "")] = r

    out = []
    for base, pair in sorted(by_base.items()):
        if "eager" not in pair or "torchscript" not in pair:
            continue
        e = pair["eager"]
        t = pair["torchscript"]
        e_med = as_float(e.get("median_ms"))
        t_med = as_float(t.get("median_ms"))
        speedup = e_med / t_med if e_med and t_med else None

        out.append({
            "case": base,
            "dataset": t.get("dataset", e.get("dataset", "")),
            "resolution": t.get("resolution", e.get("resolution", "")),
            "eager_median_ms": e.get("median_ms", ""),
            "torchscript_median_ms": t.get("median_ms", ""),
            "torchscript_speedup_vs_eager": speedup,
        })

    write_csv(OUT_DIR / "paper_fno_backend_speedup_table.csv", out)


def make_precision_tables():
    # These were already created by previous precision scripts.
    raw = read_csv(OUT_DIR / "precision_runs_raw.csv")
    fail_sum = read_csv(OUT_DIR / "precision_failure_summary.csv")
    fail_ex = read_csv(OUT_DIR / "precision_failure_examples.csv")
    tf32 = read_csv(OUT_DIR / "precision_tf32_vs_fp32.csv")
    fft_safe = read_csv(Path("results/fno_precision_fft_safe_summary.csv"))

    if tf32:
        write_csv(OUT_DIR / "paper_fno_precision_tf32_table.csv", tf32)
    if fail_sum:
        write_csv(OUT_DIR / "paper_fno_precision_failure_summary_table.csv", fail_sum)
    if fail_ex:
        write_csv(OUT_DIR / "paper_fno_precision_failure_examples_table.csv", fail_ex)
    if fft_safe:
        write_csv(OUT_DIR / "paper_fno_precision_fft_safe_workaround_table.csv", fft_safe)


def make_energy_table():
    rows = read_csv(OUT_DIR / "fno_energy_long_summary.csv")
    selected = []
    for r in rows:
        selected.append({
            "tag": r.get("tag", ""),
            "dataset": r.get("dataset", ""),
            "resolution": r.get("resolution", ""),
            "parameter_count": r.get("parameter_count", ""),
            "mode": r.get("mode", ""),
            "precision_mode": r.get("precision_mode", ""),
            "measure_seconds_actual": r.get("measure_seconds_actual", ""),
            "measure_iters": r.get("measure_iters", ""),
            "throughput_inf_s": r.get("throughput_inf_s", ""),
            "median_ms": r.get("median_ms", ""),
            "p95_ms": r.get("p95_ms", ""),
            "avg_power_w": r.get("avg_power_w", ""),
            "median_power_w": r.get("median_power_w", ""),
            "p95_power_w": r.get("p95_power_w", ""),
            "peak_power_w": r.get("peak_power_w", ""),
            "energy_per_inf_j": r.get("energy_per_inf_j", ""),
            "ram_used_peak_mb": r.get("ram_used_peak_mb", ""),
            "peak_cuda_alloc_mb": r.get("peak_cuda_alloc_mb", ""),
            "gpu_temp_start_c": r.get("gpu_temp_start_c", ""),
            "gpu_temp_end_c": r.get("gpu_temp_end_c", ""),
            "gpu_temp_peak_c": r.get("gpu_temp_peak_c", ""),
            "tj_temp_peak_c": r.get("tj_temp_peak_c", ""),
        })
    write_csv(OUT_DIR / "paper_fno_long_energy_table.csv", selected)


def make_frontier_table():
    rows = read_csv(Path("results/jetson_fno_oom_frontier/frontier_status.csv"))
    if rows:
        write_csv(OUT_DIR / "paper_fno_oom_frontier_table.csv", rows)

    frontier = []
    for source in [
        Path("results/fno_all_results_summary.csv"),
        Path("results/jetson_fno_summary.csv"),
    ]:
        for r in read_csv(source):
            if "jetson_fno_frontier" in r.get("results_dir", "") or "on421" in r.get("tag", "") or "_on_421" in r.get("tag", "") or "r8192" in r.get("input_bank", ""):
                frontier.append(r)

    if frontier:
        write_csv(OUT_DIR / "paper_fno_native_frontier_latency_table.csv", frontier)


def make_validity_table():
    rows = read_csv(OUT_DIR / "fno_quality_validity_table.csv")
    if rows:
        write_csv(OUT_DIR / "paper_fno_quality_validity_table.csv", rows)


def make_profile_tables():
    nsys = read_csv(Path("results/jetson_fno_profile_nsys/nsys_profile_summary.csv"))
    kernels = read_csv(Path("results/jetson_fno_profile_nsys/nsys_top_kernels.csv"))
    ncu_summary = read_csv(OUT_DIR / "ncu_kernel_summary.csv")
    ncu_long = read_csv(OUT_DIR / "ncu_kernel_metrics_long.csv")

    if nsys:
        write_csv(OUT_DIR / "paper_fno_nsys_summary_table.csv", nsys)
    if kernels:
        write_csv(OUT_DIR / "paper_fno_nsys_top_kernels_table.csv", kernels)
    if ncu_summary:
        write_csv(OUT_DIR / "paper_fno_ncu_kernel_summary_table.csv", ncu_summary)
    if ncu_long:
        write_csv(OUT_DIR / "paper_fno_ncu_kernel_metrics_long_table.csv", ncu_long)


def main():
    make_main_deployability_table()
    make_resolution_scaling_table()
    make_backend_speedup_table()
    make_precision_tables()
    make_energy_table()
    make_frontier_table()
    make_validity_table()
    make_profile_tables()

    print("\nFinal FNO paper artifacts written under results/artifacts/")
    print("Recommended main-text tables:")
    print("  results/artifacts/paper_fno_main_deployability_table.csv")
    print("  results/artifacts/paper_fno_resolution_scaling_table.csv")
    print("  results/artifacts/paper_fno_backend_speedup_table.csv")
    print("  results/artifacts/paper_fno_precision_tf32_table.csv")
    print("  results/artifacts/paper_fno_precision_failure_summary_table.csv")
    print("  results/artifacts/paper_fno_long_energy_table.csv")
    print("  results/artifacts/paper_fno_native_frontier_latency_table.csv")
    print("  results/artifacts/paper_fno_quality_validity_table.csv")


if __name__ == "__main__":
    main()
