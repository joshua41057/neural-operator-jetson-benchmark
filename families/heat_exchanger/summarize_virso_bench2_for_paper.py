#!/usr/bin/env python3
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

ROOT = os.path.expanduser("~/VirSO/For_Jetson/For_Jetson")
REG = os.path.join(ROOT, "inference_runs", "virso_direct_benchmark_registry.csv")
OUT = os.path.join(ROOT, "inference_runs", "virso_bench2_paper_summary.csv")

CASE_NAME = {
    "full_fp32_feas": "full_fp32",
    "full_fp32": "full_fp32",
    "full_ampfp16": "full_ampfp16",
    "spectral_fp32": "spectral_fp32",
    "layer2_fp32": "layer2_fp32",
}

PRETTY = {
    "full_fp32": "VIRSO full FP32",
    "full_ampfp16": "VIRSO full AMP FP16",
    "spectral_fp32": "VIRSO spectral-only FP32",
    "layer2_fp32": "VIRSO 2-layer FP32",
}

def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.mean(xs) if xs else None

def stdev(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0

def get_first(row, names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None

if not os.path.exists(REG):
    raise SystemExit(f"Missing registry: {REG}")

with open(REG, newline="") as f:
    reg = list(csv.DictReader(f))

records = []
for r in reg:
    raw_case = r.get("case")
    if raw_case not in CASE_NAME:
        continue

    run_ts = r["run_ts"]
    run_dir = os.path.join(ROOT, "inference_runs", run_ts)
    report = glob.glob(os.path.join(run_dir, "reports", "jetson_reviewer_report_*.csv"))

    rec = dict(r)
    rec["case_clean"] = CASE_NAME[raw_case]
    rec["run_dir"] = run_dir
    rec["has_report"] = bool(report)

    if report:
        with open(report[0], newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            rec.update(rows[0])

    records.append(rec)

print("=== PER-RUN RECORDS ===")
for r in records:
    print(
        f"{r.get('run_ts')} case={r.get('case_clean')} "
        f"exit={r.get('exit_status')} has_report={r.get('has_report')} "
        f"run_status={r.get('run_status')}"
    )

groups = defaultdict(list)
for r in records:
    if str(r.get("exit_status")) != "0":
        continue
    if not r.get("has_report"):
        continue
    groups[r["case_clean"]].append(r)

metric_alias = {
    "median_ms": ["p50_latency_ms", "median_latency_ms", "latency_p50_ms"],
    "p95_ms": ["p95_latency_ms", "latency_p95_ms"],
    "avg_ms": ["avg_latency_ms", "mean_latency_ms"],
    "avg_power_w": ["avg_power_w", "mean_power_w", "avg_total_power_w"],
    "j_per_it": ["energy_j_per_sample_from_total", "energy_j_per_sample", "j_per_sample", "energy_j_per_iter"],
    "peak_ram_mb": ["peak_ram_used_mb", "peak_ram_mb", "max_ram_used_mb"],
    "max_temp_c": ["max_temp_c", "max_gpu_temp_c", "max_cpu_temp_c"],
    "rel_l2": ["avg_total_loss", "mean_rel_l2", "relative_l2", "rel_l2"],
    "params": ["model_params", "num_params", "params"],
    "num_samples": ["num_samples"],
    "num_nodes": ["num_nodes"],
}

summary = []
print("\n=== AGGREGATE ===")
for case in ["full_fp32", "full_ampfp16", "spectral_fp32", "layer2_fp32"]:
    rs = groups.get(case, [])
    print(f"\n[{case}] {PRETTY.get(case, case)} n={len(rs)}")
    row = {
        "case": case,
        "label": PRETTY.get(case, case),
        "n": len(rs),
        "runs": ";".join(r["run_ts"] for r in rs),
    }

    for m, aliases in metric_alias.items():
        vals = [fnum(get_first(r, aliases)) for r in rs]
        mu = mean(vals)
        sd = stdev(vals)
        cv = sd / mu * 100.0 if mu not in (None, 0) else None
        row[m + "_mean"] = mu
        row[m + "_std"] = sd
        row[m + "_cv_pct"] = cv
        print(f"  {m}: mean={mu} std={sd} cv={cv}")

    summary.append(row)

fields = sorted({k for r in summary for k in r.keys()})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in summary:
        w.writerow(r)

print("\nSaved:", OUT)
