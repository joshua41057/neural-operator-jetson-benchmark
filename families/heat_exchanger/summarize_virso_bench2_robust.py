#!/usr/bin/env python3
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict

ROOT = os.path.expanduser("~/VirSO/For_Jetson/For_Jetson")
REG = os.path.join(ROOT, "inference_runs", "virso_direct_benchmark_registry.csv")
OUT = os.path.join(ROOT, "inference_runs", "virso_bench2_paper_summary.csv")
RAW_OUT = os.path.join(ROOT, "inference_runs", "virso_bench2_joined_raw.csv")

CASE_CLEAN = {
    "full_fp32_feas": "full_fp32",
    "full_fp32": "full_fp32",
    "full_ampfp16": "full_ampfp16",
    "spectral_fp32": "spectral_fp32",
    "layer2_fp32": "layer2_fp32",
}

LABEL = {
    "full_fp32": "VIRSO full FP32",
    "full_ampfp16": "VIRSO full AMP FP16",
    "spectral_fp32": "VIRSO spectral-only FP32",
    "layer2_fp32": "VIRSO 2-layer FP32",
}

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def parse_num(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    # tolerate values like "562.9 ms", "7.5 W", "5,123"
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def load_csv_flexible(path):
    """
    Returns normalized dict from a CSV that may be:
    - one-row wide table
    - many-row key/value table
    - metric,value table
    """
    out = {}
    if not os.path.exists(path):
        return out

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return out

    cols = list(rows[0].keys())
    ncols = [norm(c) for c in cols]

    # Long format: metric,value or key,value.
    metric_candidates = {"metric", "name", "key", "field", "item"}
    value_candidates = {"value", "val", "mean", "measurement"}

    metric_col = None
    value_col = None
    for c, nc in zip(cols, ncols):
        if nc in metric_candidates:
            metric_col = c
        if nc in value_candidates:
            value_col = c

    if metric_col and value_col:
        for r in rows:
            k = norm(r.get(metric_col, ""))
            v = r.get(value_col, "")
            if k:
                out[k] = v
        return out

    # Two-column anonymous-ish key/value fallback.
    if len(cols) == 2 and len(rows) > 1:
        c0, c1 = cols
        # If first column has nonnumeric keys and second has numeric-ish values.
        nonnumeric_keys = sum(parse_num(r.get(c0)) is None for r in rows[:10])
        numeric_vals = sum(parse_num(r.get(c1)) is not None for r in rows[:10])
        if nonnumeric_keys >= max(1, min(5, len(rows[:10]) // 2)) and numeric_vals >= 1:
            for r in rows:
                k = norm(r.get(c0, ""))
                v = r.get(c1, "")
                if k:
                    out[k] = v
            return out

    # Wide format: use first row.
    for k, v in rows[0].items():
        out[norm(k)] = v

    return out

def get_metric(d, aliases):
    # exact normalized alias first
    for a in aliases:
        na = norm(a)
        if na in d:
            return parse_num(d[na]), na

    # substring fallback
    keys = list(d.keys())
    for a in aliases:
        toks = [t for t in norm(a).split("_") if t]
        for k in keys:
            if all(t in k for t in toks):
                val = parse_num(d[k])
                if val is not None:
                    return val, k
    return None, None

METRICS = {
    "median_ms": [
        "p50_latency_ms", "median_latency_ms", "latency_p50_ms", "median_ms",
        "p50_ms", "latency_median_ms"
    ],
    "p95_ms": [
        "p95_latency_ms", "latency_p95_ms", "p95_ms"
    ],
    "avg_ms": [
        "avg_latency_ms", "mean_latency_ms", "latency_mean_ms", "average_latency_ms"
    ],
    "avg_power_w": [
        "avg_power_w", "mean_power_w", "avg_total_power_w",
        "vdd_in_avg_w", "avg_vdd_in_w", "vddin_avg_w",
        "power_avg_w", "total_power_avg_w"
    ],
    "j_per_it": [
        "energy_j_per_sample_from_total", "energy_j_per_sample",
        "j_per_sample", "energy_j_per_iter", "j_per_it",
        "energy_per_sample_j", "energy_per_inference_j"
    ],
    "peak_ram_mb": [
        "peak_ram_used_mb", "peak_ram_mb", "max_ram_used_mb",
        "ram_peak_mb", "peak_memory_mb", "board_ram_peak_mb"
    ],
    "max_temp_c": [
        "max_temp_c", "max_gpu_temp_c", "max_cpu_temp_c",
        "temperature_max_c", "gpu_temp_max_c"
    ],
    "rel_l2": [
        "avg_total_loss", "mean_rel_l2", "relative_l2", "rel_l2",
        "total_rel_l2", "mean_relative_l2"
    ],
    "params": [
        "model_params", "num_params", "params", "parameter_count"
    ],
    "num_samples": [
        "num_samples", "samples", "test_samples"
    ],
    "num_nodes": [
        "num_nodes", "nodes", "n_nodes"
    ],
}

if not os.path.exists(REG):
    raise SystemExit(f"Missing registry: {REG}")

with open(REG, newline="") as f:
    reg = list(csv.DictReader(f))

records = []
detected = defaultdict(dict)

for r in reg:
    raw_case = r.get("case")
    if raw_case not in CASE_CLEAN:
        continue

    run_ts = r["run_ts"]
    run_dir = os.path.join(ROOT, "inference_runs", run_ts)
    report_dir = os.path.join(run_dir, "reports")

    merged = {}
    # Load all report csvs, not only reviewer report.
    for p in sorted(glob.glob(os.path.join(report_dir, "*.csv"))):
        loaded = load_csv_flexible(p)
        # later files can add/overwrite, okay
        merged.update(loaded)

    rec = dict(r)
    rec["case_clean"] = CASE_CLEAN[raw_case]
    rec["run_dir"] = run_dir
    rec["has_report"] = bool(merged)
    rec["_merged_keys"] = ";".join(sorted(merged.keys()))

    for m, aliases in METRICS.items():
        val, key = get_metric(merged, aliases)
        rec[m] = val
        rec[m + "_source_key"] = key
        if key:
            detected[rec["case_clean"]][m] = key

    records.append(rec)

print("=== PER-RUN RECORDS ===")
for r in records:
    print(
        f"{r.get('run_ts')} case={r.get('case_clean')} "
        f"exit={r.get('exit_status')} has_report={r.get('has_report')} "
        f"avg_ms={r.get('avg_ms')} median={r.get('median_ms')} "
        f"p95={r.get('p95_ms')} power={r.get('avg_power_w')} "
        f"J={r.get('j_per_it')} ram={r.get('peak_ram_mb')} rel_l2={r.get('rel_l2')}"
    )

print("\n=== DETECTED SOURCE KEYS ===")
for case, d in detected.items():
    print(f"[{case}]")
    for m, k in sorted(d.items()):
        print(f"  {m} <- {k}")

groups = defaultdict(list)
for r in records:
    if str(r.get("exit_status")) != "0":
        continue
    if not r.get("has_report"):
        continue
    groups[r["case_clean"]].append(r)

def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else None

def stdev(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0

summary = []
print("\n=== AGGREGATE ===")
for case in ["full_fp32", "full_ampfp16", "spectral_fp32", "layer2_fp32"]:
    rs = groups.get(case, [])
    print(f"\n[{case}] {LABEL.get(case, case)} n={len(rs)}")
    row = {
        "case": case,
        "label": LABEL.get(case, case),
        "n": len(rs),
        "runs": ";".join(r["run_ts"] for r in rs),
    }

    for m in METRICS.keys():
        vals = [r.get(m) for r in rs]
        mu = mean(vals)
        sd = stdev(vals)
        cv = (sd / mu * 100.0) if mu not in (None, 0) else None
        row[m + "_mean"] = mu
        row[m + "_std"] = sd
        row[m + "_cv_pct"] = cv
        print(f"  {m}: mean={mu} std={sd} cv={cv}")

    summary.append(row)

# Save raw joined records.
raw_fields = sorted({k for r in records for k in r.keys() if k != "_merged_keys"}) + ["_merged_keys"]
with open(RAW_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    for r in records:
        w.writerow(r)

fields = sorted({k for r in summary for k in r.keys()})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in summary:
        w.writerow(r)

print("\nSaved:")
print(" ", RAW_OUT)
print(" ", OUT)
