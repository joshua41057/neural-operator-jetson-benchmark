import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("inference_runs")
OUT_RAW = ROOT / "virso_tf32_bf16_sustained_completion_wide_raw.csv"
OUT_SUM = ROOT / "virso_tf32_bf16_sustained_completion_summary_fixed.csv"

def parse_run_name(name):
    m = re.match(r"completion_(full|spectral|layer2)_(tf32|bf16)_r(\d+)", name)
    if not m:
        return None
    return {
        "case": m.group(1),
        "precision": m.group(2),
        "rep": int(m.group(3)),
    }

def read_report_csv(path):
    df = pd.read_csv(path)

    # Long format: field,value
    if set(["field", "value"]).issubset(df.columns):
        out = {}
        for _, r in df.iterrows():
            out[str(r["field"])] = r["value"]
        return out

    # Alternate long format: metric,value
    if set(["metric", "value"]).issubset(df.columns):
        out = {}
        for _, r in df.iterrows():
            out[str(r["metric"])] = r["value"]
        return out

    # Wide format: one row with many columns
    if len(df) >= 1:
        return df.iloc[0].to_dict()

    return {}

rows = []

for d in sorted(ROOT.glob("completion_*")):
    meta = parse_run_name(d.name)
    if meta is None:
        continue

    row = {"run_ts": d.name, **meta}

    status_files = list((d / "logs").glob("run_status_*.txt"))
    if status_files:
        row["exit_status"] = status_files[0].read_text().strip()

    # Read all relevant CSV reports and merge fields.
    for p in sorted((d / "reports").glob("*.csv")):
        data = read_report_csv(p)
        for k, v in data.items():
            row[k] = v

    rows.append(row)

raw = pd.DataFrame(rows)

# Normalize common numeric columns.
rename = {
    "p50_latency_ms": "median_ms",
    "avg_latency_ms": "avg_ms",
    "p95_latency_ms": "p95_ms",
    "avg_power_w": "avg_power_w",
    "energy_j_per_sample_from_total": "j_per_it",
    "energy_j_per_sample_est": "j_per_it_est",
    "avg_total_loss": "rel_l2",
    "peak_ram_used_mb": "peak_ram_mb",
    "max_temp_c": "max_temp_c",
    "model_params": "params",
    "num_nodes": "num_nodes",
    "num_samples": "num_samples",
}

for old, new in rename.items():
    if old in raw.columns:
        raw[new] = pd.to_numeric(raw[old], errors="coerce")

# Keep useful columns first.
front = [
    "run_ts", "case", "precision", "rep", "exit_status",
    "median_ms", "avg_ms", "p95_ms", "avg_power_w", "j_per_it",
    "rel_l2", "peak_ram_mb", "max_temp_c", "params", "num_nodes", "num_samples",
]
cols = [c for c in front if c in raw.columns] + [c for c in raw.columns if c not in front]
raw = raw[cols]
raw.to_csv(OUT_RAW, index=False)

print("\n=== WIDE RAW ===")
print(raw[cols[:16]].to_string(index=False))

need = ["median_ms", "avg_ms", "p95_ms", "avg_power_w", "j_per_it", "rel_l2", "peak_ram_mb", "max_temp_c"]
missing = [c for c in need if c not in raw.columns]
if missing:
    print("\nERROR: missing normalized columns:", missing)
    print("Available columns:")
    print(list(raw.columns))
    raise SystemExit(1)

summary = (
    raw.groupby(["case", "precision"])
    .agg(
        n=("run_ts", "count"),
        median_ms_mean=("median_ms", "mean"),
        median_ms_std=("median_ms", "std"),
        avg_ms_mean=("avg_ms", "mean"),
        p95_ms_mean=("p95_ms", "mean"),
        avg_power_w_mean=("avg_power_w", "mean"),
        avg_power_w_std=("avg_power_w", "std"),
        j_per_it_mean=("j_per_it", "mean"),
        j_per_it_std=("j_per_it", "std"),
        rel_l2_mean=("rel_l2", "mean"),
        rel_l2_std=("rel_l2", "std"),
        peak_ram_mb_mean=("peak_ram_mb", "mean"),
        max_temp_c_mean=("max_temp_c", "mean"),
    )
    .reset_index()
)

# Existing FP32 baselines from your paper summary.
fp32_baseline = {
    "full":     {"median_ms": 638.258, "j_per_it": 4.792},
    "spectral": {"median_ms": 77.592,  "j_per_it": 0.560},
    "layer2":   {"median_ms": 142.838, "j_per_it": 1.095},
}

def calc_speedup(row):
    base = fp32_baseline[row["case"]]["median_ms"]
    return base / row["median_ms_mean"]

def calc_energy_red(row):
    base = fp32_baseline[row["case"]]["j_per_it"]
    return 100.0 * (base - row["j_per_it_mean"]) / base

summary["speedup_vs_fp32"] = summary.apply(calc_speedup, axis=1)
summary["energy_reduction_pct_vs_fp32"] = summary.apply(calc_energy_red, axis=1)

summary.to_csv(OUT_SUM, index=False)

print("\n=== SUMMARY ===")
print(summary.to_string(index=False))

print("\nWrote:")
print(" ", OUT_RAW)
print(" ", OUT_SUM)
