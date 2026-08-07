import os, glob, math
import pandas as pd
import numpy as np

ROOT = os.path.expanduser("~/VirSO/For_Jetson/For_Jetson")
RUN_ROOT = os.path.join(ROOT, "inference_runs")

GROUPS = {
    "full_fp32": {
        "label": "Full spectral--spatial",
        "mode": "FP32",
        "runs": [f"bench2_full_fp32_r{i}" for i in range(2, 7)],
        "baseline": None,
    },
    "full_ampfp16": {
        "label": "Full spectral--spatial",
        "mode": "AMP FP16",
        "runs": [f"bench2_full_ampfp16_r{i}" for i in range(1, 6)],
        "baseline": "full_fp32",
    },
    "spectral_fp32": {
        "label": "Spectral-only",
        "mode": "FP32",
        "runs": [f"bench2_spectral_fp32_r{i}" for i in range(1, 4)],
        "baseline": None,
    },
    "spectral_ampfp16": {
        "label": "Spectral-only",
        "mode": "AMP FP16",
        "runs": [f"bench2_spectral_ampfp16_r{i}" for i in range(1, 4)],
        "baseline": "spectral_fp32",
    },
    "layer2_fp32": {
        "label": "Two-layer spectral--spatial",
        "mode": "FP32",
        "runs": [f"bench2_layer2_fp32_r{i}" for i in range(1, 4)],
        "baseline": None,
    },
    "layer2_ampfp16": {
        "label": "Two-layer spectral--spatial",
        "mode": "AMP FP16",
        "runs": [f"bench2_layer2_ampfp16_r{i}" for i in range(1, 4)],
        "baseline": "layer2_fp32",
    },
}

def read_kv_csv(path):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if {"field", "value"}.issubset(df.columns):
        return {str(k): v for k, v in zip(df["field"], df["value"])}
    if len(df) > 0:
        return df.iloc[0].to_dict()
    return {}

def first_float(d, keys):
    for k in keys:
        if k in d and pd.notna(d[k]) and str(d[k]) != "":
            try:
                return float(d[k])
            except Exception:
                pass
    return np.nan

def collect_run(run_ts):
    run_dir = os.path.join(RUN_ROOT, run_ts)
    report_dir = os.path.join(run_dir, "reports")
    edge = read_kv_csv(os.path.join(report_dir, f"virso_edge_summary_{run_ts}.csv"))
    power = read_kv_csv(os.path.join(report_dir, f"jetson_power_summary_{run_ts}.csv"))
    reviewer = read_kv_csv(os.path.join(report_dir, f"jetson_reviewer_report_{run_ts}.csv"))

    d = {}
    d.update(edge)
    d.update(power)
    d.update(reviewer)

    status_path = os.path.join(run_dir, "logs", f"run_status_{run_ts}.txt")
    if os.path.exists(status_path):
        status = open(status_path).read().strip()
    else:
        status = "MISSING"

    rec = {
        "run_ts": run_ts,
        "exists": os.path.isdir(run_dir),
        "status": status,
        "params": first_float(d, ["model_params", "params", "trainable_params"]),
        "median_ms": first_float(d, ["median_ms", "p50_latency_ms", "median_latency_ms"]),
        "p95_ms": first_float(d, ["p95_ms", "p95_latency_ms"]),
        "avg_w": first_float(d, ["avg_power_w", "avg_power_W", "power_avg_w", "avg_w"]),
        "j_per_inf": first_float(d, [
            "j_per_it", "energy_j_per_sample_from_total",
            "energy_j_per_sample", "j_per_sample",
            "energy_j_per_iter", "energy_per_inference_j"
        ]),
        "peak_ram_mb": first_float(d, ["peak_ram_mb", "peak_board_ram_mb", "board_ram_peak_mb", "peak_ram"]),
        "rel_l2": first_float(d, ["rel_l2", "relative_l2", "avg_relative_l2", "avg_total_loss"]),
        "amp_mode": d.get("amp_mode", ""),
        "amp_resolved_dtype": d.get("amp_resolved_dtype", ""),
    }
    return rec

rows = []
missing = []
for key, g in GROUPS.items():
    rs = []
    for run_ts in g["runs"]:
        rec = collect_run(run_ts)
        rec["group"] = key
        rec["label"] = g["label"]
        rec["mode"] = g["mode"]
        rs.append(rec)
        rows.append(rec)
        if (not rec["exists"]) or rec["status"] not in ("0", 0):
            missing.append((key, run_ts, rec["exists"], rec["status"]))

raw = pd.DataFrame(rows)
raw_path = os.path.join(RUN_ROOT, "virso_precision_sustained_raw.csv")
raw.to_csv(raw_path, index=False)

summary_rows = []
for key, g in GROUPS.items():
    sub = raw[(raw["group"] == key) & (raw["exists"]) & (raw["status"].astype(str) == "0")]
    out = {
        "group": key,
        "label": g["label"],
        "mode": g["mode"],
        "n": len(sub),
        "runs": ";".join(sub["run_ts"].tolist()),
        "params_m": sub["params"].mean() / 1e6 if len(sub) else np.nan,
        "median_ms": sub["median_ms"].mean() if len(sub) else np.nan,
        "p95_ms": sub["p95_ms"].mean() if len(sub) else np.nan,
        "avg_w": sub["avg_w"].mean() if len(sub) else np.nan,
        "j_per_inf": sub["j_per_inf"].mean() if len(sub) else np.nan,
        "peak_ram_gb": sub["peak_ram_mb"].mean() / 1024 if len(sub) else np.nan,
        "rel_l2_pct": sub["rel_l2"].mean() * 100 if len(sub) else np.nan,
        "baseline": g["baseline"],
    }
    summary_rows.append(out)

summary = pd.DataFrame(summary_rows)

# Add speedup/reduction against matching FP32 path
summary["latency_speedup"] = np.nan
summary["energy_reduction_pct"] = np.nan
summary["rel_l2_delta_pctpt"] = np.nan

lookup = {r["group"]: r for _, r in summary.iterrows()}
for idx, r in summary.iterrows():
    b = r["baseline"]
    if isinstance(b, str) and b in lookup:
        base = lookup[b]
        if pd.notna(base["median_ms"]) and pd.notna(r["median_ms"]):
            summary.loc[idx, "latency_speedup"] = base["median_ms"] / r["median_ms"]
        if pd.notna(base["j_per_inf"]) and pd.notna(r["j_per_inf"]):
            summary.loc[idx, "energy_reduction_pct"] = 100.0 * (base["j_per_inf"] - r["j_per_inf"]) / base["j_per_inf"]
        if pd.notna(base["rel_l2_pct"]) and pd.notna(r["rel_l2_pct"]):
            summary.loc[idx, "rel_l2_delta_pctpt"] = r["rel_l2_pct"] - base["rel_l2_pct"]

summary_path = os.path.join(RUN_ROOT, "virso_precision_sustained_summary.csv")
summary.to_csv(summary_path, index=False)

def fmt(x, nd=3):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"

def fmt_pct(x, nd=1):
    return "--" if pd.isna(x) else f"{x:.{nd}f}\\%"

def fmt_speed(x):
    return "--" if pd.isna(x) else f"{x:.2f}$\\times$"

def fmt_delta(x):
    if pd.isna(x):
        return "--"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}"

table_lines = []
table_lines.append(r"\begin{table}[H]")
table_lines.append(r"\centering")
table_lines.append(r"\caption{VIRSO reduced-precision sustained telemetry on the HeatExchanger request path. Values are averaged over successful repeated Jetson runs using board-level \texttt{tegrastats} telemetry. Latency speedup, energy reduction, and relative-$L_2$ change are computed relative to the FP32 row for the same path.}")
table_lines.append(r"\label{tab:virso_precision_sustained}")
table_lines.append(r"\small")
table_lines.append(r"\setlength{\tabcolsep}{3.5pt}")
table_lines.append(r"\renewcommand{\arraystretch}{1.08}")
table_lines.append(r"\begin{tabular}{@{}llrrrrrrr@{}}")
table_lines.append(r"\toprule")
table_lines.append(r"Path & Mode & $n$ & Med. ms & P95 ms & Speedup & J/inf & Energy red. & Rel-$L_2$ (\%) \\")
table_lines.append(r"\midrule")

order = ["full_fp32", "full_ampfp16", "spectral_fp32", "spectral_ampfp16", "layer2_fp32", "layer2_ampfp16"]
for key in order:
    r = summary[summary["group"] == key].iloc[0]
    speed = "--" if r["mode"] == "FP32" else fmt_speed(r["latency_speedup"])
    ered = "--" if r["mode"] == "FP32" else fmt_pct(r["energy_reduction_pct"], 1)
    table_lines.append(
        f"{r['label']} & {r['mode']} & {int(r['n'])} & "
        f"{fmt(r['median_ms'],3)} & {fmt(r['p95_ms'],3)} & {speed} & "
        f"{fmt(r['j_per_inf'],3)} & {ered} & {fmt(r['rel_l2_pct'],2)} \\\\"
    )
    if key in ["full_ampfp16", "spectral_ampfp16"]:
        table_lines.append(r"\midrule")

table_lines.append(r"\bottomrule")
table_lines.append(r"\end{tabular}")
table_lines.append(r"\end{table}")

latex = "\n".join(table_lines)
tex_path = os.path.join(RUN_ROOT, "virso_precision_sustained_table.tex")
with open(tex_path, "w") as f:
    f.write(latex + "\n")

print("\n===== MISSING / FAILED RUNS =====")
if missing:
    for item in missing:
        print(item)
else:
    print("None")

print("\n===== RAW CSV =====")
print(raw_path)

print("\n===== SUMMARY CSV =====")
print(summary_path)
print(summary.to_string(index=False))

print("\n===== LATEX TABLE =====")
print(tex_path)
print(latex)
