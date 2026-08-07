from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ART = Path("results/artifacts")
OUT = ART / "latex_tables"
OUT.mkdir(parents=True, exist_ok=True)


def fmt_float(x, nd=3):
    if x is None or x == "":
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def fmt_int(x):
    if x is None or x == "":
        return "--"
    try:
        return f"{int(float(x)):,}"
    except Exception:
        return str(x)


def esc(s):
    s = "" if s is None else str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
    )


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"[WRITE] {path}")


def make_long_energy_table():
    path = ART / "paper_fno_long_energy_table.csv"
    rows = read_csv(path)

    keep_tags = [
        "burgers_base_r2048_ts_fp32_strict_energy",
        "burgers_base_r2048_ts_tf32_energy",
        "darcy_base_r85_ts_fp32_strict_energy",
        "darcy_base_r141_ts_fp32_strict_energy",
        "darcy_base_r141_ts_tf32_energy",
        "darcy_base_r211_ts_fp32_strict_energy",
        "darcy_base_r281_ts_fp32_strict_energy",
        "darcy_base_r281_on421_ts_fp32_strict_energy",
        "darcy_base_r281_on421_ts_tf32_energy",
        "darcy_large_r141_ts_fp32_strict_energy",
        "darcy_large_r141_on421_ts_fp32_strict_energy",
        "darcy_large_r141_on421_ts_tf32_energy",
    ]
    rows_by_tag = {r["tag"]: r for r in rows}
    selected = [rows_by_tag[t] for t in keep_tags if t in rows_by_tag]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Representative long-run FNO energy measurements on Jetson Orin Nano SUPER 8GB. All cases use TorchScript and batch size 1.}")
    lines.append(r"\label{tab:fno_long_energy}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}l l r r r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & Precision & Median ms & P95 ms & Throughput inf/s & Avg power W & P95 power W & Energy / inf J & Peak GPU temp C \\")
    lines.append(r"\midrule")

    label_map = {
        "burgers_base_r2048_ts_fp32_strict_energy": "Burgers base @2048",
        "burgers_base_r2048_ts_tf32_energy": "Burgers base @2048",
        "darcy_base_r85_ts_fp32_strict_energy": r"Darcy base @85$\times$85",
        "darcy_base_r141_ts_fp32_strict_energy": r"Darcy base @141$\times$141",
        "darcy_base_r141_ts_tf32_energy": r"Darcy base @141$\times$141",
        "darcy_base_r211_ts_fp32_strict_energy": r"Darcy base @211$\times$211",
        "darcy_base_r281_ts_fp32_strict_energy": r"Darcy base @281$\times$281",
        "darcy_base_r281_on421_ts_fp32_strict_energy": r"Darcy base @421$\times$421 frontier",
        "darcy_base_r281_on421_ts_tf32_energy": r"Darcy base @421$\times$421 frontier",
        "darcy_large_r141_ts_fp32_strict_energy": r"Darcy large @141$\times$141",
        "darcy_large_r141_on421_ts_fp32_strict_energy": r"Darcy large @421$\times$421 frontier",
        "darcy_large_r141_on421_ts_tf32_energy": r"Darcy large @421$\times$421 frontier",
    }

    for r in selected:
        tag = r["tag"]
        precision = r.get("precision_mode", "")
        precision = "FP32 strict" if precision == "fp32_strict" else precision.upper()
        lines.append(
            f"{label_map.get(tag, esc(tag))} & {esc(precision)} & "
            f"{fmt_float(r.get('median_ms'), 3)} & "
            f"{fmt_float(r.get('p95_ms'), 3)} & "
            f"{fmt_float(r.get('throughput_inf_s'), 2)} & "
            f"{fmt_float(r.get('avg_power_w'), 3)} & "
            f"{fmt_float(r.get('p95_power_w'), 3)} & "
            f"{fmt_float(r.get('energy_per_inf_j'), 4)} & "
            f"{fmt_float(r.get('gpu_temp_peak_c'), 1)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    write(OUT / "table_fno_long_energy.tex", "\n".join(lines))


def make_quality_validity_table():
    path = ART / "paper_fno_quality_validity_table.csv"
    rows = read_csv(path)

    # Main representative rows only; appendix can include all.
    wanted = {
        "burgers_fno_small_seed2",
        "burgers_fno_base_seed3",
        "burgers_fno_large_seed0",
        "darcy_fno_small_seed4",
        "darcy_fno_base_seed0",
        "darcy_fno_large_seed0",
    }
    rows = [r for r in rows if r.get("experiment_name") in wanted]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Predictive-validity gate for deployed FNO checkpoints.}")
    lines.append(r"\label{tab:fno_validity_gate}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{@{}llrrrp{0.24\linewidth}@{}}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model group & Resolution & Params & Selected seed & Held-out relative $L_2$ \\")
    lines.append(r"\midrule")

    for r in rows:
        metric = r.get("quality_metric_value", "")
        metric = "TODO" if metric in ("", None) else fmt_float(metric, 5)
        lines.append(
            f"{esc(r.get('dataset'))} & "
            f"{esc(r.get('scale'))} & "
            f"{esc(r.get('resolution_inferred'))} & "
            f"{fmt_int(r.get('parameter_count_summary'))} & "
            f"{esc(r.get('seed'))} & "
            f"{metric} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    write(OUT / "table_fno_validity_gate.tex", "\n".join(lines))


def make_precision_failure_table():
    path = ART / "paper_fno_precision_failure_summary_table.csv"
    rows = read_csv(path)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Dominant BF16/FP16 failure classes observed for full-path FNO deployment on Jetson.}")
    lines.append(r"\label{tab:fno_precision_failure_summary}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{@{}lllr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Backend & Precision mode & Failure class & Count \\")
    lines.append(r"\midrule")

    for r in rows:
        lines.append(
            f"{esc(r.get('mode'))} & "
            f"{esc(r.get('precision_mode'))} & "
            f"{esc(r.get('failure_class'))} & "
            f"{fmt_int(r.get('num_failures'))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    write(OUT / "table_fno_precision_failure_summary.tex", "\n".join(lines))


def main():
    make_long_energy_table()
    make_quality_validity_table()
    make_precision_failure_table()


if __name__ == "__main__":
    main()