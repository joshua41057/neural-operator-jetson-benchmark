#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CASE_LABELS = {
    "burgers_base_r2048_ts_fp32": "Burgers base @2048",
    "darcy_base_r141_ts_fp32": "Darcy base @141$\\times$141",
    "darcy_base_r281_ts_fp32": "Darcy base @281$\\times$281",
    "darcy_base_r281_ts_fp16_native": "Darcy base @281$\\times$281 FP16",
    "darcy_large_r141_ts_fp32": "Darcy large @141$\\times$141",
}

CASE_ROLES = {
    "burgers_base_r2048_ts_fp32": "lightweight 1D baseline",
    "darcy_base_r141_ts_fp32": "controlled 2D base",
    "darcy_base_r281_ts_fp32": "high-resolution 2D base",
    "darcy_base_r281_ts_fp16_native": "high-resolution precision path",
    "darcy_large_r141_ts_fp32": "model-capacity scaling",
}

CASE_DIAGNOSES = {
    "burgers_base_r2048_ts_fp32":
        "Low-ms branch--trunk path; launch mix is dense and elementwise rather than spectral.",
    "darcy_base_r141_ts_fp32":
        "2D output grid exposes trunk evaluation, coordinate construction, and materialization cost.",
    "darcy_base_r281_ts_fp32":
        "Resolution growth amplifies dense/trunk and output-grid materialization cost.",
    "darcy_base_r281_ts_fp16_native":
        "Native FP16 lowers storage and dense-kernel cost, while elementwise/materialization kernels remain visible.",
    "darcy_large_r141_ts_fp32":
        "Increasing model capacity raises dense MLP contribution even at controlled resolution.",
}


def latex_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def short_kernel_name(name: str) -> str:
    n = name.strip()

    if "ampere_sgemm" in n:
        m = re.search(r"ampere_sgemm[_a-zA-Z0-9]*", n)
        return m.group(0) if m else "ampere_sgemm"
    if "gemv" in n.lower():
        return "gemv / cublas GEMV"
    if "splitKreduce" in n:
        return "cublasLt splitKreduce"
    if "globalKernel" in n:
        return "cublasLt epilogue globalKernel"
    if "upsample" in n:
        return "upsample/interpolate"
    if "linspace" in n:
        return "linspace coordinate kernel"
    if "fused_mul_mul_sin_cos_cat" in n:
        return "fused sin/cos coordinate kernel"
    if "CatArrayBatchedCopy" in n or "cat" in n.lower():
        return "cat/materialization"
    if "direct_copy" in n or "copy_kernel" in n:
        return "direct copy/cast"
    if "unrolled_elementwise_kernel" in n:
        return "unrolled elementwise/cast"
    if "vectorized_elementwise_kernel" in n:
        if "Gelu" in n or "gelu" in n.lower():
            return "vectorized GELU"
        return "vectorized elementwise"
    if "elementwise_kernel_with_index" in n:
        return "indexed elementwise"
    if "elementwise_kernel" in n:
        return "elementwise"
    if "FillFunctor" in n:
        return "fill/materialization"
    if "fused" in n.lower():
        return n[:48]
    if len(n) > 52:
        return n[:49] + "..."
    return n


def classify_kernel(name: str) -> str:
    n = name.lower()

    if any(k in n for k in ["sgemm", "gemm", "gemv", "cublas", "cutlass", "splitkreduce", "globalkernel"]):
        return "dense/recombination"
    if any(k in n for k in ["upsample", "interpolate"]):
        return "sensor resampling"
    if any(k in n for k in ["linspace", "meshgrid", "sin_cos", "sincos", "fused_mul_mul_sin_cos"]):
        return "coordinate/trunk setup"
    if any(k in n for k in ["copy", "catarray", "cat", "fillfunctor", "unrolled_elementwise_kernel"]):
        return "movement/materialization"
    if any(k in n for k in ["gelu", "elementwise", "fused"]):
        return "activation/elementwise"
    return "other"


def find_col(header: list[str], patterns: list[str], avoid: list[str] | None = None) -> int | None:
    avoid = avoid or []
    candidates = []
    for i, h in enumerate(header):
        hs = h.lower()
        if all(p.lower() in hs for p in patterns) and not any(a.lower() in hs for a in avoid):
            candidates.append(i)
    if not candidates:
        return None

    # Prefer rollup/average metrics over metadata-like columns.
    priority_words = [
        ".sum", ".avg", ".pct", "pct_of_peak", "per_second",
        "duration", "throughput", "hit_rate", "active",
    ]
    def score(idx: int) -> int:
        h = header[idx].lower()
        return sum(w in h for w in priority_words)

    return sorted(candidates, key=score, reverse=True)[0]


def parse_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s in {"nan", "NaN", "--", "n/a", "N/A"}:
        return None
    s = s.replace(",", "")
    s = s.replace("%", "")
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def metric_cols(header: list[str]) -> dict[str, int | None]:
    return {
        "time_ns": find_col(header, ["gpu__time_duration"], avoid=["start", "end"]),
        "sm_pct": find_col(header, ["sm__throughput"]),
        "mem_pct": (
            find_col(header, ["gpu__compute_memory_throughput"])
            or find_col(header, ["dram"])
            or find_col(header, ["lts__throughput"])
        ),
        "l1tex_pct": find_col(header, ["l1tex__throughput"]),
        "lts_pct": find_col(header, ["lts__throughput"]),
        "l2_hit_pct": (
            find_col(header, ["lts__t_sector_hit_rate"])
            or find_col(header, ["lts__t_request_hit_rate"])
            or find_col(header, ["hit_rate"])
        ),
        "occupancy_pct": (
            find_col(header, ["achieved_occupancy"])
            or find_col(header, ["warps_active", "pct"])
            or find_col(header, ["occupancy"])
        ),
        "eligible_warps": find_col(header, ["warps_eligible"]),
    }


def parse_ncu_raw_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    with path.open(errors="ignore", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        return [], {}

    header = rows[0]
    idx_kernel = header.index("Kernel Name") if "Kernel Name" in header else 4
    cols = metric_cols(header)

    out = []
    for r in rows[2:]:
        if not r or not r[0].strip().isdigit():
            continue
        if len(r) <= idx_kernel:
            continue

        name = r[idx_kernel].strip()
        if not name:
            continue

        def get_metric(key: str) -> float | None:
            c = cols.get(key)
            if c is None or c >= len(r):
                return None
            return parse_float(r[c])

        time_ns = get_metric("time_ns")
        time_ms = None if time_ns is None else time_ns / 1.0e6

        out.append({
            "kernel_name": name,
            "kernel_short": short_kernel_name(name),
            "class": classify_kernel(name),
            "time_ms": time_ms,
            "sm_pct": get_metric("sm_pct"),
            "mem_pct": get_metric("mem_pct"),
            "l1tex_pct": get_metric("l1tex_pct"),
            "lts_pct": get_metric("lts_pct"),
            "l2_hit_pct": get_metric("l2_hit_pct"),
            "occupancy_pct": get_metric("occupancy_pct"),
            "eligible_warps": get_metric("eligible_warps"),
        })

    return out, cols


def fmt_float(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def fmt_ms(x: float | None) -> str:
    if x is None:
        return "--"
    if x < 0.01:
        return f"{x:.4f}"
    if x < 0.1:
        return f"{x:.3f}"
    return f"{x:.2f}"


def aggregate_classes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["class"] for r in rows)
    time_by_class = defaultdict(float)
    has_time = False

    for r in rows:
        if r["time_ms"] is not None:
            has_time = True
            time_by_class[r["class"]] += r["time_ms"]

    top_counts = counts.most_common(3)
    if has_time:
        top_time = sorted(time_by_class.items(), key=lambda kv: kv[1], reverse=True)[:3]
    else:
        top_time = []

    return {
        "counts": counts,
        "top_counts": top_counts,
        "top_time": top_time,
        "has_time": has_time,
    }


def write_main_table(profile_rows: dict[str, list[dict[str, Any]]], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Compact DeepONet profiling diagnosis from representative Nsight Compute runs. The table reports operation classes rather than exhaustive kernels; detailed kernel-level evidence is retained in Appendix~\ref{app:deeponet_profiling_details}.}")
    lines.append(r"\label{tab:deeponet_profile_diagnosis_matrix}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{@{}p{0.20\linewidth}p{0.20\linewidth}p{0.26\linewidth}p{0.26\linewidth}@{}}")
    lines.append(r"\toprule")
    lines.append(r"Profile case & Role & Dominant operation classes & Diagnosis \\")
    lines.append(r"\midrule")

    for pid in CASE_LABELS:
        rows = profile_rows.get(pid, [])
        agg = aggregate_classes(rows)
        cls = ", ".join([f"{c} ({n})" for c, n in agg["top_counts"]])
        lines.append(
            f"{CASE_LABELS[pid]} & "
            f"{latex_escape(CASE_ROLES.get(pid, 'representative'))} & "
            f"{latex_escape(cls)} & "
            f"{latex_escape(CASE_DIAGNOSES.get(pid, 'Representative branch--trunk runtime path.'))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n")


def select_detail_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    # Prefer kernels with timing metrics. Keep class diversity when possible.
    timed = [r for r in rows if r["time_ms"] is not None]
    base = timed if timed else rows

    by_class = defaultdict(list)
    for r in base:
        by_class[r["class"]].append(r)

    for cls in by_class:
        by_class[cls].sort(key=lambda r: (r["time_ms"] is not None, r["time_ms"] or 0.0), reverse=True)

    selected = []
    for cls in [
        "dense/recombination",
        "coordinate/trunk setup",
        "activation/elementwise",
        "movement/materialization",
        "sensor resampling",
        "other",
    ]:
        if by_class.get(cls):
            selected.append(by_class[cls][0])

    remaining = sorted(
        base,
        key=lambda r: (r["time_ms"] is not None, r["time_ms"] or 0.0),
        reverse=True,
    )
    for r in remaining:
        if len(selected) >= max_rows:
            break
        if r not in selected:
            selected.append(r)

    return selected[:max_rows]


def write_detail_csv(profile_rows: dict[str, list[dict[str, Any]]], out_csv: Path, max_rows: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile_id", "case", "class", "kernel_short", "time_ms",
        "sm_pct", "mem_pct", "l1tex_pct", "lts_pct", "l2_hit_pct",
        "occupancy_pct", "eligible_warps", "kernel_name",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pid in CASE_LABELS:
            for r in select_detail_rows(profile_rows.get(pid, []), max_rows=max_rows):
                w.writerow({
                    "profile_id": pid,
                    "case": CASE_LABELS[pid].replace("$\\times$", "x"),
                    "class": r["class"],
                    "kernel_short": r["kernel_short"],
                    "time_ms": "" if r["time_ms"] is None else f"{r['time_ms']:.6f}",
                    "sm_pct": "" if r["sm_pct"] is None else f"{r['sm_pct']:.4f}",
                    "mem_pct": "" if r["mem_pct"] is None else f"{r['mem_pct']:.4f}",
                    "l1tex_pct": "" if r["l1tex_pct"] is None else f"{r['l1tex_pct']:.4f}",
                    "lts_pct": "" if r["lts_pct"] is None else f"{r['lts_pct']:.4f}",
                    "l2_hit_pct": "" if r["l2_hit_pct"] is None else f"{r['l2_hit_pct']:.4f}",
                    "occupancy_pct": "" if r["occupancy_pct"] is None else f"{r['occupancy_pct']:.4f}",
                    "eligible_warps": "" if r["eligible_warps"] is None else f"{r['eligible_warps']:.4f}",
                    "kernel_name": r["kernel_name"],
                })


def write_detail_tex(profile_rows: dict[str, list[dict[str, Any]]], out_tex: Path, max_rows: int) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Representative DeepONet Nsight Compute kernel-level evidence. NCU timings are profiler-perturbed and are used only for mechanism diagnosis; primary latency is reported from the benchmark harness.}")
    lines.append(r"\label{tab:deeponet_kernel_detail}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{@{}p{0.16\linewidth}p{0.18\linewidth}p{0.23\linewidth}rrrrp{0.18\linewidth}@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & Class & Representative kernel & ms & SM & Mem & L2/LTS & Reading \\")
    lines.append(r"\midrule")

    for pid in CASE_LABELS:
        for r in select_detail_rows(profile_rows.get(pid, []), max_rows=max_rows):
            l2_like = r["l2_hit_pct"]
            if l2_like is None:
                l2_like = r["lts_pct"]
            if l2_like is None:
                l2_like = r["l1tex_pct"]

            reading = {
                "dense/recombination": "dense MLP or basis recombination path",
                "coordinate/trunk setup": "coordinate/trunk setup visible",
                "activation/elementwise": "activation or fused elementwise overhead",
                "movement/materialization": "copy, cast, or tensor materialization",
                "sensor resampling": "branch sensor interpolation",
                "other": "runtime support kernel",
            }.get(r["class"], "runtime support kernel")

            lines.append(
                f"{CASE_LABELS[pid]} & "
                f"{latex_escape(r['class'])} & "
                f"{latex_escape(r['kernel_short'])} & "
                f"{fmt_ms(r['time_ms'])} & "
                f"{fmt_float(r['sm_pct'])} & "
                f"{fmt_float(r['mem_pct'])} & "
                f"{fmt_float(l2_like)} & "
                f"{latex_escape(reading)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n")


def write_appendix(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(r"""
\section{DeepONet Profiling Details}
\label{app:deeponet_profiling_details}

This appendix stores representative DeepONet profiling evidence. The main text reports the compact operation-class diagnosis; this appendix retains kernel-level evidence for reviewer inspection and artifact reproducibility. Nsight Compute is used for mechanism diagnosis rather than primary latency measurement, because the profiler perturbs execution time.

\input{tables/deeponet_kernel_detail}

The DeepONet kernel mix differs from the FNO spectral path. The profiled DeepONet cases do not contain FFT or inverse-FFT kernels. Instead, the visible categories are dense MLP and recombination kernels, coordinate and activation kernels, sensor-resampling kernels, and movement or materialization kernels. This supports the main-text interpretation that DeepONet changes the bottleneck mechanism while retaining resolution sensitivity on the Darcy workload.
""".lstrip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/profiles/deeponet_ncu_basic")
    ap.add_argument("--main-tex", default="tables/deeponet_profile_diagnosis_matrix.tex")
    ap.add_argument("--detail-tex", default="tables/deeponet_kernel_detail.tex")
    ap.add_argument("--detail-csv", default="results/artifacts/deeponet_ncu_kernel_detail.csv")
    ap.add_argument("--appendix-tex", default="appendices/h_deeponet_profiling_details.tex")
    ap.add_argument("--max-detail-rows-per-case", type=int, default=4)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"Missing raw-dir: {raw_dir}")

    profile_rows: dict[str, list[dict[str, Any]]] = {}
    metric_availability: dict[str, dict[str, bool]] = {}

    for p in sorted(raw_dir.glob("*_raw.csv")):
        pid = p.name.replace("_raw.csv", "")
        if pid not in CASE_LABELS:
            continue
        rows, cols = parse_ncu_raw_csv(p)
        profile_rows[pid] = rows
        metric_availability[pid] = {k: (v is not None) for k, v in cols.items()}

    missing = [pid for pid in CASE_LABELS if pid not in profile_rows]
    if missing:
        raise SystemExit(f"Missing expected raw NCU CSVs for: {missing}")

    write_main_table(profile_rows, Path(args.main_tex))
    write_detail_tex(profile_rows, Path(args.detail_tex), args.max_detail_rows_per_case)
    write_detail_csv(profile_rows, Path(args.detail_csv), args.max_detail_rows_per_case)
    write_appendix(Path(args.appendix_tex))

    metric_json = Path("results/artifacts/deeponet_ncu_metric_availability_from_table_script.json")
    metric_json.parent.mkdir(parents=True, exist_ok=True)
    import json
    metric_json.write_text(json.dumps(metric_availability, indent=2, sort_keys=True))

    print(f"Wrote {args.main_tex}")
    print(f"Wrote {args.detail_tex}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.appendix_tex}")
    print(f"Wrote {metric_json}")


if __name__ == "__main__":
    main()
