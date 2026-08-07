#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


ROOT = Path(".")
ART = ROOT / "results" / "artifacts"

PROFILE_PLAN = ART / "deeponet_profile_plan.csv"
NSYS_FORWARD = ART / "deeponet_nsys_forward_summary.csv"
NCU_BASIC = ART / "deeponet_ncu_basic_artifact_summary.csv"

MAIN_TEX = ROOT / "tables" / "deeponet_profile_diagnosis_matrix.tex"
APP_TEX = ROOT / "appendices" / "h_deeponet_profiling_details.tex"


MAIN_PROFILE_ORDER = [
    "burgers_base_r2048_ts_fp32",
    "darcy_base_r141_ts_fp32",
    "darcy_base_r281_ts_fp32",
    "darcy_base_r281_ts_fp16_native",
    "darcy_large_r141_ts_fp32",
]


CASE_NAMES = {
    "burgers_base_r2048_ts_fp32": r"Burgers base @2048",
    "darcy_base_r141_ts_fp32": r"Darcy base @141$\times$141",
    "darcy_base_r281_ts_fp32": r"Darcy base @281$\times$281",
    "darcy_base_r281_ts_fp16_native": r"Darcy base @281$\times$281",
    "darcy_large_r141_ts_fp32": r"Darcy large @141$\times$141",
    "darcy_base_r281_ts_tf32": r"Darcy base @281$\times$281",
    "darcy_base_r281_ts_bf16_autocast": r"Darcy base @281$\times$281",
    "darcy_base_r281_ts_fp16_autocast": r"Darcy base @281$\times$281",
}


DIAGNOSIS_TEXT = {
    "burgers_base_r2048_ts_fp32":
        "Low-latency 1D case; dense and elementwise launches are visible, while spatial working-set pressure remains limited.",
    "darcy_base_r141_ts_fp32":
        "Controlled 2D case; trunk evaluation, coordinate construction, and recombination begin to dominate the non-spectral path.",
    "darcy_base_r281_ts_fp32":
        "High-resolution 2D case; output-coordinate growth increases dense, elementwise, and movement-related launch pressure.",
    "darcy_base_r281_ts_fp16_native":
        "Reduced-precision dense path; FP16 native lowers latency and memory pressure without invoking FFT-limited operators.",
    "darcy_large_r141_ts_fp32":
        "Capacity-scaling case; larger branch/trunk networks increase dense and activation launches at fixed output resolution.",
    "darcy_base_r281_ts_tf32":
        "TF32 path; FP32 tensors remain in use while eligible dense kernels can use TF32 backend arithmetic.",
    "darcy_base_r281_ts_bf16_autocast":
        "Autocast path; additional cast/materialization launches are expected and should be interpreted as runtime-path overhead.",
    "darcy_base_r281_ts_fp16_autocast":
        "Autocast path; additional cast/materialization launches are expected and should be interpreted as runtime-path overhead.",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def by_key(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    out = {}
    for r in rows:
        if r.get(key):
            out[r[key]] = r
    return out


def tex_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def fmt_float(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "--"


def fmt_int(x: Any) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return "--"


def precision_label(p: str) -> str:
    mapping = {
        "fp32_strict": "FP32 strict",
        "fp32": "FP32",
        "tf32": "TF32",
        "bf16_autocast": "BF16 autocast",
        "fp16_autocast": "FP16 autocast",
        "fp16_native": "FP16 native",
    }
    return mapping.get(p, p.replace("_", r"\_"))


def role_label(r: str) -> str:
    mapping = {
        "lightweight_1d_baseline": "lightweight 1D",
        "controlled_2d_base": "controlled 2D",
        "high_resolution_2d_base": "high-res. 2D",
        "high_resolution_2d_precision": "precision stress",
        "model_capacity_scaling": "capacity scaling",
    }
    return mapping.get(r, r.replace("_", " "))


def dominant_class(ncu: Dict[str, str]) -> str:
    counts = {
        "dense": int(float(ncu.get("dense_gemm_gemv_launches", "0") or 0)),
        "elementwise/coord": int(float(ncu.get("elementwise_activation_coord_launches", "0") or 0)),
        "movement": int(float(ncu.get("movement_materialization_launches", "0") or 0)),
        "resampling": int(float(ncu.get("sensor_resampling_launches", "0") or 0)),
        "recombination": int(float(ncu.get("basis_recombination_launches", "0") or 0)),
        "other": int(float(ncu.get("other_launches", "0") or 0)),
    }
    max_count = max(counts.values()) if counts else 0
    winners = [k for k, v in counts.items() if v == max_count and v > 0]
    if not winners:
        return "--"
    if winners[0] == "elementwise/coord":
        return "elementwise / coordinate"
    return winners[0]


def op_mix(ncu: Dict[str, str]) -> str:
    dense = fmt_int(ncu.get("dense_gemm_gemv_launches"))
    elem = fmt_int(ncu.get("elementwise_activation_coord_launches"))
    move = fmt_int(ncu.get("movement_materialization_launches"))
    resamp = fmt_int(ncu.get("sensor_resampling_launches"))
    recomb = fmt_int(ncu.get("basis_recombination_launches"))
    return rf"Dense {dense}; elem./coord. {elem}; move {move}; resamp. {resamp}; recomb. {recomb}"


def read_forward_json(path_str: str) -> Dict[str, Any]:
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def first_existing_forward_json(profile_id: str, nsys_row: Dict[str, str]) -> Dict[str, Any]:
    for key in ["forward_json", "json"]:
        obj = read_forward_json(nsys_row.get(key, ""))
        if obj:
            return obj

    candidates = [
        ROOT / "results" / "profiles" / "deeponet_nsys" / f"{profile_id}_forward.json",
        ROOT / "results" / "profiles" / "deeponet_ncu_basic" / f"{profile_id}_forward.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def table_preamble(caption: str, label: str, cols: str) -> List[str]:
    return [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
    ]


def write_main_table(plan_by_id, nsys_by_id, ncu_by_id):
    lines: List[str] = []
    lines += table_preamble(
        "DeepONet profiling diagnosis matrix for representative TorchScript cases. The table summarizes operation-class evidence from NSYS and NCU profiling rather than listing every kernel.",
        "tab:deeponet_profile_diagnosis_matrix",
        r"@{}p{0.22\linewidth}p{0.15\linewidth}p{0.24\linewidth}p{0.29\linewidth}@{}",
    )
    lines += [
        r"Profile case & Precision & Observed launch mix & Diagnosis \\",
        r"\midrule",
    ]

    for pid in MAIN_PROFILE_ORDER:
        plan = plan_by_id.get(pid, {})
        nsys = nsys_by_id.get(pid, {})
        ncu = ncu_by_id.get(pid, {})
        fwd = first_existing_forward_json(pid, nsys)

        case = CASE_NAMES.get(pid, tex_escape(pid))
        precision = precision_label(plan.get("precision") or fwd.get("precision") or "")
        mix = op_mix(ncu)
        diag = DIAGNOSIS_TEXT.get(pid, "")

        lines += [
            rf"{case} & {precision} & {mix} & {diag} \\",
            "",
        ]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    MAIN_TEX.parent.mkdir(parents=True, exist_ok=True)
    MAIN_TEX.write_text("\n".join(lines))
    print(f"wrote {MAIN_TEX}")


def write_appendix(plan_by_id, nsys_by_id, ncu_by_id):
    rows = []
    for pid in sorted(ncu_by_id):
        ncu = ncu_by_id[pid]
        plan = plan_by_id.get(pid, {})
        nsys = nsys_by_id.get(pid, {})
        fwd = first_existing_forward_json(pid, nsys)

        rows.append({
            "profile_id": pid,
            "case": CASE_NAMES.get(pid, pid.replace("_", r"\_")),
            "role": role_label(plan.get("case_role", "")),
            "precision": precision_label(plan.get("precision") or fwd.get("precision") or ""),
            "mean_ms": fmt_float(fwd.get("mean_ms") or fwd.get("mean_ms_under_profiler") or nsys.get("mean_ms"), 2),
            "peak_mb": fmt_float(fwd.get("peak_cuda_allocated_mb") or nsys.get("peak_cuda_allocated_mb"), 2),
            "launches": fmt_int(ncu.get("n_kernel_launch_rows_from_log") or ncu.get("n_kernel_launch_rows")),
            "unique": fmt_int(ncu.get("n_unique_kernel_names_from_log") or ncu.get("n_unique_kernel_names")),
            "dense": fmt_int(ncu.get("dense_gemm_gemv_launches")),
            "elem": fmt_int(ncu.get("elementwise_activation_coord_launches")),
            "move": fmt_int(ncu.get("movement_materialization_launches")),
            "resamp": fmt_int(ncu.get("sensor_resampling_launches")),
            "recomb": fmt_int(ncu.get("basis_recombination_launches")),
            "dominant": dominant_class(ncu),
        })

    lines: List[str] = []
    lines += [
        r"\section{DeepONet Profiling Details}",
        r"\label{app:deeponet_profiling_details}",
        "",
        "This appendix stores the DeepONet profiling details used to support the compact diagnosis matrix in the main text. "
        "The main manuscript reports operation-class evidence rather than a full kernel catalog, because the benchmark claim is about deployment mechanism rather than every individual CUDA launch.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{DeepONet NSYS/NCU profiling artifact summary. Latency values in profiling runs are used only for profiling sanity because profiler overhead perturbs runtime. Primary latency values are reported in the deployment tables.}",
        r"\label{tab:deeponet_profile_artifact_summary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}p{0.20\linewidth}p{0.10\linewidth}p{0.12\linewidth}rrrrrrp{0.14\linewidth}@{}}",
        r"\toprule",
        r"Profile case & Role & Precision & Launches & Unique & Dense & Elem. & Move & Recomb. & Dominant class \\",
        r"\midrule",
    ]

    for r in rows:
        lines.append(
            rf"{r['case']} & {r['role']} & {r['precision']} & "
            rf"{r['launches']} & {r['unique']} & {r['dense']} & {r['elem']} & {r['move']} & {r['recomb']} & "
            rf"{r['dominant']} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "The launch-class counts are derived from the generated profiling summaries and should be interpreted as a compact inventory of the observed runtime path. "
        "They are not used as primary latency measurements. The primary deployment numbers remain the warm-run latency, memory, precision, and sustained-energy measurements reported in the main tables.",
        "",
    ]

    APP_TEX.parent.mkdir(parents=True, exist_ok=True)
    APP_TEX.write_text("\n".join(lines))
    print(f"wrote {APP_TEX}")


def main():
    plan_rows = read_csv(PROFILE_PLAN)
    nsys_rows = read_csv(NSYS_FORWARD)
    ncu_rows = read_csv(NCU_BASIC)

    plan_by_id = by_key(plan_rows, "profile_id")
    nsys_by_id = by_key(nsys_rows, "profile_id")
    ncu_by_id = by_key(ncu_rows, "profile_id")

    missing = [pid for pid in MAIN_PROFILE_ORDER if pid not in ncu_by_id]
    if missing:
        raise RuntimeError(f"main profile rows missing from {NCU_BASIC}: {missing}")

    write_main_table(plan_by_id, nsys_by_id, ncu_by_id)
    write_appendix(plan_by_id, nsys_by_id, ncu_by_id)


if __name__ == "__main__":
    main()
