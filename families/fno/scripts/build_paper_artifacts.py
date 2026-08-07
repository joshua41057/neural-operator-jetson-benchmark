#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

RESULTS_ROOT = Path("results")
OUTDIR = RESULTS_ROOT / "artifacts"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def flatten_resolution(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        if len(x) == 2:
            return f"{x[0]}x{x[1]}"
        return "x".join(str(v) for v in x)
    return str(x)


def resolution_to_int(res: Optional[str]) -> Optional[int]:
    """
    For 2D like 421x421 -> 421.
    For 1D like 2048 -> 2048.
    """
    if res is None:
        return None
    m2 = re.match(r"^\s*(\d+)x(\d+)\s*$", res)
    if m2:
        return int(m2.group(1))
    m1 = re.match(r"^\s*(\d+)\s*$", res)
    if m1:
        return int(m1.group(1))
    return None


def infer_model_family(name: str) -> str:
    lname = name.lower()
    if "fno" in lname:
        return "FNO"
    if "deeponet" in lname:
        return "DeepONet"
    if "vs_wno" in lname or "vs-wno" in lname:
        return "VS-WNO"
    if "wno" in lname:
        return "WNO"
    return "unknown"


def infer_backend(name: str, payload: Dict[str, Any]) -> Optional[str]:
    mode = payload.get("mode")
    if mode:
        if mode.lower() == "torchscript":
            return "torchscript"
        if mode.lower() == "eager":
            return "eager"
    lname = name.lower()
    if "torchscript" in lname or "_ts_" in lname or name.endswith("_ts_fp32") or name.endswith("_ts_fp16"):
        return "torchscript"
    if "eager" in lname:
        return "eager"
    return None


def infer_variant_type(group: str, name: str) -> str:
    lname = name.lower()

    if group == "jetson_fno_frontier":
        return "frontier"
    if group == "jetson_fno_oom_frontier":
        return "oom_frontier"
    if group == "jetson_fno_sustained":
        return "sustained"
    if group == "jetson_fno_samples":
        return "sample_variance"
    if "modes" in lname or "nocoords" in lname or "pad" in lname:
        return "ablation"
    if re.search(r"_r\d+_", lname):
        return "resolution"
    if "small" in lname or "base" in lname or "large" in lname:
        return "scale"
    return "other"


def soft_infeasible(mean_ms: Optional[float], threshold_ms: float = 1000.0) -> bool:
    return mean_ms is not None and mean_ms >= threshold_ms


# ============================================================
# Index scanning
# ============================================================

@dataclass
class RunIndex:
    run_id: str
    source_group: str
    experiment_name: str
    json_path: Optional[str]
    tegrastats_path: Optional[str]
    stderr_path: Optional[str]
    stdout_path: Optional[str]
    nsys_summary_path: Optional[str]
    ncu_per_kernel_path: Optional[str]
    ncu_sched_per_kernel_path: Optional[str]


def scan_results_tree(root: Path) -> pd.DataFrame:
    rows: List[RunIndex] = []

    interesting_groups = [
        "jetson_fno",
        "jetson_fno_frontier",
        "jetson_fno_oom_frontier",
        "jetson_fno_sustained",
        "jetson_fno_samples",
        "jetson_fno_profile_nsys",
        "jetson_fno_profile_ncu",
        "jetson_fno_profile_ncu_frontier",
        "jetson_fno_profile_ncu_sched",
        "jetson_fno_profile_module",
    ]

    for group in interesting_groups:
        gdir = root / group
        if not gdir.exists():
            continue

        stems = set()
        for p in gdir.iterdir():
            if p.is_file():
                stem = p.name
                stem = re.sub(r"_tegrastats\.log$", "", stem)
                stem = re.sub(r"_nsys_stats\.txt$", "", stem)
                stem = re.sub(r"_ncu_sched_per_kernel\.txt$", "", stem)
                stem = re.sub(r"_ncu_per_kernel\.txt$", "", stem)
                stem = re.sub(r"\.json$", "", stem)
                stem = re.sub(r"\.stderr$", "", stem)
                stem = re.sub(r"\.stdout$", "", stem)
                stems.add(stem)

        for stem in sorted(stems):
            row = RunIndex(
                run_id=f"{group}:{stem}",
                source_group=group,
                experiment_name=stem,
                json_path=str(gdir / f"{stem}.json") if (gdir / f"{stem}.json").exists() else None,
                tegrastats_path=str(gdir / f"{stem}_tegrastats.log") if (gdir / f"{stem}_tegrastats.log").exists() else None,
                stderr_path=str(gdir / f"{stem}.stderr") if (gdir / f"{stem}.stderr").exists() else None,
                stdout_path=str(gdir / f"{stem}.stdout") if (gdir / f"{stem}.stdout").exists() else None,
                nsys_summary_path=str(gdir / f"{stem}_nsys_stats.txt") if (gdir / f"{stem}_nsys_stats.txt").exists() else None,
                ncu_per_kernel_path=str(gdir / f"{stem}_per_kernel.txt") if (gdir / f"{stem}_per_kernel.txt").exists() else None,
                ncu_sched_per_kernel_path=str(gdir / f"{stem}_sched_per_kernel.txt") if (gdir / f"{stem}_sched_per_kernel.txt").exists() else None,
            )
            rows.append(row)

    df = pd.DataFrame(asdict(r) for r in rows)
    df.to_csv(OUTDIR / "index_runs.csv", index=False)
    return df


# ============================================================
# JSON run parsing
# ============================================================

def parse_json_runs(index_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, r in index_df.iterrows():
        if not r["json_path"]:
            continue
        p = Path(r["json_path"])
        try:
            payload = safe_load_json(p)
        except Exception as e:
            rows.append({
                "run_id": r["run_id"],
                "json_path": str(p),
                "json_parse_error": str(e),
            })
            continue

        exp = r["experiment_name"]
        source_group = r["source_group"]

        dataset = payload.get("dataset")
        model_family = infer_model_family(exp)

        train_res = flatten_resolution(payload.get("resolution"))
        input_res = None
        bank_meta = payload.get("bank_meta", {}) if isinstance(payload.get("bank_meta"), dict) else {}
        if bank_meta.get("resolution") is not None:
            input_res = flatten_resolution(bank_meta.get("resolution"))
        elif payload.get("resolution") is not None:
            input_res = flatten_resolution(payload.get("resolution"))

        row = {
            "run_id": r["run_id"],
            "source_group": source_group,
            "experiment_name": exp,
            "dataset": dataset,
            "model_family": model_family,
            "variant_type": infer_variant_type(source_group, exp),
            "backend": infer_backend(exp, payload),
            "precision": payload.get("precision"),
            "batch_size": payload.get("batch_size"),
            "mode": payload.get("mode"),
            "params": payload.get("params"),
            "train_resolution": train_res,
            "input_resolution": input_res,
            "mean_ms": payload.get("mean_ms"),
            "median_ms": payload.get("median_ms"),
            "p95_ms": payload.get("p95_ms"),
            "p99_ms": payload.get("p99_ms"),
            "min_ms": payload.get("min_ms"),
            "max_ms": payload.get("max_ms"),
            "cuda_peak_allocated_mb": payload.get("cuda_peak_allocated_mb"),
            "input_bank": payload.get("input_bank"),
            "source": payload.get("source"),
            "sample_index": payload.get("sample_index"),
            "notes": payload.get("notes"),
            "bank_meta_dataset": bank_meta.get("dataset"),
            "bank_meta_resolution": flatten_resolution(bank_meta.get("resolution")),
            "bank_meta_num_samples": bank_meta.get("num_samples"),
            "synthetic_frontier": bank_meta.get("synthetic_frontier", False),
            "synthetic_from": bank_meta.get("synthetic_from"),
            "soft_infeasible_200ms": payload.get("mean_ms", 0) >= 200 if payload.get("mean_ms") is not None else False,
            "soft_infeasible_500ms": payload.get("mean_ms", 0) >= 500 if payload.get("mean_ms") is not None else False,
            "soft_infeasible_1000ms": payload.get("mean_ms", 0) >= 1000 if payload.get("mean_ms") is not None else False,
            "json_path": str(p),
            "tegrastats_path": r["tegrastats_path"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# tegrastats parsing
# ============================================================

RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB", re.IGNORECASE)
SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB", re.IGNORECASE)

# Try multiple common power patterns across Jetson tegrastats variants.
POWER_RE_LIST = [
    re.compile(r"VDD_IN\s+(\d+)mW/(\d+)mW", re.IGNORECASE),
    re.compile(r"POM_5V_IN\s+(\d+)mW/(\d+)mW", re.IGNORECASE),
]

GPU_UTIL_RE = re.compile(r"GR3D_FREQ\s+(\d+)%", re.IGNORECASE)
TEMP_GPU_RE = re.compile(r"GPU@([0-9.]+)C", re.IGNORECASE)


def parse_tegrastats_log(path: Path) -> Dict[str, Any]:
    text = safe_read_text(path)
    ram_used = []
    swap_used = []
    board_pwr = []
    gpu_util = []
    gpu_temp = []

    for line in text.splitlines():
        m = RAM_RE.search(line)
        if m:
            ram_used.append(float(m.group(1)))

        m = SWAP_RE.search(line)
        if m:
            swap_used.append(float(m.group(1)))

        for pre in POWER_RE_LIST:
            m = pre.search(line)
            if m:
                board_pwr.append(float(m.group(1)) / 1000.0)
                break

        m = GPU_UTIL_RE.search(line)
        if m:
            gpu_util.append(float(m.group(1)))

        m = TEMP_GPU_RE.search(line)
        if m:
            gpu_temp.append(float(m.group(1)))

    def stat(xs: List[float], fn: str) -> Optional[float]:
        if not xs:
            return None
        if fn == "mean":
            return sum(xs) / len(xs)
        if fn == "peak":
            return max(xs)
        return None

    return {
        "sample_count": max(len(ram_used), len(board_pwr), len(gpu_util), len(gpu_temp)),
        "ram_mb_mean": stat(ram_used, "mean"),
        "ram_mb_peak": stat(ram_used, "peak"),
        "swap_mb_peak": stat(swap_used, "peak"),
        "board_power_w_mean": stat(board_pwr, "mean"),
        "board_power_w_peak": stat(board_pwr, "peak"),
        "gpu_util_pct_mean": stat(gpu_util, "mean"),
        "gpu_util_pct_peak": stat(gpu_util, "peak"),
        "temp_gpu_c_mean": stat(gpu_temp, "mean"),
        "temp_gpu_c_peak": stat(gpu_temp, "peak"),
    }


def build_tegrastats_summary(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in index_df.iterrows():
        if not r["tegrastats_path"]:
            continue
        p = Path(r["tegrastats_path"])
        try:
            parsed = parse_tegrastats_log(p)
        except Exception as e:
            parsed = {"tegrastats_parse_error": str(e)}
        parsed["run_id"] = r["run_id"]
        parsed["tegrastats_path"] = str(p)
        rows.append(parsed)

    return pd.DataFrame(rows)


# ============================================================
# OOM frontier parsing
# ============================================================

OOM_RE = re.compile(
    r"out of memory|cuda out of memory|allocation failed|CUDNN_STATUS_ALLOC_FAILED|INTERNAL ASSERT FAILED",
    re.IGNORECASE,
)

def parse_oom_frontier(root: Path) -> pd.DataFrame:
    f = root / "jetson_fno_oom_frontier" / "frontier_status.csv"
    if not f.exists():
        return pd.DataFrame()

    df = pd.read_csv(f)
    df["oom_flag"] = df["status"].astype(str).str.contains("oom", case=False, na=False)
    df["alloc_fail_flag"] = df["status"].astype(str).str.contains("alloc", case=False, na=False)
    df["success_flag"] = df["status"].astype(str).eq("success")

    res_vals = []
    for bank in df["input_bank"].astype(str):
        m = re.search(r"_r(\d+)_bank", bank)
        res_vals.append(int(m.group(1)) if m else None)
    df["effective_input_resolution"] = res_vals
    return df


# ============================================================
# NSYS parsing
# ============================================================

def parse_nsys_summary(root: Path) -> pd.DataFrame:
    csv_path = root / "jetson_fno_profile_nsys" / "nsys_profile_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["profile_case"] = df["tag"]
        return df
    return pd.DataFrame()


# ============================================================
# NCU parsing
# ============================================================

SECTION_HEADER_RE = re.compile(r"^\s*Section:\s*(.+?)\s*$")
KERNEL_HEADER_RE = re.compile(
    r"^\s{2}(.+?),\s+Device\s+\d+,\s+CC\s+[\d.]+(?:,\s+Invocations\s+(\d+))?\s*$"
)

TABLE_ROW_RE = re.compile(r"^\s{4,}(.+?)\s{2,}(.+?)\s{2,}(.+?)\s*$")


def canonical_kernel_family(kernel_name: str) -> str:
    name = kernel_name

    patterns = [
        (r"regular_bluestein_fft", "regular_bluestein_fft"),
        (r"ampere_sgemm_128x64_nn", "ampere_sgemm_128x64_nn"),
        (r"ampere_sgemm_32x128_tn", "ampere_sgemm_32x128_tn"),
        (r"fused_add_gelu", "fused_add_gelu"),
        (r"fused_add_mul_add", "fused_add_mul_add"),
        (r"fused_unsqueeze_sub_add_div", "fused_unsqueeze_sub_add_div"),
        (r"preprocess_kernel", "preprocess_kernel"),
        (r"postprocess_kernel", "postprocess_kernel"),
        (r"packR2C_kernel", "packR2C_kernel"),
        (r"unpackC2R_kernel", "unpackC2R_kernel"),
        (r"direct_copy_kernel_cuda", "direct_copy_kernel_cuda"),
        (r"CatArrayBatchedCopy", "CatArrayBatchedCopy"),
        (r"gemv2N_kernel", "gemv2N_kernel"),
        (r"internal::kernel<", "cublas_internal_kernel"),
        (r"vectorized_elementwise_kernel", "vectorized_elementwise_kernel"),
        (r"elementwise_kernel<", "elementwise_kernel"),
        (r"FillFunctor<c10::complex<float>>", "fill_complex"),
        (r"FillFunctor<float>", "fill_float"),
        (r"regular_fft_factor", "regular_fft_factor"),
        (r"prime_fft_factor", "prime_fft_factor"),
        (r"prime_fft<", "prime_fft"),
        (r"vector_fft<", "vector_fft"),
        (r"cutlass::Kernel2", "cutlass_kernel"),
        (r"generate_chirp_signal", "generate_chirp_signal"),
        (r"scale_conjugate", "scale_conjugate"),
        (r"convert<", "convert"),
    ]

    for pat, fam in patterns:
        if re.search(pat, name):
            return fam
    return name[:120]


def parse_metric_value(s: str) -> Optional[float]:
    s = s.strip().replace(",", "")
    if s in {"", "N/A", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_ncu_per_kernel_txt(path: Path, profile_case: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - long metrics table
      - kernel summary table
    """
    text = safe_read_text(path)
    lines = text.splitlines()

    long_rows = []
    current_kernel = None
    current_invocations = None
    current_section = None

    i = 0
    while i < len(lines):
        line = lines[i]

        km = KERNEL_HEADER_RE.match(line)
        if km:
            current_kernel = km.group(1).strip()
            current_invocations = int(km.group(2)) if km.group(2) else None
            current_section = None
            i += 1
            continue

        sm = SECTION_HEADER_RE.match(line)
        if sm:
            current_section = sm.group(1).strip()
            i += 1

            # skip separator/header lines
            while i < len(lines):
                l2 = lines[i]
                if re.match(r"^\s*-+\s+-+\s+", l2) or "Metric Name" in l2:
                    i += 1
                    continue
                if not l2.strip():
                    break
                if SECTION_HEADER_RE.match(l2) or KERNEL_HEADER_RE.match(l2):
                    break

                # table rows can be:
                # name unit min max avg
                # or name unit value
                parts = re.split(r"\s{2,}", l2.strip())
                if len(parts) >= 3 and current_kernel and current_section:
                    metric_name = parts[0].strip()
                    metric_unit = parts[1].strip()

                    metric_min = metric_max = metric_avg = None
                    if len(parts) == 3:
                        metric_avg = parse_metric_value(parts[2])
                    elif len(parts) >= 5:
                        metric_min = parse_metric_value(parts[2])
                        metric_max = parse_metric_value(parts[3])
                        metric_avg = parse_metric_value(parts[4])

                    long_rows.append({
                        "profile_case": profile_case,
                        "kernel_name_raw": current_kernel,
                        "kernel_family": canonical_kernel_family(current_kernel),
                        "invocations": current_invocations,
                        "section": current_section,
                        "metric_name": metric_name,
                        "metric_unit": metric_unit,
                        "metric_min": metric_min,
                        "metric_max": metric_max,
                        "metric_avg": metric_avg,
                    })
                i += 1
            continue

        i += 1

    long_df = pd.DataFrame(long_rows)

    if long_df.empty:
        return long_df, pd.DataFrame()

    def fetch_metric(g: pd.DataFrame, section: str, name: str) -> Optional[float]:
        sub = g[(g["section"] == section) & (g["metric_name"] == name)]
        if sub.empty:
            return None
        return sub["metric_avg"].iloc[0]

    summary_rows = []
    for (profile_case_, kernel_name_raw, kernel_family, invocations), g in long_df.groupby(
        ["profile_case", "kernel_name_raw", "kernel_family", "invocations"], dropna=False
    ):
        row = {
            "profile_case": profile_case_,
            "kernel_name_raw": kernel_name_raw,
            "kernel_family": kernel_family,
            "invocations": invocations,
            "avg_duration_ms": fetch_metric(g, "GPU Speed Of Light Throughput", "Duration"),
            "avg_compute_pct": fetch_metric(g, "GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
            "avg_memory_pct": fetch_metric(g, "GPU Speed Of Light Throughput", "Memory Throughput"),
            "avg_l1tex_hit_pct": fetch_metric(g, "Memory Workload Analysis", "L1/TEX Hit Rate"),
            "avg_l2_hit_pct": fetch_metric(g, "Memory Workload Analysis", "L2 Hit Rate"),
            "avg_achieved_occupancy_pct": fetch_metric(g, "Occupancy", "Achieved Occupancy"),
            "avg_eligible_warps_per_scheduler": fetch_metric(g, "Scheduler Statistics", "Eligible Warps Per Scheduler"),
            "avg_no_eligible_pct": fetch_metric(g, "Scheduler Statistics", "No Eligible"),
            "avg_warp_cycles_per_issued_inst": fetch_metric(g, "Warp State Statistics", "Warp Cycles Per Issued Instruction"),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return long_df, summary_df


def build_ncu_tables(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ncu_groups = [
        "jetson_fno_profile_ncu",
        "jetson_fno_profile_ncu_frontier",
        "jetson_fno_profile_ncu_sched",
    ]

    long_parts = []
    sum_parts = []

    for group in ncu_groups:
        gdir = root / group
        if not gdir.exists():
            continue

        for txt in sorted(gdir.glob("*_per_kernel.txt")):
            profile_case = txt.stem.replace("_per_kernel", "")
            long_df, summary_df = parse_ncu_per_kernel_txt(txt, profile_case)
            if not long_df.empty:
                long_df["source_group"] = group
                long_parts.append(long_df)
            if not summary_df.empty:
                summary_df["source_group"] = group
                sum_parts.append(summary_df)

    long_all = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    sum_all = pd.concat(sum_parts, ignore_index=True) if sum_parts else pd.DataFrame()

    if not sum_all.empty:
        sum_all["category"] = sum_all["kernel_family"].map(classify_kernel_category)
        sum_all["paper_keep_flag"] = sum_all.apply(mark_paper_keep, axis=1)

    return long_all, sum_all


def classify_kernel_category(kernel_family: str) -> str:
    fam = kernel_family.lower()
    if "bluestein" in fam or "fft" in fam or "packr2c" in fam or "unpackc2r" in fam or "preprocess" in fam or "postprocess" in fam:
        return "fft_transform"
    if "sgemm" in fam or "cutlass" in fam or "gemv" in fam or "cublas" in fam:
        return "dense_math"
    if "copy" in fam or "catarray" in fam or "fill_" in fam:
        return "movement_copy"
    if "gelu" in fam or "elementwise" in fam or "fused_" in fam:
        return "elementwise_fused"
    return "other"


def mark_paper_keep(row: pd.Series) -> bool:
    fam = str(row.get("kernel_family", "")).lower()
    dur = row.get("avg_duration_ms")
    if dur is None or pd.isna(dur):
        return False
    if dur >= 1.0:
        return True
    if any(k in fam for k in ["bluestein", "sgemm", "cutlass", "gemv", "postprocess", "preprocess", "packr2c", "unpackc2r", "direct_copy", "fused_add_gelu"]):
        return True
    return False


# ============================================================
# Merge + derived tables
# ============================================================

def build_deploy_runs_table(json_df: pd.DataFrame, tegra_df: pd.DataFrame) -> pd.DataFrame:
    if json_df.empty:
        return json_df
    df = json_df.merge(tegra_df, on="run_id", how="left", suffixes=("", "_tegrastats"))
    df["success"] = True
    df["oom_flag"] = False
    return df


def build_paper_derived_tables(
    deploy_df: pd.DataFrame,
    oom_df: pd.DataFrame,
    nsys_df: pd.DataFrame,
    ncu_sum_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    derived: Dict[str, pd.DataFrame] = {}

    # 1) deployability summary
    if not deploy_df.empty:
        paper_deploy = deploy_df.copy()
        paper_deploy["effective_resolution_int"] = paper_deploy["input_resolution"].map(resolution_to_int)
        derived["paper_deployability_summary"] = paper_deploy

    # 2) frontier summary
    if not deploy_df.empty or not oom_df.empty:
        frontier_parts = []
        if not deploy_df.empty:
            frontier_parts.append(
                deploy_df[deploy_df["variant_type"].isin(["frontier", "oom_frontier"])].copy()
            )
        if not oom_df.empty:
            frontier_parts.append(oom_df.copy())
        derived["paper_frontier_summary"] = pd.concat(frontier_parts, ignore_index=True) if frontier_parts else pd.DataFrame()

    # 3) representative kernel attribution
    if not ncu_sum_df.empty:
        keep = ncu_sum_df[ncu_sum_df["paper_keep_flag"]].copy()
        keep = keep.sort_values(["profile_case", "avg_duration_ms"], ascending=[True, False])
        derived["paper_kernel_attribution"] = keep

    # 4) r281 vs 421 comparison
    if not ncu_sum_df.empty:
        subset = ncu_sum_df[
            ncu_sum_df["profile_case"].isin([
                "darcy_r281_ts_fp32_ncu_sched",
                "darcy_large_on421_ts_fp32_ncu_sched",
                "darcy_r281_ts_fp32_ncu",
                "darcy_large_on421_ts_fp32_ncu",
            ])
        ].copy()
        derived["paper_r281_vs_421_compare"] = subset

    return derived


# ============================================================
# Main
# ============================================================

def main() -> None:
    index_df = scan_results_tree(RESULTS_ROOT)
    index_df.to_csv(OUTDIR / "index_runs.csv", index=False)

    json_df = parse_json_runs(index_df)
    json_df.to_csv(OUTDIR / "json_runs_raw.csv", index=False)

    tegra_df = build_tegrastats_summary(index_df)
    tegra_df.to_csv(OUTDIR / "tegrastats_summary.csv", index=False)

    deploy_df = build_deploy_runs_table(json_df, tegra_df)
    deploy_df.to_csv(OUTDIR / "deploy_runs.csv", index=False)

    oom_df = parse_oom_frontier(RESULTS_ROOT)
    oom_df.to_csv(OUTDIR / "oom_status.csv", index=False)

    nsys_df = parse_nsys_summary(RESULTS_ROOT)
    nsys_df.to_csv(OUTDIR / "nsys_summary.csv", index=False)

    ncu_long_df, ncu_sum_df = build_ncu_tables(RESULTS_ROOT)
    ncu_long_df.to_csv(OUTDIR / "ncu_kernel_metrics_long.csv", index=False)
    ncu_sum_df.to_csv(OUTDIR / "ncu_kernel_summary.csv", index=False)

    derived = build_paper_derived_tables(deploy_df, oom_df, nsys_df, ncu_sum_df)
    for name, df in derived.items():
        df.to_csv(OUTDIR / f"{name}.csv", index=False)

    print("Wrote normalized artifacts to:", OUTDIR)


if __name__ == "__main__":
    main()