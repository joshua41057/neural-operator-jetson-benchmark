#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

IN_DIR = Path("results/profiles/deeponet_ncu_detailed_exports")
OUT = Path("results/artifacts/deeponet_ncu_class_aggregate.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

CASES = [
    ("burgers_base_r2048_ts_fp32", "Burgers base @2048 FP32"),
    ("darcy_base_r141_ts_fp32", "Darcy base @141x141 FP32"),
    ("darcy_base_r281_ts_fp32", "Darcy base @281x281 FP32"),
    ("darcy_base_r281_ts_fp16_native", "Darcy base @281x281 FP16 native"),
    ("darcy_large_r141_ts_fp32", "Darcy large @141x141 FP32"),
]

def to_float(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in {"", "nan", "NaN", "---", "N/A", "None"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def norm(s):
    return str(s or "").strip()

def short_kernel(name: str, max_len: int = 90) -> str:
    s = re.sub(r"\s+", " ", str(name))
    return s if len(s) <= max_len else s[:max_len - 3] + "..."

def classify_kernel(name: str) -> str:
    n = name.lower()

    # Kept for FNO/DeepONet cross-family consistency.
    if any(x in n for x in ["fft", "cufft", "bluestein", "r2c", "c2r"]):
        return "FFT/spectral"

    if any(x in n for x in [
        "sgemm", "gemm", "cublas", "cublaslt", "cutlass",
        "ampere_fp16", "s16816gemm", "matmul", "gemv"
    ]):
        return "Dense/GEMM"

    if any(x in n for x in [
        "gelu", "relu", "sigmoid", "tanh", "sin", "cos",
        "vectorized_elementwise", "elementwise", "unrolled_elementwise"
    ]):
        return "Activation/elementwise"

    if any(x in n for x in [
        "copy", "catarray", "cat", "fill", "zero", "memset",
        "index", "slice", "contiguous", "transpose", "permute"
    ]):
        return "Move/materialization"

    if any(x in n for x in [
        "cast", "convert", "typecast", "half", "float2half", "half2float"
    ]):
        return "Cast/precision"

    if any(x in n for x in [
        "upsample", "bilinear", "grid_sampler", "interpolate", "resample"
    ]):
        return "Resampling/interpolation"

    return "Other"

def metric_kind(metric_name: str):
    m = metric_name.lower()

    # NCU duration metrics.
    if "gpu__time_duration" in m or "duration" in m:
        return "duration"

    # Throughput metrics.
    if "sm__throughput" in m:
        return "sm_pct"

    if "gpu__compute_memory_throughput" in m:
        return "mem_pct"

    # L2 hit-rate metrics. Prefer sector hit rate if present.
    if "lts__t_sector_hit_rate" in m:
        return "l2_hit_pct"

    if "lts__t_request_hit_rate" in m:
        return "l2_hit_pct"

    # Fallback for label-name output.
    if "l2" in m and "hit" in m and "rate" in m:
        return "l2_hit_pct"

    return None

def convert_duration_to_ms(value, unit):
    if value is None:
        return None
    u = str(unit or "").lower().strip()

    if u in {"ms", "msecond", "mseconds", "millisecond", "milliseconds"}:
        return value
    if u in {"us", "usecond", "useconds", "microsecond", "microseconds"}:
        return value / 1000.0
    if u in {"ns", "nsecond", "nseconds", "nanosecond", "nanoseconds"}:
        return value / 1_000_000.0
    if u in {"s", "second", "seconds"}:
        return value * 1000.0

    # NCU CSV sometimes prints base units without a convenient label.
    # If unknown, keep the value as-is; the sanity check below will catch absurd totals.
    return value

def find_col(fieldnames, candidates):
    fmap = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        key = c.lower().strip()
        if key in fmap:
            return fmap[key]
    return None

def read_case_long_format(path: Path):
    """
    Reads NCU per-kernel CSV in long metric format:
      Kernel Name, Metric Name, Metric Unit, Average
    and pivots it into one record per kernel.
    """
    by_kernel = defaultdict(lambda: {
        "kernel": "",
        "ms": None,
        "sm_pct": None,
        "mem_pct": None,
        "l2_hit_pct": None,
    })

    with path.open(newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        kcol = find_col(fields, ["Kernel Name", "Kernel", "kernel_name"])
        mcol = find_col(fields, ["Metric Name", "metric_name"])
        ucol = find_col(fields, ["Metric Unit", "metric_unit", "Unit"])
        vcol = find_col(fields, ["Average", "Avg", "Value", "Metric Value"])

        if kcol is None or mcol is None or vcol is None:
            raise RuntimeError(
                f"Cannot parse long-format NCU CSV {path}. "
                f"fields={fields}"
            )

        for row in reader:
            kernel = norm(row.get(kcol))
            metric = norm(row.get(mcol))
            unit = norm(row.get(ucol)) if ucol else ""
            avg = to_float(row.get(vcol))

            if not kernel or not metric or avg is None:
                continue

            kind = metric_kind(metric)
            if kind is None:
                continue

            rec = by_kernel[kernel]
            rec["kernel"] = kernel

            if kind == "duration":
                ms = convert_duration_to_ms(avg, unit)
                # Keep the max duration metric if multiple duration-like rows appear.
                if ms is not None and (rec["ms"] is None or ms > rec["ms"]):
                    rec["ms"] = ms
            else:
                # Keep first useful value. If multiple related metrics exist,
                # this remains deterministic because CSV order is stable.
                if rec[kind] is None:
                    rec[kind] = avg

    rows = []
    for rec in by_kernel.values():
        if rec["ms"] is None or rec["ms"] <= 0:
            continue
        rec["class"] = classify_kernel(rec["kernel"])
        rows.append(rec)

    return rows

def read_case_wide_format(path: Path):
    """
    Fallback for already-normalized per-kernel CSVs.
    """
    rows = []
    with path.open(newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        kcol = find_col(fields, ["Kernel Name", "Kernel", "kernel_name", "kernel"])
        mscol = find_col(fields, ["ms", "duration_ms"])
        smcol = find_col(fields, ["sm_pct"])
        memcol = find_col(fields, ["mem_pct"])
        l2col = find_col(fields, ["l2_hit_pct"])

        if kcol is None or mscol is None:
            return []

        for row in reader:
            kernel = norm(row.get(kcol))
            ms = to_float(row.get(mscol))
            if not kernel or ms is None or ms <= 0:
                continue

            rows.append({
                "kernel": kernel,
                "class": classify_kernel(kernel),
                "ms": ms,
                "sm_pct": to_float(row.get(smcol)) if smcol else None,
                "mem_pct": to_float(row.get(memcol)) if memcol else None,
                "l2_hit_pct": to_float(row.get(l2col)) if l2col else None,
            })

    return rows

def read_case(path: Path):
    rows = read_case_wide_format(path)
    if rows:
        return rows
    return read_case_long_format(path)

def wavg(vals, weights):
    pairs = [
        (v, w) for v, w in zip(vals, weights)
        if v is not None and w is not None and w > 0
    ]
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)

out_rows = []

for case_id, case_label in CASES:
    p = IN_DIR / f"{case_id}_per_kernel.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    rows = read_case(p)
    if not rows:
        raise RuntimeError(f"No usable per-kernel rows in {p}")

    total_ms = sum(r["ms"] for r in rows)

    # Sanity warning only, not fatal. NCU replay can inflate totals.
    if total_ms <= 0:
        raise RuntimeError(f"Invalid total_ms={total_ms} for {case_id}")

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)

    for cls, rs in sorted(by_class.items()):
        cls_ms = sum(r["ms"] for r in rs)
        weights = [r["ms"] for r in rs]
        top = max(rs, key=lambda r: r["ms"])

        out_rows.append({
            "case_id": case_id,
            "case_label": case_label,
            "class": cls,
            "kernel_count": len(rs),
            "total_ms": cls_ms,
            "share_of_profiled_kernel_ms_pct": 100.0 * cls_ms / total_ms,
            "mean_sm_pct_weighted_by_ms": wavg([r["sm_pct"] for r in rs], weights),
            "mean_mem_pct_weighted_by_ms": wavg([r["mem_pct"] for r in rs], weights),
            "mean_l2_hit_pct_weighted_by_ms": wavg([r["l2_hit_pct"] for r in rs], weights),
            "max_kernel_ms": top["ms"],
            "representative_kernel": short_kernel(top["kernel"]),
        })

    out_rows.append({
        "case_id": case_id,
        "case_label": case_label,
        "class": "TOTAL",
        "kernel_count": len(rows),
        "total_ms": total_ms,
        "share_of_profiled_kernel_ms_pct": 100.0,
        "mean_sm_pct_weighted_by_ms": wavg([r["sm_pct"] for r in rows], [r["ms"] for r in rows]),
        "mean_mem_pct_weighted_by_ms": wavg([r["mem_pct"] for r in rows], [r["ms"] for r in rows]),
        "mean_l2_hit_pct_weighted_by_ms": wavg([r["l2_hit_pct"] for r in rows], [r["ms"] for r in rows]),
        "max_kernel_ms": max(r["ms"] for r in rows),
        "representative_kernel": "",
    })

fieldnames = [
    "case_id",
    "case_label",
    "class",
    "kernel_count",
    "total_ms",
    "share_of_profiled_kernel_ms_pct",
    "mean_sm_pct_weighted_by_ms",
    "mean_mem_pct_weighted_by_ms",
    "mean_l2_hit_pct_weighted_by_ms",
    "max_kernel_ms",
    "representative_kernel",
]

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in out_rows:
        rr = {}
        for k in fieldnames:
            v = r.get(k, "")
            if v is None:
                rr[k] = ""
            elif isinstance(v, float):
                rr[k] = f"{v:.6g}"
            else:
                rr[k] = v
        w.writerow(rr)

print(f"Wrote {OUT} with {len(out_rows)} rows")

# Compact console summary.
for case_id, case_label in CASES:
    print(f"\n=== {case_label} ===")
    for r in out_rows:
        if r["case_id"] == case_id and r["class"] != "TOTAL":
            print(
                f"{r['class']:28s} "
                f"share={r['share_of_profiled_kernel_ms_pct']:6.2f}% "
                f"kernels={r['kernel_count']:4d} "
                f"mem={r['mean_mem_pct_weighted_by_ms']} "
                f"l2={r['mean_l2_hit_pct_weighted_by_ms']} "
                f"rep={r['representative_kernel']}"
            )
