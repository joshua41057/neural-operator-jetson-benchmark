#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

IN_DIR = Path("results/profiles/deeponet_ncu_detailed_exports")
OUT = Path("results/artifacts/deeponet_ncu_paper_kernel_selection.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

CASES = [
    ("burgers_base_r2048_ts_fp32", "Burgers base @2048", 4),
    ("darcy_base_r141_ts_fp32", "Darcy base @141x141", 4),
    ("darcy_base_r281_ts_fp32", "Darcy base @281x281", 5),
    ("darcy_base_r281_ts_fp16_native", "Darcy base @281x281 FP16", 5),
    ("darcy_large_r141_ts_fp32", "Darcy large @141x141", 4),
]

def sget(row: dict, key: str) -> str:
    v = row.get(key, "")
    if v is None:
        return ""
    return str(v).strip()

def to_float(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in {"", "nan", "NaN", "---", "N/A", "n/a"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def classify_kernel(name: str) -> str:
    n = name.lower()

    if any(x in n for x in ["sgemm", "gemm", "cublas", "cutlass", "matmul"]):
        return "Dense/GEMM"

    if any(x in n for x in ["gemv", "bmm", "dot"]):
        return "Recombination"

    if any(x in n for x in [
        "gelu", "relu", "sigmoid", "tanh", "sin", "cos",
        "elementwise", "pointwise", "add", "mul", "div", "sub"
    ]):
        return "Activation/elementwise"

    if any(x in n for x in [
        "copy", "memcpy", "cat", "concat", "index", "gather",
        "scatter", "fill", "zero", "slice", "contiguous", "clone"
    ]):
        return "Move/materialization"

    if any(x in n for x in ["cast", "convert", "half", "fp16", "bf16"]):
        return "Cast/precision"

    return "Other"

def short_kernel(name: str, n: int = 90) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    name = name.replace("void ", "")
    if len(name) <= n:
        return name
    return name[: n - 3] + "..."

def metric_lookup(metrics: dict, candidates: list[str]):
    # Exact match first.
    for c in candidates:
        if c in metrics:
            return metrics[c]

    # Substring fallback because NCU names can vary by version/set.
    for key, value in metrics.items():
        lk = key.lower()
        for c in candidates:
            if c.lower() in lk:
                return value

    return None, ""

def duration_to_ms(value, unit: str):
    if value is None:
        return None

    u = (unit or "").strip().lower()

    if "nsecond" in u or u == "ns":
        return value / 1e6
    if "usecond" in u or "microsecond" in u or u in {"us", "µs"}:
        return value / 1e3
    if "msecond" in u or "millisecond" in u or u == "ms":
        return value
    if "second" in u or u == "s":
        return value * 1e3

    # NCU base-unit raw exports often store durations in ns.
    # If the number is very large, interpret as ns. Otherwise keep as ms.
    if value > 1000.0:
        return value / 1e6
    return value

def read_case(case: str):
    p = IN_DIR / f"{case}_per_kernel.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    by_kernel = defaultdict(dict)
    invocations = defaultdict(float)

    with p.open(newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue

            kernel = sget(row, "Kernel Name")
            metric = sget(row, "Metric Name")
            unit = sget(row, "Metric Unit")
            val = to_float(row.get("Average"))
            inv = to_float(row.get("Invocations"))

            # Skip malformed/header/empty rows.
            if not kernel or not metric or val is None:
                continue

            by_kernel[kernel][metric] = (val, unit)

            if inv is not None:
                invocations[kernel] = max(invocations[kernel], inv)

    rows = []

    for kernel, metrics in by_kernel.items():
        dur, dur_unit = metric_lookup(metrics, [
            "gpu__time_duration.sum",
            "gpu__time_duration.avg",
            "gpu__time_duration",
            "Duration",
        ])

        ms = duration_to_ms(dur, dur_unit)
        if ms is None:
            continue

        sm, _ = metric_lookup(metrics, [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_active",
            "sm__throughput",
            "Compute Throughput",
        ])

        mem, _ = metric_lookup(metrics, [
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput",
            "Memory Throughput",
        ])

        l2, _ = metric_lookup(metrics, [
            "lts__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate",
            "lts__t_request_hit_rate.pct",
            "lts__t_request_hit_rate",
            "L2 Hit Rate",
        ])

        occ, _ = metric_lookup(metrics, [
            "launch__occupancy_limit_active_warps_pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__occupancy",
            "Achieved Occupancy",
            "Occupancy",
        ])

        elig, _ = metric_lookup(metrics, [
            "smsp__warps_eligible.avg.per_cycle_active",
            "smsp__warps_eligible",
            "Eligible Warps Per Scheduler",
        ])

        rows.append({
            "case": case,
            "kernel": kernel,
            "kernel_short": short_kernel(kernel),
            "class": classify_kernel(kernel),
            "ms": ms,
            "sm_pct": sm,
            "mem_pct": mem,
            "l2_hit_pct": l2,
            "occ_pct": occ,
            "eligible": elig,
            "invocations": invocations[kernel],
        })

    rows.sort(key=lambda r: r["ms"], reverse=True)
    return rows

def memory_stress_score(r):
    mem = r["mem_pct"] if r["mem_pct"] is not None else 0.0
    l2 = r["l2_hit_pct"] if r["l2_hit_pct"] is not None else 100.0
    return mem + max(0.0, 100.0 - l2)

def select_rows(rows, max_rows: int):
    selected = []
    used = set()

    if not rows:
        return selected

    # 1. Top-duration kernel.
    top = rows[0]
    selected.append(top)
    used.add(top["kernel"])

    # 2. Class-diverse representatives.
    class_order = [
        "Dense/GEMM",
        "Recombination",
        "Activation/elementwise",
        "Move/materialization",
        "Cast/precision",
        "Other",
    ]

    for cls in class_order:
        if len(selected) >= max_rows:
            break

        candidates = [
            r for r in rows
            if r["class"] == cls and r["kernel"] not in used
        ]
        if not candidates:
            continue

        best = max(candidates, key=lambda r: r["ms"])
        selected.append(best)
        used.add(best["kernel"])

    # 3. Memory hierarchy stress row.
    if len(selected) < max_rows:
        candidates = [r for r in rows if r["kernel"] not in used]
        if candidates:
            best = max(candidates, key=memory_stress_score)
            selected.append(best)
            used.add(best["kernel"])

    # 4. Fill remaining slots with top-duration rows.
    for r in rows:
        if len(selected) >= max_rows:
            break
        if r["kernel"] not in used:
            selected.append(r)
            used.add(r["kernel"])

    return selected

def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)

paper_rows = []

for case, label, max_rows in CASES:
    rows = read_case(case)
    if not rows:
        raise RuntimeError(f"No usable kernel rows parsed for case: {case}")

    selected = select_rows(rows, max_rows)

    for rank, r in enumerate(selected, 1):
        reason = []

        if rank == 1:
            reason.append("top-duration")

        if r["class"] in {
            "Dense/GEMM",
            "Recombination",
            "Activation/elementwise",
            "Move/materialization",
            "Cast/precision",
        }:
            reason.append("class-representative")

        if r["mem_pct"] is not None and r["mem_pct"] >= 60:
            reason.append("high-memory-throughput")

        if r["l2_hit_pct"] is not None and r["l2_hit_pct"] <= 50:
            reason.append("low-L2-hit")

        if not reason:
            reason.append("representative")

        paper_rows.append({
            "case": case,
            "case_label": label,
            "selection_rank": rank,
            "kernel_short": r["kernel_short"],
            "kernel_full": r["kernel"],
            "class": r["class"],
            "ms": r["ms"],
            "sm_pct": r["sm_pct"],
            "mem_pct": r["mem_pct"],
            "l2_hit_pct": r["l2_hit_pct"],
            "occ_pct": r["occ_pct"],
            "eligible": r["eligible"],
            "invocations": r["invocations"],
            "selection_reason": "+".join(reason),
        })

with OUT.open("w", newline="") as f:
    fieldnames = [
        "case",
        "case_label",
        "selection_rank",
        "kernel_short",
        "kernel_full",
        "class",
        "ms",
        "sm_pct",
        "mem_pct",
        "l2_hit_pct",
        "occ_pct",
        "eligible",
        "invocations",
        "selection_reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in paper_rows:
        writer.writerow({k: fmt(v) for k, v in r.items()})

print(f"Wrote {OUT}")
print()

for r in paper_rows:
    print(
        f"{r['case_label']:32s} | "
        f"{r['selection_rank']} | "
        f"{r['class']:24s} | "
        f"{r['ms']:.4f} ms | "
        f"SM={fmt(r['sm_pct'])} "
        f"Mem={fmt(r['mem_pct'])} "
        f"L2={fmt(r['l2_hit_pct'])} "
        f"Occ={fmt(r['occ_pct'])} | "
        f"{r['selection_reason']} | "
        f"{r['kernel_short']}"
    )

print()
print("Paper kernel selection complete.")
