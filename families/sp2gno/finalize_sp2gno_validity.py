#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path


PRECISIONS = ["fp32_strict", "tf32", "bf16_autocast", "fp16_autocast", "fp16_native"]

METRICS = [
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_latency_ms",
    "throughput_inf_s",
    "energy_j_per_inference",
    "vdd_in_mean_w",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "board_ram_mean_mb",
    "board_ram_peak_mb",
    "peak_temp_c",
    "test_rel_l2",
    "perturb_rel_l2_vs_fp32",
]


def infer_case_id(run_name: str) -> str:
    return re.sub(
        r"_(fp32_strict|tf32|bf16_autocast|fp16_autocast|fp16_native)_rep[0-9]+$",
        "",
        run_name,
    )


def as_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def is_finite(x) -> bool:
    return as_float(x) is not None


def mean(xs):
    vals = [as_float(x) for x in xs]
    vals = [v for v in vals if v is not None]
    return stats.mean(vals) if vals else ""


def pstdev(xs):
    vals = [as_float(x) for x in xs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return 0.0
    return stats.pstdev(vals)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", required=True)
    args = ap.parse_args()

    root = Path(args.suite_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing suite root: {root}")

    files = sorted(root.glob("**/reports/sp2gno_edge_summary_*.csv"))

    rows = []
    for path in files:
        with open(path, newline="") as f:
            r = next(csv.DictReader(f))

        r["summary_file"] = str(path)
        r["case_id"] = r.get("case_id") or infer_case_id(r.get("run_name", ""))

        precision = r.get("precision", "")

        if r.get("status") != "success":
            r["validity_status"] = "failed_runtime"
            r["paper_status"] = "Runtime Fail"
            r["validity_error_message"] = r.get("error", "")
        else:
            rel_ok = is_finite(r.get("test_rel_l2"))

            if precision == "fp32_strict":
                pert_ok = True
                if not is_finite(r.get("perturb_rel_l2_vs_fp32")):
                    r["perturb_rel_l2_vs_fp32"] = "0.0"
            else:
                pert_ok = is_finite(r.get("perturb_rel_l2_vs_fp32"))

            if rel_ok and pert_ok:
                r["validity_status"] = "valid"
                r["paper_status"] = "Success"
                r["validity_error_message"] = ""
            else:
                r["validity_status"] = "failed_numerical"
                r["paper_status"] = "Numerical Fail"
                r["validity_error_message"] = (
                    f"Non-finite metric: test_rel_l2={r.get('test_rel_l2')}, "
                    f"perturb={r.get('perturb_rel_l2_vs_fp32')}"
                )

        rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["case_id"], r.get("dataset", ""), r.get("precision", ""))].append(r)

    summary = []
    for (case_id, dataset, precision), rs in sorted(groups.items()):
        valid_rs = [r for r in rs if r["validity_status"] == "valid"]
        numfail_rs = [r for r in rs if r["validity_status"] == "failed_numerical"]
        runfail_rs = [r for r in rs if r["validity_status"] == "failed_runtime"]

        first = rs[0]
        if valid_rs:
            paper_status = "Success"
        elif numfail_rs:
            paper_status = "Numerical Fail"
        else:
            paper_status = "Runtime Fail"

        out = {
            "case_id": case_id,
            "dataset": dataset,
            "precision": precision,
            "paper_status": paper_status,
            "n_total": len(rs),
            "n_valid": len(valid_rs),
            "n_failed_numerical": len(numfail_rs),
            "n_failed_runtime": len(runfail_rs),
            "resolution": first.get("resolution", ""),
            "num_nodes": first.get("num_nodes", ""),
            "parameter_count": first.get("parameter_count", ""),
            "width": first.get("width", ""),
            "n_layers": first.get("n_layers", ""),
            "num_freq": first.get("num_freq", ""),
            "k": first.get("k", ""),
            "checkpoint": first.get("checkpoint", ""),
            "runtime_path": first.get("runtime_path", ""),
            "timing_boundary": first.get("timing_boundary", ""),
            "runtime_failure_message": runfail_rs[0].get("error", "") if runfail_rs else "",
            "numerical_failure_message": numfail_rs[0].get("validity_error_message", "") if numfail_rs else "",
        }

        for m in METRICS:
            out[f"{m}_mean_valid"] = mean([r.get(m) for r in valid_rs])
            out[f"{m}_std_valid"] = pstdev([r.get(m) for r in valid_rs])

        summary.append(out)

    fp32_by_case = {
        r["case_id"]: r
        for r in summary
        if r["precision"] == "fp32_strict" and r["paper_status"] == "Success"
    }

    for r in summary:
        base = fp32_by_case.get(r["case_id"])
        cur = as_float(r.get("p50_latency_ms_mean_valid"))
        ref = as_float(base.get("p50_latency_ms_mean_valid")) if base else None

        if r["paper_status"] == "Success" and ref and cur:
            r["speedup_vs_fp32_p50_valid"] = ref / cur
        else:
            r["speedup_vs_fp32_p50_valid"] = ""

    invalid = [r for r in rows if r["validity_status"] != "valid"]

    raw_path = root / "sp2gno_final_validity_raw.csv"
    summary_path = root / "sp2gno_final_validity_summary.csv"
    invalid_path = root / "sp2gno_final_invalid_results.csv"

    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    write_csv(invalid_path, invalid)

    c = Counter(r["validity_status"] for r in rows)

    print("suite_root:", root)
    print("total:", len(rows))
    print("valid:", c["valid"])
    print("failed_numerical:", c["failed_numerical"])
    print("failed_runtime:", c["failed_runtime"])
    print("summary:", summary_path)
    print("invalid:", invalid_path)

    print("\nPaper-status summary:")
    for r in summary:
        print(
            f"{r['case_id']:32s} {r['precision']:15s} "
            f"{r['paper_status']:15s} "
            f"valid={r['n_valid']} numfail={r['n_failed_numerical']} runfail={r['n_failed_runtime']} "
            f"p50={r.get('p50_latency_ms_mean_valid')} "
            f"speedup={r.get('speedup_vs_fp32_p50_valid')} "
            f"pert={r.get('perturb_rel_l2_vs_fp32_mean_valid')}"
        )


if __name__ == "__main__":
    main()
