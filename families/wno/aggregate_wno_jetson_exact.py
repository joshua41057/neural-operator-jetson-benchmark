from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = [
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "throughput_inf_s",
    "energy_j_per_inference",
    "vdd_in_mean_w",
    "vdd_in_min_w",
    "vdd_in_max_w",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "board_ram_mean_mb",
    "board_ram_peak_mb",
    "peak_temp_c",
    "gr3d_mean_pct",
    "gr3d_peak_pct",
    "test_rel_l2",
    "perturb_rel_l2_vs_fp32_case",
    "measure_seconds_actual",
    "measure_iters",
    "tegrastats_samples",
]


def base_case_id(case_id: str) -> str:
    return re.sub(r"_rep[0-9]+$", "", case_id)


def safe_float(x: Any):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def mean(vals):
    vals = [safe_float(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def pstdev(vals):
    vals = [safe_float(v) for v in vals]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.pstdev(vals)


def vmin(vals):
    vals = [safe_float(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def vmax(vals):
    vals = [safe_float(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def csv_value(x: Any):
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, default=str)
    return x


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for r in rows:
            w.writerow({k: csv_value(r.get(k)) for k in keys})


def load_results(run_dir: Path):
    rows = []
    for path in sorted(run_dir.glob("**/result.json")):
        try:
            r = json.loads(path.read_text())
        except Exception as e:
            rows.append({
                "status": "unreadable",
                "result_path": str(path),
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            continue

        r["result_path"] = str(path)
        r["base_case_id"] = base_case_id(str(r.get("case_id", "")))

        cfg = r.get("config") or {}
        if isinstance(cfg, dict):
            r["variant"] = cfg.get("variant")
            r["resolution"] = cfg.get("res")
            r["width"] = cfg.get("width")
            r["layers"] = cfg.get("layers")
            r["level"] = cfg.get("level")

        pi = r.get("precision_info") or {}
        if isinstance(pi, dict):
            r["model_cast"] = pi.get("model_cast")
            r["input_cast"] = pi.get("input_cast")
            r["autocast_enabled"] = pi.get("autocast_enabled")
            r["autocast_dtype"] = pi.get("autocast_dtype")
            r["allow_tf32_matmul"] = pi.get("allow_tf32_matmul")
            r["allow_tf32_cudnn"] = pi.get("allow_tf32_cudnn")
            r["float32_matmul_precision"] = pi.get("float32_matmul_precision")

        rows.append(r)

    return rows


def make_summary(rows: list[dict[str, Any]]):
    groups = defaultdict(list)

    for r in rows:
        key = (
            r.get("base_case_id"),
            r.get("dataset"),
            r.get("precision_mode"),
        )
        groups[key].append(r)

    summary = []

    for (case_id, dataset, precision), rs in sorted(groups.items()):
        success_rs = [r for r in rs if r.get("status") == "success"]
        fail_rs = [r for r in rs if r.get("status") != "success"]

        first = rs[0]
        out: dict[str, Any] = {
            "case_id": case_id,
            "dataset": dataset,
            "precision_mode": precision,
            "n_total": len(rs),
            "n_success": len(success_rs),
            "n_failed": len(fail_rs),
            "executability": "Success" if len(success_rs) > 0 else "Fail",
            "variant": first.get("variant"),
            "resolution": first.get("resolution"),
            "width": first.get("width"),
            "layers": first.get("layers"),
            "level": first.get("level"),
            "parameter_count": first.get("parameter_count"),
            "checkpoint": first.get("checkpoint"),
            "bank": first.get("bank"),
            "checkpoint_test_rel_l2": first.get("checkpoint_test_rel_l2"),
            "runtime_path": first.get("runtime_path"),
            "timing_boundary": first.get("timing_boundary"),
            "model_cast": first.get("model_cast"),
            "input_cast": first.get("input_cast"),
            "autocast_enabled": first.get("autocast_enabled"),
            "autocast_dtype": first.get("autocast_dtype"),
            "allow_tf32_matmul": first.get("allow_tf32_matmul"),
            "allow_tf32_cudnn": first.get("allow_tf32_cudnn"),
            "float32_matmul_precision": first.get("float32_matmul_precision"),
        }

        if fail_rs:
            out["failure_types"] = ";".join(sorted(set(str(r.get("error_type")) for r in fail_rs)))
            out["failure_messages"] = " | ".join(sorted(set(str(r.get("error_message")) for r in fail_rs)))[:1000]
        else:
            out["failure_types"] = ""
            out["failure_messages"] = ""

        for m in METRICS:
            vals = [r.get(m) for r in success_rs]
            out[f"{m}_mean"] = mean(vals)
            out[f"{m}_std"] = pstdev(vals)
            out[f"{m}_min"] = vmin(vals)
            out[f"{m}_max"] = vmax(vals)

        summary.append(out)

    # Add speedup relative to each case's FP32-strict p50/p95.
    fp32_by_case = {}
    for r in summary:
        if r["precision_mode"] == "fp32_strict" and r["n_success"] > 0:
            fp32_by_case[r["case_id"]] = r

    for r in summary:
        base = fp32_by_case.get(r["case_id"])
        if not base or r["n_success"] == 0:
            r["speedup_vs_fp32_p50"] = None
            r["speedup_vs_fp32_p95"] = None
            r["energy_ratio_vs_fp32"] = None
            continue

        fp32_p50 = safe_float(base.get("p50_latency_ms_mean"))
        cur_p50 = safe_float(r.get("p50_latency_ms_mean"))
        fp32_p95 = safe_float(base.get("p95_latency_ms_mean"))
        cur_p95 = safe_float(r.get("p95_latency_ms_mean"))
        fp32_j = safe_float(base.get("energy_j_per_inference_mean"))
        cur_j = safe_float(r.get("energy_j_per_inference_mean"))

        r["speedup_vs_fp32_p50"] = (fp32_p50 / cur_p50) if fp32_p50 and cur_p50 else None
        r["speedup_vs_fp32_p95"] = (fp32_p95 / cur_p95) if fp32_p95 and cur_p95 else None
        r["energy_ratio_vs_fp32"] = (cur_j / fp32_j) if fp32_j and cur_j else None

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir does not exist: {run_dir}")

    rows = load_results(run_dir)
    summary = make_summary(rows)

    failures = [r for r in rows if r.get("status") != "success"]
    successes = [r for r in rows if r.get("status") == "success"]

    raw_csv = run_dir / "wno_exact_raw_results.csv"
    summary_csv = run_dir / "wno_exact_summary.csv"
    failures_csv = run_dir / "wno_exact_failures.csv"

    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary)
    write_csv(failures_csv, failures)

    print("run_dir:", run_dir)
    print("result_jsons:", len(rows))
    print("success_jsons:", len(successes))
    print("failed_jsons:", len(failures))
    print("summary_rows:", len(summary))
    print("raw_csv:", raw_csv)
    print("summary_csv:", summary_csv)
    print("failures_csv:", failures_csv)

    expected_total = 10 * 5 * 3
    if len(rows) != expected_total:
        print(f"[WARN] Expected {expected_total} result.json files for full precision_all, found {len(rows)}.")
        print("[WARN] If the run is still in progress, aggregate again after it finishes.")

    print()
    print("Executability summary:")
    for r in summary:
        print(
            f"{r['case_id']:30s} {str(r['precision_mode']):15s} "
            f"success={r['n_success']} fail={r['n_failed']} "
            f"p50={r.get('p50_latency_ms_mean')} "
            f"p95={r.get('p95_latency_ms_mean')} "
            f"speedup={r.get('speedup_vs_fp32_p50')} "
            f"J={r.get('energy_j_per_inference_mean')} "
            f"pert={r.get('perturb_rel_l2_vs_fp32_case_mean')}"
        )


if __name__ == "__main__":
    main()
