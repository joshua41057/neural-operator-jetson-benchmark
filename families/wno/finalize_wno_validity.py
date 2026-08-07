from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


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
    "perturb_rel_l2_vs_fp32_case",
]


def base_case_id(case_id: str) -> str:
    return re.sub(r"_rep[0-9]+$", "", case_id)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def mean(vals):
    vals = [fnum(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def pstdev(vals):
    vals = [fnum(v) for v in vals]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.pstdev(vals)


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
            w.writerow({k: r.get(k) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    root = Path(args.run_dir)
    rows = []

    for p in sorted(root.glob("**/result.json")):
        r = json.loads(p.read_text())
        r["result_path"] = str(p)
        r["base_case_id"] = base_case_id(str(r.get("case_id", "")))
        r["runtime_status"] = r.get("status")

        if r.get("status") != "success":
            r["validity_status"] = "failed_runtime"
            r["paper_status"] = "Runtime Fail"
            r["validity_error_message"] = r.get("error_message")
        else:
            rel_ok = finite(r.get("test_rel_l2"))

            if r.get("precision_mode") == "fp32_strict":
                pert_ok = True
            else:
                pert_ok = finite(r.get("perturb_rel_l2_vs_fp32_case"))

            if rel_ok and pert_ok:
                r["validity_status"] = "valid"
                r["paper_status"] = "Success"
                r["validity_error_message"] = ""
            else:
                r["validity_status"] = "failed_numerical"
                r["paper_status"] = "Numerical Fail"
                r["validity_error_message"] = (
                    f"Non-finite metric: "
                    f"test_rel_l2={r.get('test_rel_l2')}, "
                    f"perturb={r.get('perturb_rel_l2_vs_fp32_case')}"
                )

        p.write_text(json.dumps(r, indent=2, default=str))
        rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["base_case_id"], r.get("dataset"), r.get("precision_mode"))].append(r)

    summary = []
    for (case_id, dataset, precision), rs in sorted(groups.items()):
        valid_rs = [r for r in rs if r["validity_status"] == "valid"]
        numfail_rs = [r for r in rs if r["validity_status"] == "failed_numerical"]
        runfail_rs = [r for r in rs if r["validity_status"] == "failed_runtime"]

        first = rs[0]
        cfg = first.get("config") or {}

        if len(valid_rs) > 0:
            paper_status = "Success"
        elif len(numfail_rs) > 0:
            paper_status = "Numerical Fail"
        else:
            paper_status = "Runtime Fail"

        out = {
            "case_id": case_id,
            "dataset": dataset,
            "precision_mode": precision,
            "paper_status": paper_status,
            "n_total": len(rs),
            "n_valid": len(valid_rs),
            "n_failed_numerical": len(numfail_rs),
            "n_failed_runtime": len(runfail_rs),
            "variant": cfg.get("variant"),
            "resolution": cfg.get("res"),
            "width": cfg.get("width"),
            "layers": cfg.get("layers"),
            "level": cfg.get("level"),
            "parameter_count": first.get("parameter_count"),
            "checkpoint_test_rel_l2": first.get("checkpoint_test_rel_l2"),
            "runtime_path": first.get("runtime_path"),
            "timing_boundary": first.get("timing_boundary"),
            "metric_eval_batch_size": first.get("metric_eval_batch_size"),
        }

        for m in METRICS:
            out[f"{m}_mean_valid"] = mean([r.get(m) for r in valid_rs])
            out[f"{m}_std_valid"] = pstdev([r.get(m) for r in valid_rs])

        out["numerical_failure_message"] = (
            numfail_rs[0].get("validity_error_message", "") if numfail_rs else ""
        )
        out["runtime_failure_message"] = (
            runfail_rs[0].get("error_message", "") if runfail_rs else ""
        )

        summary.append(out)

    fp32 = {
        r["case_id"]: r
        for r in summary
        if r["precision_mode"] == "fp32_strict" and r["paper_status"] == "Success"
    }

    for r in summary:
        base = fp32.get(r["case_id"])
        if not base or r["paper_status"] != "Success":
            r["speedup_vs_fp32_p50_valid"] = None
            r["energy_ratio_vs_fp32_valid"] = None
            continue

        fp32_p50 = fnum(base.get("p50_latency_ms_mean_valid"))
        cur_p50 = fnum(r.get("p50_latency_ms_mean_valid"))
        fp32_j = fnum(base.get("energy_j_per_inference_mean_valid"))
        cur_j = fnum(r.get("energy_j_per_inference_mean_valid"))

        r["speedup_vs_fp32_p50_valid"] = fp32_p50 / cur_p50 if fp32_p50 and cur_p50 else None
        r["energy_ratio_vs_fp32_valid"] = cur_j / fp32_j if fp32_j and cur_j else None

    invalid_rows = [r for r in rows if r["validity_status"] != "valid"]

    write_csv(root / "wno_final_validity_raw.csv", rows)
    write_csv(root / "wno_final_validity_summary.csv", summary)
    write_csv(root / "wno_final_invalid_results.csv", invalid_rows)

    c = Counter(r["validity_status"] for r in rows)

    print("run_dir:", root)
    print("total:", len(rows))
    print("valid:", c["valid"])
    print("failed_numerical:", c["failed_numerical"])
    print("failed_runtime:", c["failed_runtime"])
    print("summary:", root / "wno_final_validity_summary.csv")
    print("invalid:", root / "wno_final_invalid_results.csv")

    print("\nPaper-status summary:")
    for r in summary:
        print(
            f"{r['case_id']:30s} {r['precision_mode']:15s} "
            f"{r['paper_status']:15s} "
            f"valid={r['n_valid']} numfail={r['n_failed_numerical']} runfail={r['n_failed_runtime']} "
            f"p50={r.get('p50_latency_ms_mean_valid')} "
            f"pert={r.get('perturb_rel_l2_vs_fp32_case_mean_valid')}"
        )


if __name__ == "__main__":
    main()
