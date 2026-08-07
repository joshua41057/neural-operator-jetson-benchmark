from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

from bench_wno_jetson_exact import evaluate_full_bank, rel_l2_between


PRECISION_ORDER = [
    "fp32_strict",
    "tf32",
    "bf16_autocast",
    "fp16_autocast",
    "fp16_native",
]


def base_case_id(case_id: str) -> str:
    return re.sub(r"_rep[0-9]+$", "", case_id)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch-size", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.run_dir)
    result_paths = sorted(root.glob("**/result.json"))

    rows = []
    for p in result_paths:
        r = json.loads(p.read_text())
        r["_path"] = str(p)
        r["_base_case_id"] = base_case_id(str(r.get("case_id", "")))
        rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["_base_case_id"], r.get("precision_mode"))].append(r)

    # Determine one representative JSON per case/precision.
    reps = {}
    for key, rs in groups.items():
        reps[key] = rs[0]

    # Compute FP32 reference predictions once per base case.
    fp32_refs = {}
    fp32_metrics = {}

    base_cases = sorted(set(r["_base_case_id"] for r in rows))

    for case in base_cases:
        key = (case, "fp32_strict")
        if key not in reps:
            print(f"[WARN] missing fp32_strict for {case}")
            continue

        r = reps[key]
        print(f"[EVAL1] FP32 reference {case}")

        try:
            out = evaluate_full_bank(
                ckpt_path=r["checkpoint"],
                bank=torch.load(r["bank"], map_location="cpu", weights_only=False),
                dataset=r["dataset"],
                precision_mode="fp32_strict",
                device=args.device,
                eval_batch_size=args.eval_batch_size,
                return_predictions=True,
            )
            fp32_refs[case] = out["predictions"]
            fp32_metrics[case] = out
        except Exception as e:
            print(f"[FAILED] FP32 reference {case}: {type(e).__name__}: {e}")

    # Recompute all successful non-native metrics with eval batch size 1.
    computed = {}

    for case in base_cases:
        for precision in PRECISION_ORDER:
            key = (case, precision)
            if key not in reps:
                continue

            r = reps[key]

            # Native FP16 already failed at runtime; do not try to turn it into a metric run.
            if precision == "fp16_native":
                computed[key] = {
                    "status": "failed_runtime",
                    "test_rel_l2": None,
                    "perturb": None,
                    "error": r.get("error_message"),
                }
                continue

            if precision == "fp32_strict":
                out = fp32_metrics.get(case)
                if out is None:
                    computed[key] = {
                        "status": "failed_eval",
                        "test_rel_l2": None,
                        "perturb": None,
                        "error": "missing fp32 reference",
                    }
                else:
                    computed[key] = {
                        "status": "success",
                        "test_rel_l2": float(out["rel_l2"]),
                        "perturb": 0.0,
                        "checkpoint_test_rel_l2": out.get("checkpoint_test_rel_l2"),
                        "error": None,
                    }
                continue

            print(f"[EVAL1] {case} {precision}")

            try:
                out = evaluate_full_bank(
                    ckpt_path=r["checkpoint"],
                    bank=torch.load(r["bank"], map_location="cpu", weights_only=False),
                    dataset=r["dataset"],
                    precision_mode=precision,
                    device=args.device,
                    eval_batch_size=args.eval_batch_size,
                    return_predictions=True,
                )

                ref = fp32_refs.get(case)
                if ref is None:
                    perturb = None
                else:
                    perturb = rel_l2_between(out["predictions"], ref)

                computed[key] = {
                    "status": "success",
                    "test_rel_l2": float(out["rel_l2"]),
                    "perturb": float(perturb) if perturb is not None else None,
                    "checkpoint_test_rel_l2": out.get("checkpoint_test_rel_l2"),
                    "error": None,
                }

            except Exception as e:
                computed[key] = {
                    "status": "failed_eval",
                    "test_rel_l2": None,
                    "perturb": None,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"[FAILED] {case} {precision}: {type(e).__name__}: {e}")

    # Update every result JSON while preserving timing/energy.
    updated = 0

    for r in rows:
        key = (r["_base_case_id"], r.get("precision_mode"))
        c = computed.get(key)
        p = Path(r["_path"])

        r.pop("_path", None)
        r.pop("_base_case_id", None)

        r["metric_eval_batch_size"] = args.eval_batch_size
        r["metric_recomputed_eval_batch_size_one"] = True

        if c is None:
            r["metric_recompute_status"] = "missing"
        else:
            r["metric_recompute_status"] = c["status"]

            if c["status"] == "success":
                r["test_rel_l2"] = c["test_rel_l2"]
                r["perturb_rel_l2_vs_fp32_case"] = c["perturb"]
                if c.get("checkpoint_test_rel_l2") is not None:
                    r["checkpoint_test_rel_l2"] = c["checkpoint_test_rel_l2"]

            elif c["status"] == "failed_runtime":
                # Preserve original runtime failure fields.
                pass

            else:
                r["metric_recompute_error"] = c.get("error")

        # Validity classification.
        if r.get("status") != "success":
            r["runtime_status"] = "failed"
            r["validity_status"] = "failed_runtime"
            r["paper_status"] = "Runtime Fail"
        else:
            rel_ok = finite(r.get("test_rel_l2"))
            if r.get("precision_mode") == "fp32_strict":
                pert_ok = True
            else:
                pert_ok = finite(r.get("perturb_rel_l2_vs_fp32_case"))

            if rel_ok and pert_ok:
                r["runtime_status"] = "success"
                r["validity_status"] = "valid"
                r["paper_status"] = "Success"
            else:
                r["runtime_status"] = "success"
                r["validity_status"] = "failed_numerical"
                r["paper_status"] = "Numerical Fail"
                r["validity_error_message"] = (
                    f"Non-finite metric under eval_batch_size={args.eval_batch_size}: "
                    f"test_rel_l2={r.get('test_rel_l2')}, "
                    f"perturb={r.get('perturb_rel_l2_vs_fp32_case')}"
                )

        p.write_text(json.dumps(r, indent=2, default=str))
        updated += 1

    print(f"[OK] updated result JSON files: {updated}")
    print(f"[OK] preserved existing 120s timing/energy values")
    print(f"[OK] metric_eval_batch_size={args.eval_batch_size}")


if __name__ == "__main__":
    main()
