#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

SUMMARY_DIRS = [
    Path("artifacts/summaries"),
    Path("artifacts/aggregates"),
]

OUT = Path("results/artifacts/paper_fno_quality_validity_table_rebuilt.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Baseline deployed checkpoints that appear in the manuscript validity gate.
BASELINE_ROWS = [
    ("burgers_fno_small_seed2", "Burgers", "small", "2048", 72033, 2),
    ("burgers_fno_base_seed3", "Burgers", "base", "2048", 235537, 3),
    ("burgers_fno_large_seed0", "Burgers", "large", "2048", 820033, 0),
    ("darcy_fno_small_seed4", "Darcy", "small", "141x141", 667713, 4),
    ("darcy_fno_base_seed0", "Darcy", "base", "141x141", 3287553, 0),
    ("darcy_fno_large_seed0", "Darcy", "large", "141x141", 28345217, 0),
]

# Also include resolution-scaling checkpoints, because reviewers will ask whether
# deployment scaling is measured on valid trained models.
RESOLUTION_ROWS = [
    ("burgers_fno_base_r512_seed2", "Burgers", "base_r512", "512", 235537, 2),
    ("burgers_fno_base_r1024_seed0", "Burgers", "base_r1024", "1024", 235537, 0),
    ("burgers_fno_base_r2048_seed2", "Burgers", "base_r2048", "2048", 235537, 2),
    ("burgers_fno_base_r4096_seed0", "Burgers", "base_r4096", "4096", 235537, 0),
    ("burgers_fno_base_r8192_seed1", "Burgers", "base_r8192", "8192", 235537, 1),
    ("darcy_fno_base_r85_seed2", "Darcy", "base_r85", "85x85", 3287553, 2),
    ("darcy_fno_base_r141_seed0", "Darcy", "base_r141", "141x141", 3287553, 0),
    ("darcy_fno_base_r211_seed1", "Darcy", "base_r211", "211x211", 3287553, 1),
    ("darcy_fno_base_r281_seed1", "Darcy", "base_r281", "281x281", 3287553, 1),
    ("darcy_fno_base_r421_seed1", "Darcy", "base_r421", "421x421", 3287553, 1),
]

ROWS = BASELINE_ROWS + RESOLUTION_ROWS

# Strict priority: prefer true held-out test relative L2 if present.
# Fall back to validation only if no test metric exists, and record source.
KEY_PRIORITY = [
    "test_rel_l2",
    "test_relative_l2",
    "test_relative_l2_error",
    "test_error_rel_l2",
    "test_error",
    "heldout_rel_l2",
    "held_out_rel_l2",
    "relative_l2",
    "rel_l2",
    "best_test_rel_l2",
    "best_test_relative_l2",
    "test_loss",
    "best_val_loss",
    "val_loss",
    "best_metric",
]

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(flatten(v, key))
    else:
        out[prefix] = obj
    return out

def numeric_or_blank(v):
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    if isinstance(v, str):
        try:
            x = float(v)
            if math.isfinite(x):
                return x
        except Exception:
            pass
    return None

def find_summary(exp):
    candidates = []
    for d in SUMMARY_DIRS:
        candidates += list(d.glob(f"{exp}_summary.json"))
        # aggregate files usually drop seed suffix.
        base_no_seed = exp.rsplit("_seed", 1)[0]
        candidates += list(d.glob(f"{base_no_seed}_aggregate_summary.json"))
    seen = []
    for p in candidates:
        if p.exists() and p not in seen:
            seen.append(p)
    return seen

def select_metric(flat):
    # 1) exact basename match by priority.
    for target in KEY_PRIORITY:
        for k, v in flat.items():
            if k.split(".")[-1].lower() == target.lower():
                x = numeric_or_blank(v)
                if x is not None:
                    return x, k

    # 2) fuzzy test relative L2 path.
    fuzzy_groups = [
        ("test relative L2", lambda k: "test" in k and "l2" in k and ("rel" in k or "relative" in k)),
        ("relative L2", lambda k: "l2" in k and ("rel" in k or "relative" in k)),
        ("test loss", lambda k: "test" in k and "loss" in k),
        ("val loss", lambda k: ("val" in k or "valid" in k) and "loss" in k),
        ("metric", lambda k: "metric" in k),
    ]
    for _, pred in fuzzy_groups:
        for k, v in flat.items():
            kl = k.lower()
            if pred(kl):
                x = numeric_or_blank(v)
                if x is not None:
                    return x, k

    return None, ""

records = []
for exp, task, group, resolution, params, seed in ROWS:
    metric = None
    source_key = ""
    source_file = ""
    found_files = find_summary(exp)

    for p in found_files:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        flat = flatten(data)
        metric, source_key = select_metric(flat)
        if metric is not None:
            source_file = str(p)
            break

    records.append({
        "experiment_name": exp,
        "task": task,
        "model_group": group,
        "resolution": resolution,
        "params": params,
        "selected_seed": seed,
        "heldout_or_best_available_metric": "" if metric is None else f"{metric:.8g}",
        "metric_source_key": source_key,
        "metric_source_file": source_file,
        "status": "FOUND" if metric is not None else "MISSING",
        "note": (
            "Use as held-out relative L2 only if metric_source_key is a decoded-space test relative L2 key."
            if metric is not None else
            "Metric not found in summaries/aggregates; rerun src.eval.evaluate_fno."
        ),
    })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    w.writerows(records)

print(f"Wrote {OUT}")
print()
for r in records:
    print(
        f"{r['experiment_name']:40s} "
        f"{r['status']:7s} "
        f"{r['heldout_or_best_available_metric']:>12s} "
        f"{r['metric_source_key']}"
    )
