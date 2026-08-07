#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

SUMMARY_DIR = Path("artifacts/summaries")
OUT = Path("results/artifacts/paper_fno_ablation_validity_table.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

ROWS = [
    ("burgers_fno_base_modes12_seed1", "Burgers", "modes12", "2048", 124945, 1),
    ("burgers_fno_base_modes16_seed2", "Burgers", "modes16", "2048", 161809, 2),
    ("burgers_fno_base_modes32_seed1", "Burgers", "modes32", "2048", 309265, 1),
    ("burgers_fno_base_nocoords_seed0", "Burgers", "nocoords", "2048", 235489, 0),
    ("burgers_fno_base_pad0_seed1", "Burgers", "pad0", "2048", 235537, 1),
    ("burgers_fno_base_pad40_seed1", "Burgers", "pad40", "2048", 235537, 1),

    ("darcy_fno_base_modes12_seed1", "Darcy", "modes12", "141x141", 1853953, 1),
    ("darcy_fno_base_modes24_seed1", "Darcy", "modes24", "141x141", 7383553, 1),
    ("darcy_fno_base_nocoords_seed2", "Darcy", "nocoords", "141x141", 3287473, 2),
    ("darcy_fno_base_pad0_seed1", "Darcy", "pad0", "141x141", 3287553, 1),
    ("darcy_fno_base_pad15_seed0", "Darcy", "pad15", "141x141", 3287553, 0),
]

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out

def get_metric(path):
    data = json.loads(path.read_text())
    flat = flatten(data)

    for k, v in flat.items():
        if k.split(".")[-1] == "test_rel_l2_mean":
            return float(v), k

    for k, v in flat.items():
        kl = k.lower()
        if "test" in kl and "l2" in kl and ("rel" in kl or "relative" in kl):
            try:
                return float(v), k
            except Exception:
                pass

    return None, ""

records = []
for exp, task, ablation, resolution, params, seed in ROWS:
    p = SUMMARY_DIR / f"{exp}_summary.json"
    metric, key = (None, "")
    if p.exists():
        metric, key = get_metric(p)

    records.append({
        "experiment_name": exp,
        "task": task,
        "ablation": ablation,
        "resolution": resolution,
        "params": params,
        "selected_seed": seed,
        "test_rel_l2_mean": "" if metric is None else f"{metric:.8g}",
        "metric_source_key": key,
        "metric_source_file": str(p) if p.exists() else "",
        "status": "FOUND" if metric is not None else "MISSING",
    })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    w.writerows(records)

print(f"Wrote {OUT}")
for r in records:
    print(f"{r['experiment_name']:42s} {r['status']:7s} {r['test_rel_l2_mean']}")
