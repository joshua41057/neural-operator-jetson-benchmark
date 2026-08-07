#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

plan_path = Path("results/artifacts/deeponet_profile_plan.csv")
profile_dir = Path("results/profiles/deeponet_nsys")
out_path = Path("results/artifacts/deeponet_nsys_forward_summary.csv")

plan = list(csv.DictReader(plan_path.open()))

rows = []
for r in plan:
    pid = r["profile_id"]
    jpath = profile_dir / f"{pid}_forward.json"
    rep = profile_dir / f"{pid}.nsys-rep"
    sqlite = profile_dir / f"{pid}.sqlite"

    data = json.loads(jpath.read_text()) if jpath.exists() else {}

    rows.append({
        "profile_id": pid,
        "case_role": r["case_role"],
        "precision": r["precision"],
        "main_or_appendix": r["main_or_appendix"],
        "input_bank": r["input_bank"],
        "mean_ms": data.get("mean_ms", ""),
        "min_ms": data.get("min_ms", ""),
        "max_ms": data.get("max_ms", ""),
        "peak_cuda_allocated_mb": data.get("peak_cuda_allocated_mb", ""),
        "input_bank_shape": data.get("input_bank_shape", ""),
        "output_shape": data.get("output_shape", ""),
        "nsys_rep": str(rep),
        "nsys_rep_exists": rep.exists(),
        "sqlite": str(sqlite),
        "sqlite_exists": sqlite.exists(),
        "forward_json": str(jpath),
        "forward_json_exists": jpath.exists(),
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out_path}")
