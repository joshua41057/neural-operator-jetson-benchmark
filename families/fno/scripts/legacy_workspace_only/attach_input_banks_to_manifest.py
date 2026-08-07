from __future__ import annotations

import csv
import json
from pathlib import Path

manifest = Path("manifests/fno_jetson_manifest.csv")

with open(manifest, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    dataset = row["dataset"].lower()
    input_shape = json.loads(row["input_shape"])

    if dataset == "burgers":
        r = int(input_shape[0])
        row["input_bank_path"] = f"artifacts/benchmark_inputs/burgers_r{r}_bank.pt"
    elif dataset == "darcy":
        r = int(input_shape[0])
        row["input_bank_path"] = f"artifacts/benchmark_inputs/darcy_r{r}_bank.pt"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

with open(manifest, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {manifest}")