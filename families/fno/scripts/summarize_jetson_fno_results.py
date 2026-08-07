from __future__ import annotations

import json
import csv
import re
from pathlib import Path

results_dir = Path("results/jetson_fno")
out_csv = Path("results/jetson_fno_summary.csv")

rows = []

def parse_tegrastats(path: Path):
    peak_ram_mb = None
    peak_vdd_in_mw = None
    avg_vdd_in_mw = None
    vals = []

    ram_re = re.compile(r"RAM (\d+)/(\d+)MB")
    pwr_re = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")

    if not path.exists():
        return {
            "peak_ram_mb": None,
            "peak_vdd_in_mw": None,
            "avg_vdd_in_mw": None,
        }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ram_re.search(line)
            if m:
                used = int(m.group(1))
                if peak_ram_mb is None or used > peak_ram_mb:
                    peak_ram_mb = used

            p = pwr_re.search(line)
            if p:
                inst = int(p.group(1))
                vals.append(inst)
                if peak_vdd_in_mw is None or inst > peak_vdd_in_mw:
                    peak_vdd_in_mw = inst

    if vals:
        avg_vdd_in_mw = sum(vals) / len(vals)

    return {
        "peak_ram_mb": peak_ram_mb,
        "peak_vdd_in_mw": peak_vdd_in_mw,
        "avg_vdd_in_mw": avg_vdd_in_mw,
    }

for json_path in sorted(results_dir.glob("*.json")):
    if json_path.name.startswith("smoke_"):
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    tag = json_path.stem
    tegra_path = results_dir / f"{tag}_tegrastats.log"
    tegra = parse_tegrastats(tegra_path)

    avg_power_w = tegra["avg_vdd_in_mw"] / 1000.0 if tegra["avg_vdd_in_mw"] is not None else None
    energy_j = None
    if avg_power_w is not None:
        energy_j = avg_power_w * (d["mean_ms"] / 1000.0)

    rows.append({
        "tag": tag,
        "dataset": d.get("dataset"),
        "resolution": d.get("resolution"),
        "mode": d.get("mode"),
        "precision": d.get("precision"),
        "batch_size": d.get("batch_size"),
        "mean_ms": d.get("mean_ms"),
        "median_ms": d.get("median_ms"),
        "p95_ms": d.get("p95_ms"),
        "p99_ms": d.get("p99_ms"),
        "min_ms": d.get("min_ms"),
        "max_ms": d.get("max_ms"),
        "device": d.get("device"),
        "source": d.get("source"),
        "input_bank": d.get("input_bank"),
        "sample_index": d.get("sample_index"),
        "peak_ram_mb": tegra["peak_ram_mb"],
        "avg_vdd_in_mw": tegra["avg_vdd_in_mw"],
        "peak_vdd_in_mw": tegra["peak_vdd_in_mw"],
        "avg_power_w": avg_power_w,
        "energy_per_inf_j": energy_j,
    })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_csv} with {len(rows)} rows")