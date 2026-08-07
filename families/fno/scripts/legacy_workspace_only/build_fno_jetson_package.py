# scripts/build_fno_jetson_package.py
from __future__ import annotations

import csv
import shutil
from pathlib import Path

selection_csv = Path("results/fno_seed_selection.csv")
pkg_root = Path("jetson_fno_package")

dirs = {
    "ckpt": pkg_root / "artifacts/checkpoints",
    "cfg": pkg_root / "artifacts/configs",
    "sum": pkg_root / "artifacts/summaries",
    "agg": pkg_root / "artifacts/aggregates",
    "ts": pkg_root / "artifacts/torchscript",
    "manifest": pkg_root / "manifests",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

rows_out = []

with open(selection_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        exp = row["experiment_name"]
        seed = int(row["selected_seed"])

        src_seed_dir = Path("checkpoints") / exp / f"seed{seed}"
        src_exp_dir = Path("checkpoints") / exp

        ckpt_src = src_seed_dir / "best.pt"
        cfg_src = src_seed_dir / "resolved_config.json"
        sum_src = src_seed_dir / "summary.json"
        agg_src = src_exp_dir / "aggregate_summary.json"

        ckpt_dst = dirs["ckpt"] / f"{exp}_seed{seed}_best.pt"
        cfg_dst = dirs["cfg"] / f"{exp}_seed{seed}_resolved_config.json"
        sum_dst = dirs["sum"] / f"{exp}_seed{seed}_summary.json"
        agg_dst = dirs["agg"] / f"{exp}_aggregate_summary.json"

        shutil.copy2(ckpt_src, ckpt_dst)
        shutil.copy2(cfg_src, cfg_dst)
        shutil.copy2(sum_src, sum_dst)
        shutil.copy2(agg_src, agg_dst)

        rows_out.append({
            "experiment_name": exp,
            "selected_seed": seed,
            "selection_rule": row["selection_rule"],
            "checkpoint_path": str(ckpt_dst),
            "config_path": str(cfg_dst),
            "summary_path": str(sum_dst),
            "aggregate_path": str(agg_dst),
            "torchscript_path": str(dirs["ts"] / f"{exp}_seed{seed}.ts"),
            "dataset": row["dataset"],
            "spatial_dim": row["spatial_dim"],
            "input_shape": row["input_shape"],
            "output_shape": row["output_shape"],
            "parameter_count": row["parameter_count"],
            "run_eager_fp32": 1,
            "run_eager_fp16": 1,
            "run_ts_fp32": 1,
            "run_ts_fp16": 1,
        })

manifest_path = dirs["manifest"] / "fno_jetson_manifest.csv"
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Wrote manifest to {manifest_path}")
print(f"Prepared package at {pkg_root}")