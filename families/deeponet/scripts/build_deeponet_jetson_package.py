from __future__ import annotations

import csv
import shutil
from pathlib import Path

selection_csv = Path("results/deeponet_seed_selection.csv")
pkg_root = Path("jetson_deeponet_package")

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

def input_bank_for(dataset: str, input_shape: str) -> str:
    s = input_shape.replace(" ", "")
    if dataset == "burgers":
        if s.startswith("[512,"):
            return "artifacts/benchmark_inputs/burgers_r512_bank.pt"
        if s.startswith("[1024,"):
            return "artifacts/benchmark_inputs/burgers_r1024_bank.pt"
        if s.startswith("[2048,"):
            return "artifacts/benchmark_inputs/burgers_r2048_bank.pt"
        if s.startswith("[4096,"):
            return "artifacts/benchmark_inputs/burgers_r4096_bank.pt"
    if dataset == "darcy":
        if s.startswith("[85,85,"):
            return "artifacts/benchmark_inputs/darcy_r85_bank.pt"
        if s.startswith("[141,141,"):
            return "artifacts/benchmark_inputs/darcy_r141_bank.pt"
        if s.startswith("[211,211,"):
            return "artifacts/benchmark_inputs/darcy_r211_bank.pt"
        if s.startswith("[281,281,"):
            return "artifacts/benchmark_inputs/darcy_r281_bank.pt"
    raise ValueError(f"Cannot infer input bank for dataset={dataset}, input_shape={input_shape}")

rows_out = []

with open(selection_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        exp = row["experiment_name"]
        seed = int(row["selected_seed"])
        dataset = row["dataset"]

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
        ts_dst = dirs["ts"] / f"{exp}_seed{seed}.ts"

        for src in [ckpt_src, cfg_src, sum_src, agg_src]:
            if not src.exists():
                raise FileNotFoundError(src)

        shutil.copy2(ckpt_src, ckpt_dst)
        shutil.copy2(cfg_src, cfg_dst)
        shutil.copy2(sum_src, sum_dst)
        shutil.copy2(agg_src, agg_dst)

        out = dict(row)
        out.update({
            "checkpoint_path": str(ckpt_dst.relative_to(pkg_root)),
            "config_path": str(cfg_dst.relative_to(pkg_root)),
            "summary_path": str(sum_dst.relative_to(pkg_root)),
            "aggregate_path": str(agg_dst.relative_to(pkg_root)),
            "torchscript_path": str(ts_dst.relative_to(pkg_root)),
            "run_eager_fp32": 1,
            "run_ts_fp32": 1,
            "run_precision": 1,
            "input_bank_path": input_bank_for(dataset, row["input_shape"]),
        })
        rows_out.append(out)

manifest_path = dirs["manifest"] / "deeponet_jetson_manifest.csv"
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Wrote manifest to {manifest_path}")
print(f"Prepared package at {pkg_root}")
print(f"Selected DeepONet checkpoints: {len(rows_out)}")
