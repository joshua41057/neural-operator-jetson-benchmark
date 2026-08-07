#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def infer_experiment_name(tag: str) -> str:
    for suffix in [
        "_torchscript_fp32_strict_120s",
        "_torchscript_tf32_120s",
        "_torchscript_bf16_autocast_120s",
        "_torchscript_fp16_autocast_120s",
        "_torchscript_fp16_native_120s",
        "_torchscript_fp32_strict_20s",
        "_torchscript_tf32_20s",
        "_torchscript_bf16_autocast_20s",
        "_torchscript_fp16_autocast_20s",
        "_torchscript_fp16_native_20s",
    ]:
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    # fallback
    return tag.split("_torchscript_")[0]


def load_manifest() -> dict[str, dict[str, str]]:
    manifest = Path("manifests/deeponet_jetson_manifest.csv")
    out: dict[str, dict[str, str]] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["experiment_name"]] = row
    return out


def main():
    results_dir = Path("results/jetson_deeponet_long_energy")
    out = Path("results/artifacts/deeponet_long_energy_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    rows = []

    for p in sorted(results_dir.glob("*.json")):
        d = json.load(open(p, "r", encoding="utf-8"))

        tag = d.get("result_tag", p.stem)
        exp = infer_experiment_name(tag)
        mr = manifest.get(exp, {})

        rows.append({
            "experiment_name": exp,
            "task": mr.get("dataset", ""),
            "spatial_dim": mr.get("spatial_dim", ""),
            "input_shape": mr.get("input_shape", ""),
            "parameter_count": mr.get("parameter_count", ""),
            "selected_seed": mr.get("selected_seed", ""),
            "test_rel_l2_mean": mr.get("test_rel_l2_mean", ""),
            "test_rel_l2_std": mr.get("test_rel_l2_std", ""),
            "sensor_resolution": mr.get("sensor_resolution", ""),
            "basis_dim": mr.get("basis_dim", ""),
            "branch_hidden_dim": mr.get("branch_hidden_dim", ""),
            "branch_layers": mr.get("branch_layers", ""),
            "trunk_hidden_dim": mr.get("trunk_hidden_dim", ""),
            "trunk_layers": mr.get("trunk_layers", ""),
            "result_tag": tag,
            "backend": d.get("mode"),
            "precision": d.get("precision"),
            "median_ms": d.get("median_ms"),
            "mean_ms": d.get("mean_ms"),
            "p95_ms": d.get("p95_ms"),
            "p99_ms": d.get("p99_ms"),
            "throughput_inf_s": d.get("throughput_inf_s"),
            "avg_power_w": d.get("avg_power_w"),
            "median_power_w": d.get("median_power_w"),
            "p95_power_w": d.get("p95_power_w"),
            "peak_power_w": d.get("peak_power_w"),
            "energy_per_inference_j": d.get("energy_per_inference_j"),
            "avg_gpu_temp_c": d.get("avg_gpu_temp_c"),
            "peak_gpu_temp_c": d.get("peak_gpu_temp_c"),
            "avg_cpu_temp_c": d.get("avg_cpu_temp_c"),
            "peak_cpu_temp_c": d.get("peak_cpu_temp_c"),
            "peak_cuda_allocated_mb": d.get("peak_cuda_allocated_mb"),
            "tegrastats_samples": d.get("tegrastats_samples"),
            "duration_sec_actual": d.get("duration_sec_actual"),
            "num_inferences": d.get("num_inferences"),
            "checkpoint": d.get("checkpoint"),
            "torchscript": d.get("torchscript"),
            "input_bank": d.get("input_bank"),
            "tegrastats_raw": d.get("tegrastats_raw"),
            "json": str(p),
        })

    if not rows:
        raise SystemExit(f"No JSON files found in {results_dir}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
