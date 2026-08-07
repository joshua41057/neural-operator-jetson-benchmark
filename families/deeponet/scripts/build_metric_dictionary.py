#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROWS = [
    {
        "artifact_pattern": "results/jetson_deeponet_fp32/*.json",
        "source_level": "DeepONet short-run PyTorch harness",
        "primary_metrics": "mean_ms, median_ms, p95_ms, p99_ms, peak_cuda_allocated_mb",
        "metric_semantics": "Warm batch-size-one inference latency; PyTorch CUDA allocator peak.",
        "paper_use": "Main/appendix latency tables; CUDA working-set diagnostic.",
        "directly_comparable_to_fno": "Latency yes if same harness/backend/precision; CUDA allocation only if FNO also reports CUDA allocation.",
        "do_not_use_as": "Do not use peak_cuda_allocated_mb as board-level or process-resident RAM.",
    },
    {
        "artifact_pattern": "results/jetson_deeponet_precision/*.json",
        "source_level": "DeepONet precision execution harness",
        "primary_metrics": "status, precision, latency, peak_cuda_allocated_mb",
        "metric_semantics": "Precision-mode executability and short-run inference behavior.",
        "paper_use": "Precision feasibility table and appendix precision matrix.",
        "directly_comparable_to_fno": "Execution status yes; latency only for successful modes with same backend and input bank.",
        "do_not_use_as": "Do not interpret successful reduced precision as accuracy preservation without numerical perturbation check.",
    },
    {
        "artifact_pattern": "results/artifacts/deeponet_precision_numerics.csv",
        "source_level": "DeepONet numerical sanity check",
        "primary_metrics": "rel_l2_vs_fp32, max_abs_diff_vs_fp32, mean_abs_diff_vs_fp32",
        "metric_semantics": "Perturbation of reduced-precision predictions relative to FP32 predictions on benchmark input bank.",
        "paper_use": "Main compact numerical perturbation table; appendix full numerical table.",
        "directly_comparable_to_fno": "Comparable only as reduced-vs-FP32 perturbation if FNO has same diagnostic.",
        "do_not_use_as": "Do not report as PDE target error or held-out relative L2.",
    },
    {
        "artifact_pattern": "results/jetson_deeponet_long_energy/*_tegrastats_raw.log",
        "source_level": "DeepONet tegrastats sustained telemetry",
        "primary_metrics": "VDD_IN power, RAM, SWAP, GR3D_FREQ, EMC_FREQ, temperatures",
        "metric_semantics": "Board-level telemetry sampled during sustained inference.",
        "paper_use": "Long-run board-level energy, thermal behavior, and board RAM diagnostics.",
        "directly_comparable_to_fno": "Yes, if same tegrastats parsing, sustained window length, power mode, and clocks.",
        "do_not_use_as": "Do not interpret RAM as model-only memory.",
    },
    {
        "artifact_pattern": "results/profiles/deeponet_nsys/*.nsys-rep, *.sqlite",
        "source_level": "DeepONet Nsight Systems profile",
        "primary_metrics": "CUDA API summary, kernel summary, NVTX forward-range timeline",
        "metric_semantics": "Timeline and launch-level attribution, not detailed microarchitectural counters.",
        "paper_use": "Profiling diagnosis and appendix reproducibility evidence.",
        "directly_comparable_to_fno": "Comparable at operation-class level if same representative-case logic is used.",
        "do_not_use_as": "Do not use NSYS as cache-hit or occupancy source.",
    },
    {
        "artifact_pattern": "results/profiles/deeponet_ncu_basic/*.ncu-rep, *_raw.csv",
        "source_level": "DeepONet Nsight Compute profile",
        "primary_metrics": "kernel names, compute throughput, memory throughput, cache/occupancy metrics when present",
        "metric_semantics": "Profiler-perturbed per-kernel microarchitectural evidence.",
        "paper_use": "Appendix kernel evidence; main text operation-class diagnosis only.",
        "directly_comparable_to_fno": "Only for same metric names and same profiling configuration; otherwise mechanism-level comparison only.",
        "do_not_use_as": "Do not use NCU timings as primary end-to-end latency.",
    },
    {
        "artifact_pattern": "artifacts/summaries/*.json, artifacts/aggregates/*.json",
        "source_level": "Training/evaluation summary",
        "primary_metrics": "params, selected_seed, validation/test relative L2",
        "metric_semantics": "Predictive-validity and checkpoint-selection metadata.",
        "paper_use": "Validity gate and appendix checkpoint table.",
        "directly_comparable_to_fno": "Yes if same decoded-space relative-L2 definition and split policy.",
        "do_not_use_as": "Do not use validation L2 as deployment speed or feasibility metric.",
    },
]


def main() -> None:
    out = Path("results/artifacts/deeponet_metric_dictionary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact_pattern",
                "source_level",
                "primary_metrics",
                "metric_semantics",
                "paper_use",
                "directly_comparable_to_fno",
                "do_not_use_as",
            ],
        )
        writer.writeheader()
        writer.writerows(ROWS)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()