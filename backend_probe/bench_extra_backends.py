#!/usr/bin/env python3
"""
Cross-backend batch-size-1 latency benchmark for FNO on Jetson Orin Nano.

Extends the paper's eager/TorchScript comparison (Appendix B) with two more
runtime paths: torch.compile (Inductor) and TensorRT (ONNX export -> trtexec
-> TensorRT execution context). Reuses the exact model/checkpoint/input-bank
loading code already used to produce Table 16 (src.eval.common,
artifacts/benchmark_inputs), so the compared runtime paths share the same
learned operator and input as the published FP32 numbers.

Not a sustained/telemetry benchmark: short-run median/P95 over `--reps`
repetitions after `--warmup` warmup calls, matching the paper's warm,
synchronized, batch-size-one timing convention (Sec. 4.3) minus the
tegrastats telemetry collection.

Run from the EDCNO repo root, inside the `extra_bench` conda env:
    conda activate extra_bench
    export PYTHONNOUSERSITE=1; unset PYTHONPATH
    cd /home/jetson/jjyoo3/EDCNO
    python extra_inference/bench_extra_backends.py --case burgers_base_r2048
    python extra_inference/bench_extra_backends.py --case darcy_base_r141
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "_extra_bench_common"))

from src.eval.common import load_model_and_normalizers  # noqa: E402
from harness import time_callable, try_backend  # noqa: E402
from trt_utils import export_onnx, build_engine, run_engine  # noqa: E402

CASES = {
    "burgers_base_r2048": dict(
        checkpoint="artifacts/checkpoints/burgers_fno_base_r2048_seed2_best.pt",
        torchscript="artifacts/torchscript/burgers_fno_base_r2048_seed2.ts",
        input_bank="artifacts/benchmark_inputs/burgers_r2048_bank.pt",
    ),
    "darcy_base_r141": dict(
        checkpoint="artifacts/checkpoints/darcy_fno_base_r141_seed0_best.pt",
        torchscript="artifacts/torchscript/darcy_fno_base_r141_seed0.ts",
        input_bank="artifacts/benchmark_inputs/darcy_r141_bank.pt",
    ),
}

WARMUP = 5
REPS = 20


def load_sample_input(bank_path: Path) -> torch.Tensor:
    payload = torch.load(bank_path, map_location="cpu", weights_only=False)
    x = payload["x"][0:1]
    return x.float().cuda()


def bench_eager(wrapper, x):
    return time_callable(lambda: wrapper(x), warmup=WARMUP, reps=REPS)


def bench_torchscript(ts_path, x):
    ts_model = torch.jit.load(str(ts_path), map_location="cuda").eval()
    return time_callable(lambda: ts_model(x), warmup=WARMUP, reps=REPS)


def bench_compile(wrapper, x):
    compiled = torch.compile(wrapper)
    return time_callable(lambda: compiled(x), warmup=WARMUP, reps=REPS)


def bench_tensorrt(wrapper, x, workdir: Path, tag: str):
    onnx_path = workdir / f"{tag}.onnx"
    engine_path = workdir / f"{tag}.engine"
    export_onnx(wrapper, (x,), str(onnx_path), ["x"], ["y"], opset=17)
    ok, log = build_engine(str(onnx_path), str(engine_path))
    if not ok:
        raise RuntimeError(f"trtexec engine build failed:\n{log}")
    result = run_engine(str(engine_path), warmup=WARMUP, reps=REPS)
    if result.get("status") != "success":
        raise RuntimeError(f"TRT engine run failed: {result.get('error')}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    args = ap.parse_args()

    case = CASES[args.case]
    ckpt_path = REPO_ROOT / case["checkpoint"]
    ts_path = REPO_ROOT / case["torchscript"]
    bank_path = REPO_ROOT / case["input_bank"]

    _, _, _, _, _, wrapper = load_model_and_normalizers(str(ckpt_path), map_location="cpu")
    wrapper = wrapper.eval().cuda()
    x = load_sample_input(bank_path)

    workdir = Path(__file__).parent / "trt_work"
    workdir.mkdir(exist_ok=True)

    results = {
        "family": "FNO",
        "case": args.case,
        "checkpoint": case["checkpoint"],
        "input_shape": list(x.shape),
        "warmup": WARMUP,
        "reps": REPS,
        "backends": {},
    }

    results["backends"]["eager"] = try_backend("eager", lambda: bench_eager(wrapper, x))
    results["backends"]["torchscript"] = try_backend("torchscript", lambda: bench_torchscript(ts_path, x))
    results["backends"]["torch.compile"] = try_backend("torch.compile", lambda: bench_compile(wrapper, x))
    results["backends"]["tensorrt"] = try_backend(
        "tensorrt", lambda: bench_tensorrt(wrapper, x, workdir, args.case)
    )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.case}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== FNO / {args.case} ===")
    for name, r in results["backends"].items():
        if r["status"] == "success":
            print(f"  {name:14s} median={r['median_ms']:.3f} ms  p95={r['p95_ms']:.3f} ms")
        else:
            print(f"  {name:14s} FAIL  ({r['error_type']}: {r['error'][:150]})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
