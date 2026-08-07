from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.eval.common import load_model_and_normalizers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eager", "torchscript"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--torchscript", default=None)
    p.add_argument("--input-bank", required=True)
    p.add_argument(
        "--precision",
        choices=[
            "fp32_strict",
            "fp32",
            "tf32",
            "bf16_autocast",
            "fp16_autocast",
            "fp16_native",
        ],
        required=True,
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile-iters", type=int, default=1)
    p.add_argument("--result-json", default=None)
    return p.parse_args()


def load_input_bank(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.float()

    if isinstance(obj, dict):
        for key in ["x", "inputs", "input", "data"]:
            if key in obj:
                return obj[key].float()

    raise RuntimeError(f"Unsupported input bank format: {path}")


def prepare_input(bank: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    if bank.shape[0] < batch_size:
        raise RuntimeError(
            f"Input bank has only {bank.shape[0]} samples, batch_size={batch_size}"
        )
    return bank[:batch_size].contiguous().to(device)


def canonical_precision(p: str) -> str:
    return "fp32_strict" if p == "fp32" else p


def set_precision_flags(precision: str):
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"


def cast_native(model, x: torch.Tensor, precision: str):
    if precision == "fp16_native":
        return model.half(), x.half()
    return model.float(), x.float()


def run_forward(model, x: torch.Tensor, precision: str):
    if precision == "bf16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x)

    if precision == "fp16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)

    return model(x)


def cuda_profiler_start():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    args = parse_args()
    precision = canonical_precision(args.precision)

    if args.device != "cuda":
        raise RuntimeError("NCU profiling should be run on CUDA device.")

    set_precision_flags(precision)

    bank = load_input_bank(args.input_bank)
    x = prepare_input(bank, args.batch_size, args.device)

    if args.mode == "torchscript":
        if args.torchscript is None:
            raise RuntimeError("--torchscript is required when mode=torchscript")
        model = torch.jit.load(args.torchscript, map_location=args.device)
        model.eval()
    else:
        _, _, _, _, _, model = load_model_and_normalizers(
            args.checkpoint,
            map_location="cpu",
        )
        model.eval().to(args.device)

    model, x = cast_native(model, x, precision)

    torch.cuda.reset_peak_memory_stats()

    # Warmup outside NCU profiling region.
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = run_forward(model, x, precision)
        torch.cuda.synchronize()

    lat_ms = []

    cuda_profiler_start()
    torch.cuda.nvtx.range_push("deeponet_ncu_forward")

    with torch.no_grad():
        for _ in range(args.profile_iters):
            t0 = time.perf_counter()
            y = run_forward(model, x, precision)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat_ms.append((t1 - t0) * 1000.0)

    torch.cuda.nvtx.range_pop()
    cuda_profiler_stop()

    out = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "torchscript": args.torchscript,
        "input_bank": args.input_bank,
        "input_bank_shape": list(bank.shape),
        "precision": precision,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "profile_iters": args.profile_iters,
        "output_shape": list(y.shape),
        "mean_ms_under_profiler": sum(lat_ms) / len(lat_ms),
        "min_ms_under_profiler": min(lat_ms),
        "max_ms_under_profiler": max(lat_ms),
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0**2),
        "note": "Latency under NCU is profiler-perturbed; use only as profiling sanity, not primary latency.",
    }

    print(json.dumps(out, indent=2))

    if args.result_json:
        p = Path(args.result_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
