from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from src.eval.common import load_model_and_normalizers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eager", "torchscript"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--torchscript", type=str, default=None)
    p.add_argument("--input-bank", type=str, required=True)
    p.add_argument("--precision", type=str, default="fp32",
                   choices=["fp32", "fp32_strict", "tf32", "bf16_autocast", "fp16_autocast", "fp16_native"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--result-tag", type=str, required=True)
    return p.parse_args()


def load_input_bank(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        for key in ["x", "inputs", "input", "bank", "data", "x_raw"]:
            if key in obj and isinstance(obj[key], torch.Tensor):
                x = obj[key]
                break
        else:
            keys = list(obj.keys())
            raise KeyError(f"Could not find input tensor in input bank. Available keys: {keys}")
    elif isinstance(obj, (list, tuple)):
        tensors = [v for v in obj if isinstance(v, torch.Tensor)]
        if not tensors:
            raise TypeError("Input bank list/tuple contains no tensors")
        x = tensors[0]
    else:
        raise TypeError(f"Unsupported input bank type: {type(obj)}")

    # Expected raw input:
    # Burgers: [N, L] or [L]
    # Darcy:   [N, H, W] or [H, W]
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim == 2:
        # Ambiguous: [N,L] for 1D or [H,W] for one 2D field.
        # Darcy banks normally have [N,H,W], so leave [N,L] as is.
        pass
    if x.ndim == 3:
        pass

    return x.float().contiguous()


def configure_precision(precision: str):
    precision = precision.lower()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return precision


def percentile(vals, q: float) -> float:
    """Linear-interpolation percentile (numpy `percentile` method='linear').

    Unified across all four operator-family harnesses so that P95/P99 are
    estimator-identical. q is a fraction in [0, 1]. Median is reported through
    statistics.median, which coincides with this estimator at q = 0.5.
    """
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return float(s[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def get_batch(bank: torch.Tensor, start: int, batch_size: int, device: torch.device) -> torch.Tensor:
    n = bank.shape[0]
    idx = [(start + i) % n for i in range(batch_size)]
    return bank[idx].to(device, non_blocking=False)


def run_forward(model, x, precision: str):
    if precision in {"fp32", "fp32_strict", "tf32"}:
        return model(x)

    if precision == "bf16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x)

    if precision == "fp16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)

    if precision == "fp16_native":
        return model(x.half())

    raise ValueError(f"Unsupported precision: {precision}")


def main():
    args = parse_args()

    device = torch.device(args.device)
    precision = configure_precision(args.precision)

    bank = load_input_bank(args.input_bank)
    print(f"Loaded input bank: {args.input_bank}, shape={tuple(bank.shape)}, dtype={bank.dtype}")

    if args.mode == "eager":
        _, _, _, _, _, model = load_model_and_normalizers(args.checkpoint, map_location="cpu")
        model.eval().to(device)
    else:
        if args.torchscript is None:
            raise ValueError("--torchscript is required when --mode torchscript")
        model = torch.jit.load(args.torchscript, map_location=device)
        model.eval()

    if precision == "fp16_native":
        model = model.half()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    with torch.no_grad():
        for i in range(args.num_warmup):
            x = get_batch(bank, i * args.batch_size, args.batch_size, device)
            _ = run_forward(model, x, precision)
        if device.type == "cuda":
            torch.cuda.synchronize()

    lat_ms = []

    with torch.no_grad():
        for i in range(args.num_iters):
            x = get_batch(bank, i * args.batch_size, args.batch_size, device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            y = run_forward(model, x, precision)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Force materialization.
            _ = float(y.float().mean().detach().cpu())
            lat_ms.append((t1 - t0) * 1000.0)

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    out = {
        "result_tag": args.result_tag,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "torchscript": args.torchscript,
        "input_bank": args.input_bank,
        "input_bank_shape": list(bank.shape),
        "precision": precision,
        "batch_size": args.batch_size,
        "num_warmup": args.num_warmup,
        "num_iters": args.num_iters,
        "device": str(device),
        "mean_ms": float(statistics.mean(lat_ms)),
        "median_ms": float(statistics.median(lat_ms)),
        "p95_ms": percentile(lat_ms, 0.95),
        "p99_ms": percentile(lat_ms, 0.99),
        "min_ms": float(min(lat_ms)),
        "max_ms": float(max(lat_ms)),
        "std_ms": float(statistics.stdev(lat_ms)) if len(lat_ms) >= 2 else 0.0,
        "peak_cuda_allocated_mb": peak_mem_mb,
    }

    out_path = results_dir / f"{args.result_tag}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
