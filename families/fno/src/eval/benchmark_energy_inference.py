from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from src.eval.common import load_model_and_normalizers
from src.utils.io import save_json


def load_input_bank(path: str, batch_size: int, sample_index: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload["x"]

    if x.ndim < 2:
        raise ValueError(f"Unexpected input bank tensor shape: {tuple(x.shape)}")

    if batch_size != 1:
        x = x[:batch_size]
    else:
        x = x[sample_index:sample_index + 1]

    meta = {k: v for k, v in payload.items() if k != "x"}
    return x, meta


def infer_task_resolution_from_checkpoint(ckpt_path: str) -> tuple[str, list[int], int | None]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dataset = str(cfg["data"]["dataset"])
    resolution = list(cfg["data"]["resolution"])

    param_count = None
    if isinstance(ckpt.get("summary", None), dict):
        param_count = ckpt["summary"].get("parameter_count", None)

    return dataset, resolution, param_count


def configure_precision_mode(precision_mode: str) -> dict[str, Any]:
    precision_mode = precision_mode.lower()

    info: dict[str, Any] = {
        "precision_mode": precision_mode,
        "model_cast": "fp32",
        "input_cast": "fp32",
        "autocast_enabled": False,
        "autocast_dtype": None,
        "allow_tf32_matmul": None,
        "allow_tf32_cudnn": None,
        "float32_matmul_precision": None,
    }

    if precision_mode == "fp32_strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
            info["float32_matmul_precision"] = "highest"
        except Exception:
            info["float32_matmul_precision"] = "unavailable"

    elif precision_mode == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
            info["float32_matmul_precision"] = "high"
        except Exception:
            info["float32_matmul_precision"] = "unavailable"

    else:
        raise ValueError(
            f"Unsupported precision_mode={precision_mode}. "
            "For long-run energy, use only fp32_strict or tf32. "
            "BF16/FP16 are handled in precision feasibility tables, not energy baseline."
        )

    info["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    info["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    return info


def prepare_model_and_input(model, x: torch.Tensor, precision_info: dict[str, Any], device: str):
    model = model.eval().to(device).float()
    x = x.float().to(device)
    return model, x


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


def run_for_duration(
    model,
    x: torch.Tensor,
    device: str,
    warmup_seconds: float,
    measure_seconds: float,
) -> dict[str, Any]:
    is_cuda = str(device).startswith("cuda")

    warmup_iters = 0
    measure_iters = 0
    times_ms: list[float] = []

    with torch.no_grad():
        warmup_start = time.perf_counter()
        while time.perf_counter() - warmup_start < warmup_seconds:
            _ = model(x)
            warmup_iters += 1

        if is_cuda:
            torch.cuda.synchronize()

        measure_start = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now - measure_start >= measure_seconds:
                break

            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0)
            measure_iters += 1

        measure_end = time.perf_counter()

    elapsed_s = measure_end - measure_start

    return {
        "warmup_seconds": warmup_seconds,
        "measure_seconds_requested": measure_seconds,
        "measure_seconds_actual": elapsed_s,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "throughput_inf_s": measure_iters / elapsed_s if elapsed_s > 0 else None,
        "mean_ms": statistics.mean(times_ms) if times_ms else None,
        "median_ms": statistics.median(times_ms) if times_ms else None,
        "p95_ms": percentile(times_ms, 0.95),
        "p99_ms": percentile(times_ms, 0.99),
        "min_ms": min(times_ms) if times_ms else None,
        "max_ms": max(times_ms) if times_ms else None,
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["eager", "torchscript"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--torchscript", type=str, default=None)

    p.add_argument("--input-bank", type=str, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)

    p.add_argument("--precision-mode", choices=["fp32_strict", "tf32"], required=True)
    p.add_argument("--warmup-seconds", type=float, default=20.0)
    p.add_argument("--measure-seconds", type=float, default=120.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", type=str, default="results/jetson_fno_energy_long")
    p.add_argument("--result-tag", type=str, required=True)

    return p.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.result_tag}.json"

    result: dict[str, Any] = {
        "status": "unknown",
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "torchscript": args.torchscript,
        "input_bank": args.input_bank,
        "sample_index": args.sample_index,
        "batch_size": args.batch_size,
        "precision_mode": args.precision_mode,
        "device": args.device,
    }

    try:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        if args.mode == "eager":
            ckpt, cfg, model, x_norm, y_norm, wrapper = load_model_and_normalizers(
                args.checkpoint, map_location="cpu"
            )
            model = wrapper
        elif args.mode == "torchscript":
            if args.torchscript is None:
                raise ValueError("TorchScript mode requires --torchscript")
            model = torch.jit.load(args.torchscript, map_location=args.device)
        else:
            raise ValueError(f"Unsupported mode={args.mode}")

        dataset, resolution, param_count = infer_task_resolution_from_checkpoint(args.checkpoint)
        x, bank_meta = load_input_bank(args.input_bank, args.batch_size, args.sample_index)

        precision_info = configure_precision_mode(args.precision_mode)
        model, x = prepare_model_and_input(model, x, precision_info, args.device)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        timed = run_for_duration(
            model=model,
            x=x,
            device=args.device,
            warmup_seconds=args.warmup_seconds,
            measure_seconds=args.measure_seconds,
        )

        peak_cuda_alloc_mb = None
        peak_cuda_reserved_mb = None
        if args.device.startswith("cuda"):
            peak_cuda_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_cuda_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        result.update(timed)
        result.update(
            {
                "status": "success",
                "dataset": dataset,
                "resolution": resolution,
                "parameter_count": param_count,
                "bank_meta": bank_meta,
                "precision_info": precision_info,
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
                "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
            }
        )

    except Exception as e:
        result.update(
            {
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
        )

    save_json(out_path, result)
    print(json.dumps(result, indent=2, default=str))
    print(f"Saved to {out_path}")

    if result["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
