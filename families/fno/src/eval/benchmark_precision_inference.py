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
    if "summary" in ckpt and isinstance(ckpt["summary"], dict):
        param_count = ckpt["summary"].get("parameter_count", None)

    return dataset, resolution, param_count


def configure_precision_mode(precision_mode: str) -> dict[str, Any]:
    """
    Precision modes:
      fp32_strict    : disable TF32 paths.
      tf32           : enable TF32 for matmul/cuDNN-eligible FP32 ops.
      bf16_autocast  : keep model/input FP32, run forward under CUDA BF16 autocast.
      fp16_autocast  : keep model/input FP32, run forward under CUDA FP16 autocast.
      fp16_native    : cast model/input to FP16 and run without autocast.
    """
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

    elif precision_mode == "bf16_autocast":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
            info["float32_matmul_precision"] = "highest"
        except Exception:
            info["float32_matmul_precision"] = "unavailable"
        info["autocast_enabled"] = True
        info["autocast_dtype"] = "bfloat16"

    elif precision_mode == "fp16_autocast":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
            info["float32_matmul_precision"] = "highest"
        except Exception:
            info["float32_matmul_precision"] = "unavailable"
        info["autocast_enabled"] = True
        info["autocast_dtype"] = "float16"

    elif precision_mode == "fp16_native":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
            info["float32_matmul_precision"] = "highest"
        except Exception:
            info["float32_matmul_precision"] = "unavailable"
        info["model_cast"] = "fp16"
        info["input_cast"] = "fp16"

    else:
        raise ValueError(f"Unsupported precision_mode={precision_mode}")

    info["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    info["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    return info


def autocast_context(device: str, precision_info: dict[str, Any]):
    if not str(device).startswith("cuda"):
        return nullcontext()

    if not precision_info["autocast_enabled"]:
        return nullcontext()

    dtype_name = precision_info["autocast_dtype"]
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported autocast dtype: {dtype_name}")

    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


def prepare_model_and_input(model, x: torch.Tensor, precision_info: dict[str, Any], device: str):
    model = model.eval().to(device)

    if precision_info["model_cast"] == "fp16":
        model = model.half()
    elif precision_info["model_cast"] == "fp32":
        model = model.float()
    else:
        raise ValueError(f"Unsupported model_cast={precision_info['model_cast']}")

    if precision_info["input_cast"] == "fp16":
        x = x.half()
    elif precision_info["input_cast"] == "fp32":
        x = x.float()
    else:
        raise ValueError(f"Unsupported input_cast={precision_info['input_cast']}")

    x = x.to(device)
    return model, x


def percentile(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 0:
        return float("nan")
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def run_timed_repeats(
    model,
    x: torch.Tensor,
    precision_info: dict[str, Any],
    device: str,
    num_warmup: int,
    num_iters: int,
    repeats: int,
) -> dict[str, Any]:
    is_cuda = str(device).startswith("cuda")

    all_times: list[float] = []
    repeat_summaries: list[dict[str, float]] = []

    with torch.no_grad():
        for rep in range(repeats):
            for _ in range(num_warmup):
                with autocast_context(device, precision_info):
                    _ = model(x)

            if is_cuda:
                torch.cuda.synchronize()

            times_ms = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                with autocast_context(device, precision_info):
                    _ = model(x)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

            s = sorted(times_ms)
            rep_summary = {
                "repeat": float(rep),
                "mean_ms": statistics.mean(times_ms),
                "median_ms": statistics.median(times_ms),
                "p95_ms": percentile(s, 0.95),
                "p99_ms": percentile(s, 0.99),
                "min_ms": min(times_ms),
                "max_ms": max(times_ms),
            }
            repeat_summaries.append(rep_summary)
            all_times.extend(times_ms)

    all_sorted = sorted(all_times)
    rep_medians = [r["median_ms"] for r in repeat_summaries]
    rep_means = [r["mean_ms"] for r in repeat_summaries]

    return {
        "mean_ms": statistics.mean(all_times),
        "median_ms": statistics.median(all_times),
        "p95_ms": percentile(all_sorted, 0.95),
        "p99_ms": percentile(all_sorted, 0.99),
        "min_ms": min(all_times),
        "max_ms": max(all_times),
        "repeat_mean_ms_median": statistics.median(rep_means),
        "repeat_median_ms_median": statistics.median(rep_medians),
        "repeat_median_ms_std": statistics.pstdev(rep_medians) if len(rep_medians) > 1 else 0.0,
        "num_warmup": num_warmup,
        "num_iters": num_iters,
        "repeats": repeats,
        "repeat_summaries": repeat_summaries,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["eager", "torchscript"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--torchscript", type=str, default=None)

    parser.add_argument("--input-bank", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument(
        "--precision-mode",
        choices=["fp32_strict", "tf32", "bf16_autocast", "fp16_autocast", "fp16_native"],
        required=True,
    )

    parser.add_argument("--num-warmup", type=int, default=30)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--results-dir", type=str, default="results/jetson_fno_precision")
    parser.add_argument("--result-tag", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.result_tag}.json"

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

        x, bank_meta = load_input_bank(
            args.input_bank,
            batch_size=args.batch_size,
            sample_index=args.sample_index,
        )

        precision_info = configure_precision_mode(args.precision_mode)
        model, x = prepare_model_and_input(model, x, precision_info, args.device)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        timed = run_timed_repeats(
            model=model,
            x=x,
            precision_info=precision_info,
            device=args.device,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            repeats=args.repeats,
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
