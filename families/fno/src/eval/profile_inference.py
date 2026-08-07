from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from src.eval.common import load_model_and_normalizers
from src.utils.io import save_json

try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except Exception:
    HAS_NVTX = False


def nvtx_push(msg: str) -> None:
    if HAS_NVTX and torch.cuda.is_available():
        nvtx.range_push(msg)


def nvtx_pop() -> None:
    if HAS_NVTX and torch.cuda.is_available():
        nvtx.range_pop()


def load_input_bank(path: str, batch_size: int, sample_index: int = 0) -> tuple[torch.Tensor, dict]:
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


def cast_input_precision(x: torch.Tensor, precision: str) -> torch.Tensor:
    precision = precision.lower()
    if precision == "fp32":
        return x.float()
    if precision == "fp16":
        return x.half()
    raise ValueError(f"Unsupported precision: {precision}")


def cast_model_precision(model, precision: str):
    precision = precision.lower()
    if precision == "fp32":
        return model.float()
    if precision == "fp16":
        return model.half()
    raise ValueError(f"Unsupported precision: {precision}")


def infer_task_resolution_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dataset = str(cfg["data"]["dataset"])
    resolution = list(cfg["data"]["resolution"])
    return dataset, resolution


def maybe_reset_cuda_peak_memory(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device=device)


def maybe_get_cuda_peak_memory_mb(device: str) -> float | None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        return peak_bytes / (1024.0 * 1024.0)
    return None


def summarize_times(times_ms: list[float]) -> dict[str, Any]:
    times_sorted = sorted(times_ms)
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": times_sorted[int(0.95 * (len(times_sorted) - 1))],
        "p99_ms": times_sorted[int(0.99 * (len(times_sorted) - 1))],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def time_inference(
    model,
    x: torch.Tensor,
    num_warmup: int,
    num_iters: int,
    device: str,
    tag: str,
) -> dict[str, Any]:
    is_cuda = str(device).startswith("cuda")
    times_ms: list[float] = []

    maybe_reset_cuda_peak_memory(device)

    with torch.no_grad():
        nvtx_push(f"{tag}_warmup")
        for _ in range(num_warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        nvtx_pop()

        nvtx_push(f"{tag}_timed_region")
        for _ in range(num_iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            nvtx_push("forward")
            _ = model(x)
            nvtx_pop()

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        nvtx_pop()

    out = summarize_times(times_ms)
    out["cuda_peak_allocated_mb"] = maybe_get_cuda_peak_memory_mb(device)
    out["num_warmup"] = num_warmup
    out["num_iters"] = num_iters
    return out


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["eager", "torchscript"], required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--torchscript", type=str, default=None)

    p.add_argument("--input-bank", type=str, required=True)
    p.add_argument("--sample-index", type=int, default=0)

    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-warmup", type=int, default=20)
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--result-tag", type=str, required=True)
    p.add_argument("--notes", type=str, default=None)
    p.add_argument("--metadata-json", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "eager":
        if args.checkpoint is None:
            raise ValueError("Eager mode requires --checkpoint")
        if args.torchscript is not None:
            raise ValueError("Do not pass --torchscript in eager mode")

        nvtx_push("load_eager_model")
        _, _, _, _, _, wrapper = load_model_and_normalizers(args.checkpoint, map_location="cpu")
        model = wrapper.eval().to(args.device)
        model = cast_model_precision(model, args.precision)
        nvtx_pop()

        dataset, resolution = infer_task_resolution_from_checkpoint(args.checkpoint)
        source = args.checkpoint

    else:
        if args.torchscript is None:
            raise ValueError("TorchScript mode requires --torchscript")

        nvtx_push("load_torchscript_model")
        model = torch.jit.load(args.torchscript, map_location=args.device).eval()
        model = cast_model_precision(model, args.precision)
        nvtx_pop()

        if args.checkpoint is not None:
            dataset, resolution = infer_task_resolution_from_checkpoint(args.checkpoint)
        else:
            dataset, resolution = "unknown", "unknown"
        source = args.torchscript

    nvtx_push("load_input_bank")
    x, bank_meta = load_input_bank(args.input_bank, args.batch_size, args.sample_index)
    x = cast_input_precision(x, args.precision).to(args.device)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    nvtx_pop()

    extra_metadata = {}
    if args.metadata_json is not None:
        with open(args.metadata_json, "r", encoding="utf-8") as f:
            extra_metadata = json.load(f)

    out = time_inference(
        model=model,
        x=x,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        device=args.device,
        tag=args.result_tag,
    )

    out.update(
        {
            "dataset": dataset,
            "resolution": resolution,
            "batch_size": args.batch_size,
            "device": args.device,
            "mode": args.mode,
            "precision": args.precision,
            "input_bank": args.input_bank,
            "sample_index": args.sample_index,
            "source": source,
            "bank_meta": bank_meta,
            "notes": args.notes,
            "extra_metadata": extra_metadata,
        }
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.result_tag}.json"
    save_json(out_path, out)

    print(out)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()