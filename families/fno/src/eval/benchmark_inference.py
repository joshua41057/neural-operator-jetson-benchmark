from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from src.eval.common import load_model_and_normalizers
from src.utils.io import save_json


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


def time_inference(model, x, num_warmup: int, num_iters: int, device: str):
    times_ms = []
    is_cuda = str(device).startswith("cuda")

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
            # Peak allocator memory over the timed window only, matching the
            # short-run memory convention used in the appendix tables.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        for _ in range(num_iters):
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    peak_cuda_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if is_cuda else None
    )
    times_sorted = sorted(times_ms)
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": percentile(times_ms, 0.95),
        "p99_ms": percentile(times_ms, 0.99),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "num_warmup": num_warmup,
        "num_iters": num_iters,
        "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
    }


def infer_task_resolution_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dataset = str(cfg["data"]["dataset"])
    resolution = list(cfg["data"]["resolution"])
    return dataset, resolution


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["eager", "torchscript"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--torchscript", type=str, default=None)

    parser.add_argument("--input-bank", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)

    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=str, default="results/jetson_fno")
    parser.add_argument("--result-tag", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "eager":
        if args.checkpoint is None:
            raise ValueError("Eager mode requires --checkpoint.")
        if args.torchscript is not None:
            raise ValueError("Do not provide --torchscript in eager mode.")

        ckpt, cfg, model, x_norm, y_norm, wrapper = load_model_and_normalizers(
            args.checkpoint, map_location="cpu"
        )
        model = wrapper.eval().to(args.device)
        model = cast_model_precision(model, args.precision)

        dataset, resolution = infer_task_resolution_from_checkpoint(args.checkpoint)
        source = args.checkpoint

    elif args.mode == "torchscript":
        if args.torchscript is None:
            raise ValueError("TorchScript mode requires --torchscript.")

        model = torch.jit.load(args.torchscript, map_location=args.device).eval()
        model = cast_model_precision(model, args.precision)

        # metadata only
        if args.checkpoint is not None:
            dataset, resolution = infer_task_resolution_from_checkpoint(args.checkpoint)
        else:
            dataset = "unknown"
            resolution = "unknown"

        source = args.torchscript

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    x, bank_meta = load_input_bank(
        args.input_bank,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
    )
    x = cast_input_precision(x, args.precision).to(args.device)

    out = time_inference(
        model,
        x,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        device=args.device,
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
        }
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.result_tag is not None:
        stem = args.result_tag
    else:
        src_stem = Path(source).stem
        stem = f"{src_stem}_{args.mode}_{args.precision}"

    out_path = results_dir / f"{stem}.json"
    save_json(out_path, out)

    print(out)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()