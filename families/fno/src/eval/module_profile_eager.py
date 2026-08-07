from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.eval.common import load_model_and_normalizers
from src.utils.io import save_json


def load_input_bank(path: str, batch_size: int, sample_index: int = 0) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload["x"]

    if batch_size != 1:
        x = x[:batch_size]
    else:
        x = x[sample_index:sample_index + 1]

    meta = {k: v for k, v in payload.items() if k != "x"}
    return x, meta


def cast_input_precision(x: torch.Tensor, precision: str) -> torch.Tensor:
    if precision == "fp32":
        return x.float()
    if precision == "fp16":
        return x.half()
    raise ValueError(precision)


def cast_model_precision(model, precision: str):
    if precision == "fp32":
        return model.float()
    if precision == "fp16":
        return model.half()
    raise ValueError(precision)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input-bank", type=str, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--result-tag", type=str, required=True)
    return p.parse_args()


def module_should_profile(name: str, module: torch.nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    if any(k in name.lower() for k in ["lift", "proj", "fc", "spectral", "conv", "block", "layer"]):
        return True
    if any(k in cls for k in ["spectral", "conv", "linear"]):
        return True
    return False


def shape_of(x: Any):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [shape_of(v) for v in x]
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items()}
    return str(type(x))


def main():
    args = parse_args()

    _, _, _, _, _, wrapper = load_model_and_normalizers(args.checkpoint, map_location="cpu")
    model = wrapper.eval().to(args.device)
    model = cast_model_precision(model, args.precision)

    x, bank_meta = load_input_bank(args.input_bank, args.batch_size, args.sample_index)
    x = cast_input_precision(x, args.precision).to(args.device)

    is_cuda = str(args.device).startswith("cuda")
    timings = defaultdict(list)
    meta = {}

    handles = []
    start_events = {}

    def pre_hook(name):
        def _hook(module, inputs):
            if is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                start_events[name] = (start, end)
            else:
                start_events[name] = time.perf_counter()
            if name not in meta:
                meta[name] = {"input_shape": shape_of(inputs)}
        return _hook

    def post_hook(name):
        def _hook(module, inputs, outputs):
            if is_cuda:
                start, end = start_events[name]
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                t0 = start_events[name]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            timings[name].append(elapsed_ms)
            if "output_shape" not in meta[name]:
                meta[name]["output_shape"] = shape_of(outputs)
                meta[name]["module_class"] = module.__class__.__name__
        return _hook

    for name, module in model.named_modules():
        if name == "":
            continue
        if module_should_profile(name, module):
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))

    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()

        total_times = []
        for _ in range(args.num_iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_times.append((t1 - t0) * 1000.0)

    for h in handles:
        h.remove()

    summary = []
    for name, vals in timings.items():
        summary.append(
            {
                "module_name": name,
                "module_class": meta[name].get("module_class", "unknown"),
                "mean_ms": sum(vals) / len(vals),
                "max_ms": max(vals),
                "calls": len(vals),
                "input_shape": meta[name].get("input_shape"),
                "output_shape": meta[name].get("output_shape"),
            }
        )

    summary.sort(key=lambda d: d["mean_ms"], reverse=True)

    out = {
        "checkpoint": args.checkpoint,
        "input_bank": args.input_bank,
        "precision": args.precision,
        "device": args.device,
        "batch_size": args.batch_size,
        "sample_index": args.sample_index,
        "num_warmup": args.num_warmup,
        "num_iters": args.num_iters,
        "forward_mean_ms": sum(total_times) / len(total_times),
        "module_summary": summary,
        "bank_meta": bank_meta,
    }

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{args.result_tag}.json"
    save_json(outpath, out)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()