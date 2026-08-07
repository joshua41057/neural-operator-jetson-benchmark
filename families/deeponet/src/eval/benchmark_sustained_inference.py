from __future__ import annotations

import argparse
import json
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

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
        choices=["fp32_strict", "fp32", "tf32", "bf16_autocast", "fp16_autocast", "fp16_native"],
        default="fp32_strict",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-sec", type=float, default=10.0)
    p.add_argument(
        "--unified-protocol", type=int, default=0,
        help="cross-family unified protocol: hold a single preloaded device-resident "
             "request tensor for the whole window (matching FNO/WNO/Sp2GNO) instead of "
             "cycling the bank, and use the warmup/telemetry settings passed in",
    )
    p.add_argument("--duration-sec", type=float, default=120.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tegrastats-interval-ms", type=int, default=1000)
    p.add_argument("--results-dir", default="results/jetson_deeponet_long_energy")
    p.add_argument("--result-tag", required=True)
    return p.parse_args()


def load_bank(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.Tensor):
        return obj.float().contiguous()

    if isinstance(obj, dict):
        for key in ["x", "inputs", "input", "bank", "data", "x_raw"]:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].float().contiguous()
        raise KeyError(f"No tensor input found in bank dict. Keys={list(obj.keys())}")

    raise TypeError(f"Unsupported input bank type: {type(obj)}")


def configure_precision(precision: str) -> str:
    precision = precision.lower()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if precision == "fp32":
        precision = "fp32_strict"

    return precision


def run_forward(model, x: torch.Tensor, precision: str):
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


def start_tegrastats(out_path: Path, interval_ms: int):
    f = out_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, f


def stop_tegrastats(proc, f):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        try:
            f.close()
        except Exception:
            pass


def parse_tegrastats(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="ignore") if path.exists() else ""

    powers_w: list[float] = []
    gpu_temps_c: list[float] = []
    cpu_temps_c: list[float] = []
    ram_used_mb: list[float] = []

    for line in text.splitlines():
        # Jetson common: VDD_IN 12243mW/12158mW
        m = re.search(r"VDD_IN\s+([0-9]+)mW", line)
        if m:
            powers_w.append(float(m.group(1)) / 1000.0)

        # Sometimes rail name spacing varies.
        if not m:
            m_alt = re.search(r"VDD_[A-Z0-9_]+\s+([0-9]+)mW", line)
            if m_alt and "VDD_IN" in line:
                powers_w.append(float(m_alt.group(1)) / 1000.0)

        m_gpu = re.search(r"GPU@([0-9.]+)C", line)
        if m_gpu:
            gpu_temps_c.append(float(m_gpu.group(1)))

        m_cpu = re.search(r"CPU@([0-9.]+)C", line)
        if m_cpu:
            cpu_temps_c.append(float(m_cpu.group(1)))

        m_ram = re.search(r"RAM\s+([0-9]+)/", line)
        if m_ram:
            ram_used_mb.append(float(m_ram.group(1)))

    def mean_or_none(xs):
        return float(statistics.mean(xs)) if xs else None

    def median_or_none(xs):
        return float(statistics.median(xs)) if xs else None

    return {
        "tegrastats_samples": len(powers_w),
        "avg_power_w": mean_or_none(powers_w),
        "median_power_w": median_or_none(powers_w),
        "p95_power_w": percentile(powers_w, 0.95) if powers_w else None,
        "peak_power_w": max(powers_w) if powers_w else None,
        "avg_gpu_temp_c": mean_or_none(gpu_temps_c),
        "peak_gpu_temp_c": max(gpu_temps_c) if gpu_temps_c else None,
        "avg_cpu_temp_c": mean_or_none(cpu_temps_c),
        "peak_cpu_temp_c": max(cpu_temps_c) if cpu_temps_c else None,
        "avg_ram_used_mb": mean_or_none(ram_used_mb),
        "peak_ram_used_mb": max(ram_used_mb) if ram_used_mb else None,
    }


def main():
    args = parse_args()

    device = torch.device(args.device)
    precision = configure_precision(args.precision)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_json = results_dir / f"{args.result_tag}.json"
    raw_tegrastats = results_dir / f"{args.result_tag}_tegrastats_raw.log"

    bank = load_bank(args.input_bank)
    print(f"Loaded input bank: {args.input_bank}, shape={tuple(bank.shape)}, dtype={bank.dtype}")

    if args.mode == "eager":
        _, _, _, _, _, model = load_model_and_normalizers(args.checkpoint, map_location="cpu")
        model.eval().to(device)
    else:
        if args.torchscript is None:
            raise ValueError("--torchscript is required for torchscript mode")
        model = torch.jit.load(args.torchscript, map_location=device)
        model.eval()

    if precision == "fp16_native":
        model = model.half()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup before tegrastats starts.
    warmup_end = time.perf_counter() + float(args.warmup_sec)
    i = 0
    with torch.no_grad():
        while time.perf_counter() < warmup_end:
            x = get_batch(bank, i * args.batch_size, args.batch_size, device)
            _ = run_forward(model, x, precision)
            i += 1
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Unified protocol: one preloaded device-resident request, staged before timing.
    fixed_x = get_batch(bank, 0, args.batch_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ts_proc, ts_file = start_tegrastats(raw_tegrastats, args.tegrastats_interval_ms)

    lat_ms: list[float] = []
    count = 0
    t_start = time.perf_counter()

    try:
        with torch.no_grad():
            while True:
                elapsed = time.perf_counter() - t_start
                if elapsed >= args.duration_sec:
                    break

                x = fixed_x if args.unified_protocol else get_batch(
                    bank, count, args.batch_size, device
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                y = run_forward(model, x, precision)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                # Force output materialization without adding it to timed interval.
                _ = float(y.float().mean().detach().cpu())

                lat_ms.append((t1 - t0) * 1000.0)
                count += args.batch_size

    finally:
        stop_tegrastats(ts_proc, ts_file)

    duration_actual = time.perf_counter() - t_start
    throughput = count / duration_actual if duration_actual > 0 else None

    power = parse_tegrastats(raw_tegrastats)
    avg_power = power.get("avg_power_w")
    energy_per_inf = None
    if avg_power is not None and throughput is not None and throughput > 0:
        energy_per_inf = avg_power / throughput

    peak_cuda_allocated_mb = None
    if device.type == "cuda":
        peak_cuda_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = {
        "result_tag": args.result_tag,
        "mode": args.mode,
        "precision": precision,
        "checkpoint": args.checkpoint,
        "torchscript": args.torchscript,
        "input_bank": args.input_bank,
        "input_bank_shape": list(bank.shape),
        "batch_size": args.batch_size,
        "warmup_sec": args.warmup_sec,
        "protocol": "unified_prestaged" if args.unified_protocol else "family_native",
        "duration_sec_requested": args.duration_sec,
        "duration_sec_actual": duration_actual,
        "num_inferences": count,
        "throughput_inf_s": throughput,
        "mean_ms": float(statistics.mean(lat_ms)),
        "median_ms": float(statistics.median(lat_ms)),
        "p95_ms": percentile(lat_ms, 0.95),
        "p99_ms": percentile(lat_ms, 0.99),
        "min_ms": float(min(lat_ms)),
        "max_ms": float(max(lat_ms)),
        "std_ms": float(statistics.stdev(lat_ms)) if len(lat_ms) >= 2 else 0.0,
        "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
        "energy_per_inference_j": energy_per_inf,
        "tegrastats_raw": str(raw_tegrastats),
        **power,
    }

    out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
