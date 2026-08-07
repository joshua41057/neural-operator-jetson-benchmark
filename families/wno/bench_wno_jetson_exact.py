from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from train_wno_burgers import WNO1d
from train_wno_darcy import WNO2d
from sample_codes.utils import LpLoss, count_params


PRECISION_MODES = [
    "fp32_strict",
    "tf32",
    "bf16_autocast",
    "fp16_autocast",
    "fp16_native",
]


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


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


def configure_precision_mode(precision_mode: str) -> dict[str, Any]:
    """
    Common precision policy, aligned with FNO/DeepONet.

    fp32_strict:
      model FP32, input FP32, autocast off, TF32 disabled.

    tf32:
      model FP32, input FP32, autocast off, TF32 enabled for eligible ops.

    bf16_autocast:
      model FP32, input FP32, forward under CUDA BF16 autocast, TF32 disabled.

    fp16_autocast:
      model FP32, input FP32, forward under CUDA FP16 autocast, TF32 disabled.

    fp16_native:
      model FP16, input FP16, autocast off, TF32 disabled.

    Important:
      This function mutates global PyTorch CUDA precision flags.
      Therefore it must be called immediately before any timed region.
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
        raise ValueError(f"Unsupported autocast dtype={dtype_name}")

    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


def build_wno_model(
    ckpt_path: str,
    dataset: str,
    device: str,
    precision_info: dict[str, Any],
):
    """
    Build the original WNO runtime path.

    No dtype/grid patch is applied here.
    That is intentional.

    For native FP16, if the original WNO path creates FP32 grid tensors and
    therefore fails against FP16 Linear/Conv weights, that is recorded as
    a native-FP16 executability failure. This matches the FNO/DeepONet policy:
    cast model/input according to the precision mode, but do not source-patch
    the architecture to force success.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    if dataset == "burgers":
        model = WNO1d(
            width=cfg["width"],
            level=cfg["level"],
            layers=cfg["layers"],
            size=cfg["res"],
            wavelet="db6",
            in_channel=2,
            grid_range=1,
            padding=0,
        )

    elif dataset == "darcy":
        model = WNO2d(
            width=cfg["width"],
            level=cfg["level"],
            layers=cfg["layers"],
            size=[cfg["res"], cfg["res"]],
            wavelet="db6",
            in_channel=3,
            grid_range=[1, 1],
            padding=1,
        )

    else:
        raise ValueError(f"Unsupported dataset={dataset}")

    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.eval().to(device)

    if precision_info["model_cast"] == "fp16":
        model = model.half()
    elif precision_info["model_cast"] == "fp32":
        model = model.float()
    else:
        raise ValueError(f"Unsupported model_cast={precision_info['model_cast']}")

    return model, ck, cfg


def cast_input(x: torch.Tensor, precision_info: dict[str, Any], device: str) -> torch.Tensor:
    if precision_info["input_cast"] == "fp16":
        x = x.half()
    elif precision_info["input_cast"] == "fp32":
        x = x.float()
    else:
        raise ValueError(f"Unsupported input_cast={precision_info['input_cast']}")
    return x.to(device)


def load_bank(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare_runtime_bank(bank: dict[str, Any], dataset: str, device: str) -> dict[str, Any]:
    """
    Prepare non-input runtime tensors outside the timed loop.

    For Darcy, y_mean/y_std are part of the deployed output decoding path,
    but they should not be copied to GPU inside every timed iteration.
    """
    runtime = dict(bank)

    if dataset == "darcy":
        runtime["_y_mean_device"] = bank["y_mean"].to(device).float()
        runtime["_y_std_device"] = bank["y_std"].to(device).float()
        runtime["_eps"] = float(bank["eps"])

    return runtime


def decoded_prediction(
    model,
    x: torch.Tensor,
    runtime_bank: dict[str, Any],
    dataset: str,
    precision_info: dict[str, Any],
    device: str,
):
    """
    Deployed prediction path.

    For Burgers:
      model(x) -> physical output.

    For Darcy:
      model(x) -> normalized output -> y_normalizer.decode output.

    Decode is included because FNO/DeepONet wrappers include output
    postprocessing in the deployed prediction path.
    """
    with autocast_context(device, precision_info):
        out = model(x)

    if dataset == "burgers":
        return out.reshape(out.shape[0], -1).float()

    if dataset == "darcy":
        b = out.shape[0]
        s = runtime_bank["y"].shape[-1]
        pred = out.reshape(b, s, s).float()
        pred = pred * (runtime_bank["_y_std_device"] + runtime_bank["_eps"]) + runtime_bank["_y_mean_device"]
        return pred.float()

    raise ValueError(dataset)


@torch.no_grad()
def evaluate_full_bank(
    ckpt_path: str,
    bank: dict[str, Any],
    dataset: str,
    precision_mode: str,
    device: str,
    eval_batch_size: int,
    return_predictions: bool,
):
    """
    Full-bank accuracy and optional prediction capture under one precision mode.

    This intentionally uses the same unpatched model path as sustained timing.
    If a precision mode is not executable, this function raises and the caller
    records the run as failed.
    """
    precision_info = configure_precision_mode(precision_mode)
    model, ck, cfg = build_wno_model(ckpt_path, dataset, device, precision_info)
    runtime_bank = prepare_runtime_bank(bank, dataset, device)

    y_all = bank["y"]
    n = y_all.shape[0]
    myloss = LpLoss(size_average=False)

    total = 0.0
    seen = 0
    preds_cpu: list[torch.Tensor] = []

    for i in range(0, n, eval_batch_size):
        x_cpu = bank["x"][i:i + eval_batch_size]
        y = y_all[i:i + eval_batch_size].to(device).float()
        x = cast_input(x_cpu, precision_info, device)

        pred = decoded_prediction(model, x, runtime_bank, dataset, precision_info, device)

        b = pred.shape[0]
        total += myloss(pred.reshape(b, -1), y.reshape(b, -1)).item()
        seen += b

        if return_predictions:
            preds_cpu.append(pred.detach().cpu())

    rel_l2 = total / seen
    preds = torch.cat(preds_cpu, dim=0) if return_predictions else None

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "rel_l2": float(rel_l2),
        "predictions": preds,
        "checkpoint_test_rel_l2": ck.get("test_rel_l2"),
        "config": cfg,
    }


def rel_l2_between(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred = pred.reshape(pred.shape[0], -1).float()
    ref = ref.reshape(ref.shape[0], -1).float()
    vals = torch.linalg.norm(pred - ref, dim=1) / torch.linalg.norm(ref, dim=1).clamp_min(1e-12)
    return float(vals.mean().item())


def start_tegrastats(log_path: Path, interval_ms: int):
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms), "--logfile", str(log_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        return None


def stop_tegrastats(proc) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def parse_tegrastats(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "tegrastats_samples": 0,
            "vdd_in_mean_w": None,
            "vdd_in_min_w": None,
            "vdd_in_max_w": None,
            "board_ram_mean_mb": None,
            "board_ram_peak_mb": None,
            "peak_temp_c": None,
            "gr3d_mean_pct": None,
            "gr3d_peak_pct": None,
        }

    vdd_w = []
    ram_mb = []
    temps = []
    gr3d = []

    # Common Jetson format:
    # VDD_IN 9123mW/8421mW
    # RAM 2430/7620MB
    # GR3D_FREQ 52%
    re_vdd = re.compile(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?")
    re_ram = re.compile(r"RAM\s+(\d+)/")
    re_temp = re.compile(r"@([0-9]+(?:\.[0-9]+)?)C")
    re_gr3d = re.compile(r"GR3D_FREQ\s+([0-9]+)%")

    for line in log_path.read_text(errors="ignore").splitlines():
        m = re_vdd.search(line)
        if m:
            vdd_w.append(float(m.group(1)) / 1000.0)

        m = re_ram.search(line)
        if m:
            ram_mb.append(float(m.group(1)))

        m = re_gr3d.search(line)
        if m:
            gr3d.append(float(m.group(1)))

        for tm in re_temp.findall(line):
            try:
                temps.append(float(tm))
            except Exception:
                pass

    return {
        "tegrastats_samples": len(vdd_w),
        "vdd_in_mean_w": statistics.mean(vdd_w) if vdd_w else None,
        "vdd_in_min_w": min(vdd_w) if vdd_w else None,
        "vdd_in_max_w": max(vdd_w) if vdd_w else None,
        "board_ram_mean_mb": statistics.mean(ram_mb) if ram_mb else None,
        "board_ram_peak_mb": max(ram_mb) if ram_mb else None,
        "peak_temp_c": max(temps) if temps else None,
        "gr3d_mean_pct": statistics.mean(gr3d) if gr3d else None,
        "gr3d_peak_pct": max(gr3d) if gr3d else None,
    }


@torch.no_grad()
def run_short(
    model,
    x: torch.Tensor,
    runtime_bank: dict[str, Any],
    dataset: str,
    precision_info: dict[str, Any],
    device: str,
    num_warmup: int,
    num_iters: int,
    dump_latencies: bool = False,
) -> dict[str, Any]:
    """Short-run timing class.

    Fixed-iteration-count window with no concurrent telemetry, matching the
    FNO/DeepONet short-run protocol (30 warmup iterations, 100 timed
    iterations, batch size one, per-iteration CUDA synchronization) so that
    cross-family comparisons stay inside a single measurement class.
    """
    is_cuda = str(device).startswith("cuda")
    times_ms: list[float] = []

    for _ in range(num_warmup):
        _ = decoded_prediction(model, x, runtime_bank, dataset, precision_info, device)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(num_iters):
        t0 = time.perf_counter()
        _ = decoded_prediction(model, x, runtime_bank, dataset, precision_info, device)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    peak_alloc = peak_reserved = None
    if is_cuda:
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        "timing_class": "short_run",
        "num_warmup": num_warmup,
        "num_iters": num_iters,
        "mean_latency_ms": statistics.mean(times_ms) if times_ms else None,
        "p50_latency_ms": statistics.median(times_ms) if times_ms else None,
        "p95_latency_ms": percentile(times_ms, 0.95),
        "p99_latency_ms": percentile(times_ms, 0.99),
        "min_latency_ms": min(times_ms) if times_ms else None,
        "max_latency_ms": max(times_ms) if times_ms else None,
        "cuda_peak_allocated_mb": peak_alloc,
        "cuda_peak_reserved_mb": peak_reserved,
        **({"latencies_ms": times_ms} if dump_latencies else {}),
    }


@torch.no_grad()
def run_sustained(
    model,
    x: torch.Tensor,
    runtime_bank: dict[str, Any],
    dataset: str,
    precision_info: dict[str, Any],
    device: str,
    warmup_seconds: float,
    measure_seconds: float,
    tegrastats_log: Path,
    tegrastats_interval_ms: int,
) -> dict[str, Any]:
    is_cuda = str(device).startswith("cuda")

    warmup_iters = 0
    measure_iters = 0
    times_ms: list[float] = []

    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < warmup_seconds:
        _ = decoded_prediction(model, x, runtime_bank, dataset, precision_info, device)
        warmup_iters += 1

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tg_proc = start_tegrastats(tegrastats_log, tegrastats_interval_ms)
    time.sleep(0.25)

    measure_start = time.perf_counter()
    try:
        while True:
            if time.perf_counter() - measure_start >= measure_seconds:
                break

            t0 = time.perf_counter()
            _ = decoded_prediction(model, x, runtime_bank, dataset, precision_info, device)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0)
            measure_iters += 1

    finally:
        measure_end = time.perf_counter()
        stop_tegrastats(tg_proc)

    elapsed = measure_end - measure_start
    tg = parse_tegrastats(tegrastats_log)

    throughput = measure_iters / elapsed if elapsed > 0 else None
    energy = None
    if throughput and tg["vdd_in_mean_w"] is not None:
        energy = tg["vdd_in_mean_w"] / throughput

    peak_alloc = None
    peak_reserved = None
    if is_cuda:
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        "warmup_seconds": warmup_seconds,
        "measure_seconds_requested": measure_seconds,
        "measure_seconds_actual": elapsed,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "throughput_inf_s": throughput,
        "mean_latency_ms": statistics.mean(times_ms) if times_ms else None,
        "p50_latency_ms": statistics.median(times_ms) if times_ms else None,
        "p95_latency_ms": percentile(times_ms, 0.95),
        "p99_latency_ms": percentile(times_ms, 0.99),
        "min_latency_ms": min(times_ms) if times_ms else None,
        "max_latency_ms": max(times_ms) if times_ms else None,
        "energy_j_per_inference": energy,
        "cuda_peak_allocated_mb": peak_alloc,
        "cuda_peak_reserved_mb": peak_reserved,
        **tg,
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--case-id", required=True)
    p.add_argument("--dataset", choices=["burgers", "darcy"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", required=True)

    p.add_argument(
        "--precision-mode",
        choices=PRECISION_MODES,
        required=True,
    )

    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=10)

    p.add_argument("--warmup-seconds", type=float, default=20.0)
    p.add_argument("--measure-seconds", type=float, default=120.0)
    p.add_argument(
        "--timing-class",
        choices=["sustained", "short_run"],
        default="sustained",
        help="sustained = 120 s window with tegrastats; "
             "short_run = fixed-iteration window, no telemetry (FNO/DeepONet protocol)",
    )
    p.add_argument("--num-warmup", type=int, default=30)
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument(
        "--dump-latencies", type=int, default=0,
        help="short_run only: retain the raw per-iteration latency array in result.json "
             "so any percentile estimator can be re-derived offline",
    )
    p.add_argument("--tegrastats-interval-ms", type=int, default=1000)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-root", default="results/jetson_wno_exact")
    p.add_argument("--run-tag", required=True)

    p.add_argument("--compute-full-eval", type=int, default=1)
    p.add_argument("--compute-perturbation", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.results_root) / args.run_tag / args.case_id / args.precision_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "result.json"
    tegrastats_log = out_dir / "tegrastats.log"

    result: dict[str, Any] = {
        "status": "unknown",
        "case_id": args.case_id,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "bank": args.bank,
        "precision_mode": args.precision_mode,
        "sample_index": args.sample_index,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "device": args.device,
        "run_tag": args.run_tag,
        "runtime_path": "original_wno_eager_no_dtype_patch",
        "protocol": "unified_prestaged",
        "timing_boundary": "preloaded_batch_size_one_request_tensor_plus_output_decode",
    }

    try:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        bank = load_bank(args.bank)
        runtime_bank = prepare_runtime_bank(bank, args.dataset, args.device)

        # Initial precision setup and model/input construction.
        precision_info = configure_precision_mode(args.precision_mode)
        model, ck, cfg = build_wno_model(
            args.checkpoint,
            args.dataset,
            args.device,
            precision_info,
        )

        parameter_count = count_params(model)

        x_cpu = bank["x"][args.sample_index:args.sample_index + args.batch_size]
        if x_cpu.shape[0] != args.batch_size:
            raise ValueError(
                f"Requested batch_size={args.batch_size} but got x shape {tuple(x_cpu.shape)}"
            )

        x = cast_input(x_cpu, precision_info, args.device)

        # Record provenance before full evaluation/timing so failed precision
        # paths still retain the exact policy and model metadata.
        result.update(
            {
                "config": cfg,
                "checkpoint_epoch": ck.get("epoch"),
                "parameter_count": parameter_count,
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "bank_x_shape": list(bank["x"].shape),
                "bank_y_shape": list(bank["y"].shape),
                "precision_info": precision_info,
                "tegrastats_log": str(tegrastats_log),
            }
        )

        eval_info: dict[str, Any] = {}

        if args.compute_full_eval:
            cur_eval = evaluate_full_bank(
                ckpt_path=args.checkpoint,
                bank=bank,
                dataset=args.dataset,
                precision_mode=args.precision_mode,
                device=args.device,
                eval_batch_size=args.eval_batch_size,
                return_predictions=bool(args.compute_perturbation),
            )
            eval_info["test_rel_l2"] = cur_eval["rel_l2"]
            eval_info["checkpoint_test_rel_l2"] = cur_eval["checkpoint_test_rel_l2"]

            if args.compute_perturbation:
                if args.precision_mode == "fp32_strict":
                    perturb = 0.0
                else:
                    ref_eval = evaluate_full_bank(
                        ckpt_path=args.checkpoint,
                        bank=bank,
                        dataset=args.dataset,
                        precision_mode="fp32_strict",
                        device=args.device,
                        eval_batch_size=args.eval_batch_size,
                        return_predictions=True,
                    )
                    perturb = rel_l2_between(cur_eval["predictions"], ref_eval["predictions"])

                eval_info["perturb_rel_l2_vs_fp32_case"] = perturb

        # Critical: full-bank eval / perturbation may have changed global CUDA flags.
        # Re-apply the requested precision policy immediately before timing.
        precision_info = configure_precision_mode(args.precision_mode)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        if args.timing_class == "short_run":
            timed = run_short(
                model=model,
                x=x,
                runtime_bank=runtime_bank,
                dataset=args.dataset,
                precision_info=precision_info,
                device=args.device,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                dump_latencies=bool(args.dump_latencies),
            )
        else:
            timed = run_sustained(
                model=model,
                x=x,
                runtime_bank=runtime_bank,
                dataset=args.dataset,
                precision_info=precision_info,
                device=args.device,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=args.measure_seconds,
                tegrastats_log=tegrastats_log,
                tegrastats_interval_ms=args.tegrastats_interval_ms,
            )
            timed.setdefault("timing_class", "sustained")

        result.update(
            {
                "status": "success",
                "config": cfg,
                "checkpoint_epoch": ck.get("epoch"),
                "parameter_count": parameter_count,
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "bank_x_shape": list(bank["x"].shape),
                "bank_y_shape": list(bank["y"].shape),
                "precision_info": precision_info,
                "tegrastats_log": str(tegrastats_log),
            }
        )
        result.update(eval_info)
        result.update(timed)

    except Exception as e:
        result.update(
            {
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
        )

    save_json(out_json, result)
    print(json.dumps(result, indent=2, default=str))
    print(f"Saved to {out_json}")

    if result["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
