#!/usr/bin/env python3
"""
bench_sp2gno_jetson.py

Paper-grade Jetson inference harness for pretrained Sp2GNO checkpoints.

What it measures:
  - checkpoint-level test rel-L2
  - batch-size-one sustained inference latency
  - P50/P95/mean latency
  - board-level tegrastats VDD_IN power
  - energy per inference
  - CUDA allocator peak memory
  - board RAM and thermal telemetry
  - precision executability and output perturbation vs FP32

Assumptions:
  - Run from: /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026
  - Existing checkpoints:
      runs/burgers/ckpt/burgers_best.pth
      runs/darcy/ckpt/darcy_best.pth
  - Existing graph cache:
      cache/burgers_s1024_k8_f64.pt
      cache/darcy_s85_k20_f64.pt
"""

import argparse
import csv
import datetime as _dt
import glob
import json
import math
import os
import platform
import re
import shlex
import signal
import statistics as stats
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext

import numpy as np
import scipy.io as sio

import torch

from sp2gno_core import (
    Sp2GNO,
    SharedGraph,
    set_all_seeds,
)


# ---------------------------------------------------------------------
# Data builders: identical normalization/splits to training scripts
# ---------------------------------------------------------------------

def _load_checkpoint(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("model_state_dict", ck)
    model.load_state_dict(state, strict=True)
    return ck


def _build_features_burgers(a, u, coord_feat, mean, std):
    M, s = a.shape
    an = ((a - mean) / std)[..., None]
    feats = torch.cat([
        coord_feat.unsqueeze(0).expand(M, -1, -1),
        torch.from_numpy(an)
    ], dim=-1)
    Y = torch.from_numpy(u.reshape(M, s, 1))
    return feats.contiguous(), Y.contiguous()


def _build_features_darcy(coeff, sol, coord_feat, mean, std):
    M, s, _ = coeff.shape
    N = s * s
    cf = ((coeff.reshape(M, N) - mean) / std)[..., None]
    feats = torch.cat([
        coord_feat.unsqueeze(0).expand(M, -1, -1),
        torch.from_numpy(cf)
    ], dim=-1)
    Y = torch.from_numpy(sol.reshape(M, N, 1))
    return feats.contiguous(), Y.contiguous()


def _load_or_build_graph(cache_path, pos, k, num_freq, device):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        gfeat = torch.load(cache_path, map_location="cpu")
    else:
        gfeat = build_graph_features(
            pos, k=k, num_freq=num_freq, seed=0, eig_device=device
        )
        torch.save(gfeat, cache_path)
    return gfeat


def build_burgers(args, device):
    mat_path = os.path.join(args.data_dir, "burgers_data_R10.mat")
    split_path = args.burgers_split

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing Burgers data: {mat_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing Burgers split: {split_path}")

    mat = sio.loadmat(mat_path)
    a_all = mat["a"][:, ::args.sub].astype(np.float32)
    u_all = mat["u"][:, ::args.sub].astype(np.float32)
    s = a_all.shape[1]
    N = s

    with open(split_path) as f:
        split = json.load(f)

    tr_i, te_i = split["train"], split["test"]
    tr_a = a_all[tr_i]
    te_a = a_all[te_i]
    te_u = u_all[te_i]

    mean, std = float(tr_a.mean()), float(tr_a.std())

    xcoord = np.linspace(0, 1, s, dtype=np.float32)
    pos = xcoord[:, None].astype(np.float32)
    coord_feat = torch.from_numpy(pos)

    test_feats, test_Y = _build_features_burgers(te_a, te_u, coord_feat, mean, std)

    cache_path = os.path.join(args.cache_dir, f"burgers_s{s}_k{args.k}_f{args.num_freq}.pt")
    gfeat = _load_or_build_graph(cache_path, pos, args.k, args.num_freq, device)
    graph = SharedGraph(gfeat, device)

    model = Sp2GNO(
        in_dim=2,
        width=args.width,
        n_layers=args.n_layers,
        N=N,
        num_freq=args.num_freq,
        out_dim=1,
    ).to(device)

    ckpt_path = args.ckpt or os.path.join(args.run_dir, "burgers", "ckpt", "burgers_best.pth")
    ck = _load_checkpoint(model, ckpt_path, device)

    meta = {
        "dataset": "burgers",
        "resolution": s,
        "num_nodes": N,
        "num_test": int(test_feats.shape[0]),
        "input_norm_mean": mean,
        "input_norm_std": std,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ck.get("epoch", ""),
        "checkpoint_val_rel_l2": ck.get("val_rel_l2", ""),
        "checkpoint_test_rel_l2": ck.get("test_rel_l2", ""),
        "cache_path": cache_path,
    }
    return model, graph, test_feats, test_Y, meta


def _darcy_indices(res):
    # Evenly-spaced indices of the native 421-grid (round(linspace(0,420,res))).
    # Matches the WNO/REPRODUCTION_NOTES protocol: --res 85/141/211/281/421 are
    # true 85/141/.../421 grids. Simple integer ::r striding cannot represent
    # res=281 (420/280 = 1.5, non-uniform spacing), so this replaces it.
    return np.round(np.linspace(0, 420, res)).astype(int)


def _load_darcy_mat(path, n, idx):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Darcy data: {path}")
    d = sio.loadmat(path)
    coeff = d["coeff"][:n][:, idx][:, :, idx].astype(np.float32)
    sol = d["sol"][:n][:, idx][:, :, idx].astype(np.float32)
    return coeff, sol


def build_darcy(args, device):
    p1 = os.path.join(args.data_dir, "piececonst_r421_N1024_smooth1.mat")
    p2 = os.path.join(args.data_dir, "piececonst_r421_N1024_smooth2.mat")

    res = args.res if args.res else (420 // args.r) + 1
    idx = _darcy_indices(res)

    coeff1, sol1 = _load_darcy_mat(p1, args.ntrain + args.nval, idx)
    coeff2, sol2 = _load_darcy_mat(p2, args.ntest, idx)

    s = coeff1.shape[1]
    N = s * s

    tr_coeff = coeff1[:args.ntrain]
    te_coeff = coeff2
    te_sol = sol2

    mean, std = float(tr_coeff.mean()), float(tr_coeff.std())

    gx, gy = np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s), indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    coord_feat = torch.from_numpy(pos)

    test_feats, test_Y = _build_features_darcy(te_coeff, te_sol, coord_feat, mean, std)

    cache_path = os.path.join(args.cache_dir, f"darcy_s{s}_k{args.k}_f{args.num_freq}.pt")
    gfeat = _load_or_build_graph(cache_path, pos, args.k, args.num_freq, device)
    graph = SharedGraph(gfeat, device)

    model = Sp2GNO(
        in_dim=3,
        width=args.width,
        n_layers=args.n_layers,
        N=N,
        num_freq=args.num_freq,
        out_dim=1,
    ).to(device)

    ckpt_path = args.ckpt or os.path.join(args.run_dir, "darcy", "ckpt", "darcy_best.pth")
    ck = _load_checkpoint(model, ckpt_path, device)

    meta = {
        "dataset": "darcy",
        "resolution": s,
        "num_nodes": N,
        "num_test": int(test_feats.shape[0]),
        "input_norm_mean": mean,
        "input_norm_std": std,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ck.get("epoch", ""),
        "checkpoint_val_rel_l2": ck.get("val_rel_l2", ""),
        "checkpoint_test_rel_l2": ck.get("test_rel_l2", ""),
        "cache_path": cache_path,
    }
    return model, graph, test_feats, test_Y, meta


# ---------------------------------------------------------------------
# Precision config
# ---------------------------------------------------------------------

def configure_precision(precision):
    precision = precision.lower()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    if precision == "fp32_strict":
        return {"autocast": None, "model_dtype": torch.float32}

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return {"autocast": None, "model_dtype": torch.float32}

    if precision == "bf16_autocast":
        return {"autocast": torch.bfloat16, "model_dtype": torch.float32}

    if precision == "fp16_autocast":
        return {"autocast": torch.float16, "model_dtype": torch.float32}

    if precision == "fp16_native":
        return {"autocast": None, "model_dtype": torch.float16}

    raise ValueError(f"Unknown precision: {precision}")


def precision_info(precision, precision_cfg):
    ac = precision_cfg["autocast"]
    if ac is torch.float16:
        ac_name = "float16"
    elif ac is torch.bfloat16:
        ac_name = "bfloat16"
    else:
        ac_name = None

    md = precision_cfg["model_dtype"]
    if md is torch.float16:
        model_cast = "fp16"
    else:
        model_cast = "fp32"

    try:
        matmul_precision = torch.get_float32_matmul_precision()
    except Exception:
        matmul_precision = "unknown"

    return {
        "precision_mode": precision,
        "model_cast": model_cast,
        "input_cast": "fp16" if precision == "fp16_native" else "fp32",
        "autocast_enabled": ac is not None,
        "autocast_dtype": ac_name,
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": matmul_precision,
    }


def autocast_context(precision_cfg):
    dtype = precision_cfg["autocast"]
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def cast_model_for_precision(model, precision_cfg):
    dtype = precision_cfg["model_dtype"]
    if dtype == torch.float16:
        return model.half()
    return model.float()


# ---------------------------------------------------------------------
# Tegrastats monitor and parser
# ---------------------------------------------------------------------

class TegraMonitor:
    def __init__(self, log_path, interval_ms=200, enabled=True):
        self.log_path = log_path
        self.interval_ms = int(interval_ms)
        self.enabled = enabled
        self.proc = None

    def __enter__(self):
        if not self.enabled:
            return self
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        cmd = ["tegrastats", "--interval", str(self.interval_ms), "--logfile", self.log_path]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        time.sleep(max(0.05, self.interval_ms / 1000.0))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        time.sleep(0.1)


def parse_tegrastats(log_path):
    out = {
        "tegrastats_samples": 0,
        "vdd_in_mean_w": math.nan,
        "vdd_in_min_w": math.nan,
        "vdd_in_max_w": math.nan,
        "vdd_in_last_avg_field_w": math.nan,
        "board_ram_mean_mb": math.nan,
        "board_ram_peak_mb": math.nan,
        "peak_temp_c": math.nan,
        "gr3d_mean_pct": math.nan,
        "gr3d_peak_pct": math.nan,
    }

    if not os.path.exists(log_path):
        return out

    vdd_inst = []
    vdd_avg_field = []
    ram_used = []
    temps = []
    gr3d = []

    re_vdd = re.compile(r"VDD_IN\s+(\d+)mW/(\d+)mW")
    re_ram = re.compile(r"RAM\s+(\d+)/(\d+)MB")
    re_temp = re.compile(r"([A-Za-z0-9_]+)@([0-9.]+)C")
    re_gr3d = re.compile(r"GR3D_FREQ\s+([0-9]+)%")

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = re_vdd.search(line)
            if m:
                vdd_inst.append(int(m.group(1)) / 1000.0)
                vdd_avg_field.append(int(m.group(2)) / 1000.0)

            m = re_ram.search(line)
            if m:
                ram_used.append(float(m.group(1)))

            for _, t in re_temp.findall(line):
                try:
                    temps.append(float(t))
                except Exception:
                    pass

            m = re_gr3d.search(line)
            if m:
                gr3d.append(float(m.group(1)))

    out["tegrastats_samples"] = len(vdd_inst)
    if vdd_inst:
        out["vdd_in_mean_w"] = float(stats.mean(vdd_inst))
        out["vdd_in_min_w"] = float(min(vdd_inst))
        out["vdd_in_max_w"] = float(max(vdd_inst))
        out["vdd_in_last_avg_field_w"] = float(vdd_avg_field[-1])
    if ram_used:
        out["board_ram_mean_mb"] = float(stats.mean(ram_used))
        out["board_ram_peak_mb"] = float(max(ram_used))
    if temps:
        out["peak_temp_c"] = float(max(temps))
    if gr3d:
        out["gr3d_mean_pct"] = float(stats.mean(gr3d))
        out["gr3d_peak_pct"] = float(max(gr3d))
    return out


# ---------------------------------------------------------------------
# Inference and metrics
# ---------------------------------------------------------------------

def rel_l2_per_sample(pred, target):
    pred = pred.float()
    target = target.float()
    return ((pred - target).flatten(1).norm(dim=1) /
            (target.flatten(1).norm(dim=1) + 1e-8))


def percentile(xs, q):
    if not xs:
        return math.nan
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


@torch.inference_mode()
def one_forward(model, graph, feats_cpu, Y_cpu, idx, precision_cfg):
    feats, U, ei, ew, lips, Y = graph.batch(feats_cpu, Y_cpu, [idx])

    if precision_cfg["model_dtype"] == torch.float16:
        feats = feats.half()
        U = U.half()
        ew = ew.half()
        lips = lips.half()
        # Y remains FP32 for metric; model output will be converted back.

    with autocast_context(precision_cfg):
        pred = model(feats, U, ei, ew, lips)

    return pred.float(), Y.float()


def prestage_request(graph, feats_cpu, Y_cpu, idx, precision_cfg):
    """Stage one batch-size-one request on the device, outside any timed region.

    Unified cross-family protocol: graph assembly, host-to-device transfer and
    eigenbasis materialisation are deployment-time staging, not part of the
    executed request path, and are therefore hoisted out of the timed window so
    that the Sp2GNO timed region matches the FNO/DeepONet/WNO one (a preloaded
    device-resident request tensor through the forward pass).
    """
    feats, U, ei, ew, lips, Y = graph.batch(feats_cpu, Y_cpu, [idx])
    if precision_cfg["model_dtype"] == torch.float16:
        feats = feats.half()
        U = U.half()
        ew = ew.half()
        lips = lips.half()
    return feats, U, ei, ew, lips, Y


@torch.inference_mode()
def forward_staged(model, staged, precision_cfg):
    feats, U, ei, ew, lips, Y = staged
    with autocast_context(precision_cfg):
        pred = model(feats, U, ei, ew, lips)
    return pred.float(), Y.float()


def run_benchmark(args):
    if torch.cuda.is_available() is False:
        raise RuntimeError("CUDA is not available on this Jetson environment.")

    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "").strip():
        raise RuntimeError(
            "PYTORCH_NO_CUDA_MEMORY_CACHING is set. "
            "Unset it for the admitted cache-enabled deployment run."
        )

    set_all_seeds(args.seed)
    device = "cuda"

    precision_cfg = configure_precision(args.precision)

    if args.dataset == "burgers":
        model, graph, test_feats, test_Y, meta = build_burgers(args, device)
    elif args.dataset == "darcy":
        model, graph, test_feats, test_Y, meta = build_darcy(args, device)
    else:
        raise ValueError(args.dataset)

    model = cast_model_for_precision(model, precision_cfg)
    model.eval()
    parameter_count = int(sum(p.numel() for p in model.parameters()))

    run_name = args.run_name or f"{args.dataset}_{args.precision}_r{args.rep}"
    case_id = args.case_id or re.sub(r"_(fp32_strict|tf32|bf16_autocast|fp16_autocast|fp16_native)_rep[0-9]+$", "", run_name)
    run_dir = os.path.join(args.suite_root, run_name)
    logs_dir = os.path.join(run_dir, "logs")
    reports_dir = os.path.join(run_dir, "reports")
    outputs_dir = os.path.join(run_dir, "outputs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    tegra_log = os.path.join(logs_dir, f"tegrastats_{run_name}.log")
    status_path = os.path.join(logs_dir, f"run_status_{run_name}.txt")
    provenance_path = os.path.join(run_dir, "provenance.txt")

    num_test = int(test_feats.shape[0])

    # Warmup is excluded from timing/telemetry.
    for w in range(args.warmup):
        idx = w % num_test
        _ = one_forward(model, graph, test_feats, test_Y, idx, precision_cfg)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    lat_ms = []
    first_pass_preds = [None] * num_test
    first_pass_errs = [None] * num_test
    n_inf = 0
    cycle = 0

    t_start = time.perf_counter()
    monitor_enabled = not args.no_tegrastats and args.timing_class != "short_run"

    if args.unified_protocol:
        # ---- Unified cross-family protocol (Fig. 2 / Tables 5 and 9) ----
        # Untimed validity pass first, so the decoded-space gate is unaffected.
        n_val = num_test if args.validity_samples <= 0 else min(num_test, args.validity_samples)
        for i in range(n_val):
            pred, Y = one_forward(model, graph, test_feats, test_Y, i, precision_cfg)
            first_pass_preds[i] = pred.cpu()
            first_pass_errs[i] = rel_l2_per_sample(pred, Y).cpu()
        if n_val < num_test:
            first_pass_preds = first_pass_preds[:n_val]
            first_pass_errs = first_pass_errs[:n_val]
            test_Y = test_Y[:n_val]
            num_test = n_val

        staged = prestage_request(
            graph, test_feats, test_Y, args.sample_index % num_test, precision_cfg
        )

        # Time-based warmup: an iteration-count warmup gives a 0.26 s warmup on
        # Burgers and a 15 s warmup on Darcy 421, i.e. a different clock/thermal
        # state per workload. Seconds make the entry state comparable.
        torch.cuda.synchronize()
        _w = time.perf_counter()
        while time.perf_counter() - _w < args.warmup_seconds:
            _ = forward_staged(model, staged, precision_cfg)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        def _timed_loop():
            nonlocal n_inf
            t0w = time.perf_counter()
            while time.perf_counter() - t0w < args.min_duration_s:
                a = time.perf_counter()
                _ = forward_staged(model, staged, precision_cfg)
                torch.cuda.synchronize()
                b = time.perf_counter()
                lat_ms.append((b - a) * 1000.0)
                n_inf += 1
            return t0w

        if args.timing_class == "short_run":
            t_start = time.perf_counter()
            for _ in range(args.num_iters):
                a = time.perf_counter()
                _ = forward_staged(model, staged, precision_cfg)
                torch.cuda.synchronize()
                b = time.perf_counter()
                lat_ms.append((b - a) * 1000.0)
                n_inf += 1
        else:
            with TegraMonitor(tegra_log, interval_ms=args.tegrastats_interval_ms,
                              enabled=not args.no_tegrastats):
                t_start = _timed_loop()
        cycle = 1

    elif args.timing_class == "short_run":
        # Short-run timing class: fixed-iteration window on a single request,
        # no concurrent telemetry, matching the FNO/DeepONet/WNO short-run
        # protocol (30 warmup / 100 timed iterations, batch size one) so that
        # cross-family comparisons stay inside a single measurement class.
        #
        # The decoded-space predictions used for the validity gate are still
        # produced, but on an untimed pass so they cannot perturb the window.
        n_val = num_test if args.validity_samples <= 0 else min(num_test, args.validity_samples)
        for i in range(n_val):
            pred, Y = one_forward(model, graph, test_feats, test_Y, i, precision_cfg)
            first_pass_preds[i] = pred.cpu()
            first_pass_errs[i] = rel_l2_per_sample(pred, Y).cpu()
        if n_val < num_test:
            # Timing, not predictive validity, is the object of the short-run
            # class; the gate itself is reported from the sustained runs.
            first_pass_preds = first_pass_preds[:n_val]
            first_pass_errs = first_pass_errs[:n_val]
            test_Y = test_Y[:n_val]
            num_test = n_val
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        idx = args.sample_index % num_test
        t_start = time.perf_counter()
        for _ in range(args.num_iters):
            t0 = time.perf_counter()
            _pred, _Y = one_forward(model, graph, test_feats, test_Y, idx, precision_cfg)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat_ms.append((t1 - t0) * 1000.0)
            n_inf += 1
        cycle = 1
    else:
        with TegraMonitor(tegra_log, interval_ms=args.tegrastats_interval_ms, enabled=monitor_enabled):
            while True:
                for i in range(num_test):
                    t0 = time.perf_counter()
                    pred, Y = one_forward(model, graph, test_feats, test_Y, i, precision_cfg)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    lat_ms.append((t1 - t0) * 1000.0)

                    if cycle == 0:
                        first_pass_preds[i] = pred.cpu()
                        first_pass_errs[i] = rel_l2_per_sample(pred, Y).cpu()

                    n_inf += 1

                cycle += 1
                elapsed = time.perf_counter() - t_start
                if elapsed >= args.min_duration_s and cycle >= args.min_cycles:
                    break
                if args.max_cycles > 0 and cycle >= args.max_cycles:
                    break

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    timed_elapsed_s = t_end - t_start

    preds = torch.cat(first_pass_preds, dim=0).numpy()
    targets = test_Y.numpy()
    rel_l2_vals = torch.cat(first_pass_errs, dim=0).numpy()
    test_rel_l2 = float(rel_l2_vals.mean())

    if args.save_outputs:
        np.save(os.path.join(outputs_dir, "predictions.npy"), preds)
        np.save(os.path.join(outputs_dir, "targets.npy"), targets)
        np.save(os.path.join(outputs_dir, "rel_l2_per_sample.npy"), rel_l2_vals)

    cuda_peak_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    cuda_peak_reserved_mb = float(torch.cuda.max_memory_reserved() / (1024 ** 2))

    power = parse_tegrastats(tegra_log)

    throughput = float(n_inf / timed_elapsed_s) if timed_elapsed_s > 0 else math.nan
    avg_power_w = power["vdd_in_mean_w"]
    energy_j = float(avg_power_w / throughput) if (avg_power_w == avg_power_w and throughput > 0) else math.nan

    perturb_vs_fp32 = math.nan
    if args.fp32_ref_predictions and os.path.exists(args.fp32_ref_predictions):
        ref = np.load(args.fp32_ref_predictions)
        if ref.shape == preds.shape:
            perturb_vs_fp32 = float(
                np.linalg.norm((preds - ref).reshape(preds.shape[0], -1), axis=1).mean()
                / (np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1).mean() + 1e-12)
            )

    row = {
        "run_name": run_name,
        "case_id": case_id,
        "run_dir": run_dir,
        "status": "success",
        "dataset": args.dataset,
        "precision": args.precision,
        "rep": args.rep,
        "runtime_path": "original_sp2gno_eager_cached_graph_no_dtype_patch",
        "timing_boundary": "preloaded_batch_size_one_request_tensor_plus_cached_graph",
        "timing_class": args.timing_class,
        "protocol": "unified_prestaged" if args.unified_protocol else "family_native",
        "num_iters_requested": args.num_iters if args.timing_class == "short_run" else None,
        "parameter_count": parameter_count,
        "precision_info": json.dumps(precision_info(args.precision, precision_cfg)),
        "num_test_samples": num_test,
        "num_timed_inferences": n_inf,
        "cycles": cycle,
        "timed_elapsed_s": timed_elapsed_s,
        "throughput_inf_s": throughput,
        "mean_latency_ms": float(stats.mean(lat_ms)),
        "p50_latency_ms": percentile(lat_ms, 50),
        "p95_latency_ms": percentile(lat_ms, 95),
        "p99_latency_ms": percentile(lat_ms, 99),
        "min_latency_ms": float(min(lat_ms)),
        "max_latency_ms": float(max(lat_ms)),
        "test_rel_l2": test_rel_l2,
        "perturb_rel_l2_vs_fp32": perturb_vs_fp32,
        "cuda_peak_allocated_mb": cuda_peak_alloc_mb,
        "cuda_peak_reserved_mb": cuda_peak_reserved_mb,
        "energy_j_per_inference": energy_j,
        **power,
        **meta,
        "width": args.width,
        "n_layers": args.n_layers,
        "num_freq": args.num_freq,
        "k": args.k,
        "batch_size": 1,
        "warmup": args.warmup,
        "min_duration_s": args.min_duration_s,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "pytorch_no_cuda_memory_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "UNSET"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "UNSET"),
    }

    summary_csv = os.path.join(reports_dir, f"sp2gno_edge_summary_{run_name}.csv")
    power_csv = os.path.join(reports_dir, f"jetson_power_summary_{run_name}.csv")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with open(power_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(power.keys()))
        writer.writeheader()
        writer.writerow(power)

    with open(status_path, "w") as f:
        f.write("success\n")

    with open(provenance_path, "w") as f:
        f.write(f"cmd={' '.join(shlex.quote(x) for x in sys.argv)}\n")
        f.write(f"timestamp={_dt.datetime.now().isoformat()}\n")
        f.write(f"hostname={platform.node()}\n")
        f.write(f"platform={platform.platform()}\n")
        f.write(f"python={sys.version}\n")
        for k in [
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
            "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_VISIBLE_DEVICES",
        ]:
            f.write(f"{k}={os.environ.get(k, 'UNSET')}\n")
        for k, v in row.items():
            f.write(f"{k}={v}\n")

    print(f"[OK] {run_name}")
    print(f"  dataset={args.dataset} precision={args.precision}")
    print(f"  relL2={test_rel_l2:.6g}")
    print(f"  p50={row['p50_latency_ms']:.3f} ms p95={row['p95_latency_ms']:.3f} ms mean={row['mean_latency_ms']:.3f} ms")
    print(f"  avgW={avg_power_w:.3f} J/inf={energy_j:.6g}")
    print(f"  CUDA alloc peak={cuda_peak_alloc_mb:.2f} MB, board RAM peak={power['board_ram_peak_mb']:.0f} MB")
    print(f"  summary={summary_csv}")

    return row


def aggregate_suite(root):
    files = sorted(glob.glob(os.path.join(root, "**", "reports", "sp2gno_edge_summary_*.csv"), recursive=True))
    rows = []
    for f in files:
        try:
            with open(f, newline="") as fh:
                r = next(csv.DictReader(fh))
                r["_summary_file"] = f
                rows.append(r)
        except Exception:
            pass

    # Attach perturb vs first FP32 prediction if outputs exist and perturb missing.
    fp32_refs = {}
    for r in rows:
        if r.get("precision") == "fp32_strict" and r.get("status") == "success":
            pred = os.path.join(r["run_dir"], "outputs", "predictions.npy")
            if os.path.exists(pred) and r["dataset"] not in fp32_refs:
                fp32_refs[r["dataset"]] = pred

    for r in rows:
        try:
            p = float(r.get("perturb_rel_l2_vs_fp32", "nan"))
        except Exception:
            p = math.nan
        if p == p:
            continue
        ref_path = fp32_refs.get(r.get("dataset"))
        pred_path = os.path.join(r["run_dir"], "outputs", "predictions.npy")
        if ref_path and os.path.exists(pred_path):
            ref = np.load(ref_path)
            pred = np.load(pred_path)
            if ref.shape == pred.shape:
                per = np.linalg.norm((pred - ref).reshape(pred.shape[0], -1), axis=1) / (
                    np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1) + 1e-12
                )
                r["perturb_rel_l2_vs_fp32"] = float(per.mean())

    raw_csv = os.path.join(root, "sp2gno_suite_raw.csv")
    if rows:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(raw_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return math.nan

    metrics = [
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "test_rel_l2",
        "perturb_rel_l2_vs_fp32",
        "vdd_in_mean_w",
        "energy_j_per_inference",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "board_ram_mean_mb",
        "board_ram_peak_mb",
        "peak_temp_c",
        "gr3d_mean_pct",
        "throughput_inf_s",
    ]

    groups = {}
    for r in rows:
        if r.get("status") != "success":
            continue
        key = (r.get("dataset"), r.get("precision"))
        groups.setdefault(key, []).append(r)

    summary = []
    for (dataset, precision), rs in sorted(groups.items()):
        out = {
            "dataset": dataset,
            "precision": precision,
            "n_runs": len(rs),
            "run_names": ";".join(r.get("run_name", "") for r in rs),
        }
        for m in metrics:
            vals = [fnum(r.get(m)) for r in rs]
            vals = [v for v in vals if v == v]
            if vals:
                out[m + "_mean"] = float(stats.mean(vals))
                out[m + "_min"] = float(min(vals))
                out[m + "_max"] = float(max(vals))
            else:
                out[m + "_mean"] = math.nan
                out[m + "_min"] = math.nan
                out[m + "_max"] = math.nan
        summary.append(out)

    sum_csv = os.path.join(root, "sp2gno_suite_summary.csv")
    if summary:
        keys = list(summary[0].keys())
        with open(sum_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)

    print(f"[AGG] raw={raw_csv}")
    print(f"[AGG] summary={sum_csv}")
    for r in summary:
        print(
            f"{r['dataset']:7s} {r['precision']:15s} "
            f"p50={r['p50_latency_ms_mean']:.3f} ms "
            f"p95={r['p95_latency_ms_mean']:.3f} ms "
            f"W={r['vdd_in_mean_w_mean']:.3f} "
            f"J={r['energy_j_per_inference_mean']:.6f} "
            f"CUDA={r['cuda_peak_allocated_mb_mean']:.2f} MB "
            f"relL2={r['test_rel_l2_mean']:.6g} "
            f"pert={r['perturb_rel_l2_vs_fp32_mean']:.6g}"
        )


def main():
    ap = argparse.ArgumentParser()

    # aggregation mode
    ap.add_argument("--aggregate_root", default=None)

    # dataset/model
    ap.add_argument("--dataset", choices=["burgers", "darcy"], default="darcy")
    ap.add_argument("--data_dir", default="Jetson_data")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--burgers_split", default="Jetson_data/burgers_split.json")

    # architecture
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--num_freq", type=int, default=64)
    ap.add_argument("--k", type=int, default=None)

    # data subsampling
    ap.add_argument("--sub", type=int, default=8)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--res", type=int, default=None,
                     help="Darcy target resolution (evenly-spaced round(linspace(0,420,res)) indices); "
                          "overrides --r. Required for resolutions with non-uniform native-grid spacing (e.g. 281).")
    ap.add_argument("--ntrain", type=int, default=900)
    ap.add_argument("--nval", type=int, default=100)
    ap.add_argument("--ntest", type=int, default=200)

    # benchmark
    ap.add_argument("--precision", choices=[
        "fp32_strict", "tf32", "fp16_autocast", "bf16_autocast", "fp16_native"
    ], default="fp32_strict")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--timing_class",
        choices=["sustained", "short_run"],
        default="sustained",
        help="sustained = duration-bounded window with tegrastats; "
             "short_run = fixed-iteration window, no telemetry (FNO/DeepONet protocol)",
    )
    ap.add_argument("--num_iters", type=int, default=100)
    ap.add_argument("--unified_protocol", type=int, default=0,
                    help="stage the request outside the timed window, use a time-based "
                         "warmup and an exact duration window (cross-family unified protocol)")
    ap.add_argument("--warmup_seconds", type=float, default=20.0,
                    help="unified_protocol only: time-based warmup length")
    ap.add_argument("--validity_samples", type=int, default=0,
                    help="short_run only: cap the untimed validity pass (0 = all test samples)")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--min_duration_s", type=float, default=30.0)
    ap.add_argument("--min_cycles", type=int, default=1)
    ap.add_argument("--max_cycles", type=int, default=0)
    ap.add_argument("--tegrastats_interval_ms", type=int, default=200)
    ap.add_argument("--no_tegrastats", action="store_true")
    ap.add_argument("--save_outputs", action="store_true")
    ap.add_argument("--fp32_ref_predictions", default=None)

    # provenance
    ap.add_argument("--suite_root", default=None)
    ap.add_argument("--case_id", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.aggregate_root:
        aggregate_suite(args.aggregate_root)
        return

    if args.k is None:
        args.k = 8 if args.dataset == "burgers" else 20

    if args.suite_root is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.suite_root = os.path.join("inference_runs", f"sp2gno_jetson_{ts}")

    try:
        run_benchmark(args)
    except Exception as e:
        # best-effort failure record
        run_name = args.run_name or f"{args.dataset}_{args.precision}_r{args.rep}"
        run_dir = os.path.join(args.suite_root, run_name)
        logs_dir = os.path.join(run_dir, "logs")
        reports_dir = os.path.join(run_dir, "reports")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        err_txt = os.path.join(logs_dir, f"error_{run_name}.txt")
        with open(err_txt, "w") as f:
            f.write("FAILED\n")
            f.write(str(e) + "\n\n")
            f.write(traceback.format_exc())

        fail_csv = os.path.join(reports_dir, f"sp2gno_edge_summary_{run_name}_FAILED.csv")
        row = {
            "run_name": run_name,
            "case_id": args.case_id or re.sub(r"_(fp32_strict|tf32|bf16_autocast|fp16_autocast|fp16_native)_rep[0-9]+$", "", run_name),
            "run_dir": run_dir,
            "status": "failed",
            "dataset": args.dataset,
            "precision": args.precision,
            "rep": args.rep,
            "runtime_path": "original_sp2gno_eager_cached_graph_no_dtype_patch",
            "timing_boundary": "preloaded_batch_size_one_request_tensor_plus_cached_graph",
            "width": args.width,
            "n_layers": args.n_layers,
            "num_freq": args.num_freq,
            "k": args.k,
            "error": str(e),
        }
        with open(fail_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        print(f"[FAILED] {run_name}: {e}")
        print(f"  error={err_txt}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
