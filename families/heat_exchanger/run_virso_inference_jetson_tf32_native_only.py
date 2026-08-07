import os

os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import sys
import csv
import time
import signal
import shutil
import random
import logging
import traceback
import subprocess
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sp2gno_model import GraphFNO
from utilities import GaussianNormalizer, RangeNormalizer, LpLoss
from data import GraphDataset
from torch.profiler import profile, ProfilerActivity


# ============================================================
# Basic paths
# ============================================================
ROOT = "/home/jetson/VirSO/For_Jetson/For_Jetson"
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_MODEL_PATH = os.path.join(ROOT, "sp2gno_final.pth")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
RESULTS_BASE = os.path.join(ROOT, "inference_runs")

RUN_TS = os.environ.get("RUN_TS", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.environ.get("RUN_DIR", os.path.join(RESULTS_BASE, RUN_TS))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(RUN_DIR, "outputs"))
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(RUN_DIR, "logs"))
REPORT_DIR = os.environ.get("REPORT_DIR", os.path.join(RUN_DIR, "reports"))
SUMMARY_CSV = os.environ.get("EDGE_CSV", os.path.join(REPORT_DIR, f"virso_edge_summary_{RUN_TS}.csv"))
PROFILE_FLOPS = os.environ.get("PROFILE_FLOPS", "1") == "1"
PROFILE_WARMUP = int(os.environ.get("PROFILE_WARMUP", "2"))
PROFILE_ITERS = int(os.environ.get("PROFILE_ITERS", "1"))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

if ROOT not in sys.path:
    sys.path.append(ROOT)


# ============================================================
# Logging
# ============================================================
LOG_PATH = os.path.join(LOG_DIR, f"virso_inference_{RUN_TS}.log")

logger = logging.getLogger("virso_inference")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def log(msg):
    logger.info(msg)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def write_summary_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        writer.writerows(rows)


def log_gpu_mem(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        log(
            f"{prefix} GPU mem allocated={allocated:.2f} MB "
            f"reserved={reserved:.2f} MB peak_alloc={peak:.2f} MB"
        )


# ============================================================
# Reproducibility / threads
# ============================================================
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.set_num_threads(min(4, os.cpu_count()))
torch.set_num_interop_threads(1)


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Config (preserve original VirSO inference flow)
# ============================================================
random_state = 50
lambda_mag = 1
model_random_state = 75

model_file = MODEL_PATH

graph_type = "knn"
k = int(os.environ.get("K_NEIGHBORS", "30"))
r = float(os.environ.get("RADIUS", "0.001"))
k_lower = int(os.environ.get("K_LOWER", "45"))
start_radius = float(os.environ.get("START_RADIUS", "0.0005"))
factor_of_lower = int(os.environ.get("FACTOR_OF_LOWER", "1"))

embed = os.environ.get("EMBED", "1") == "1"
spectral = os.environ.get("SPECTRAL", "1") == "1"
spatial = os.environ.get("SPATIAL", "1") == "1"
collab_skip = os.environ.get("COLLAB_SKIP", "1") == "1"
spectral_skip = os.environ.get("SPECTRAL_SKIP", "1") == "1"

num_sp2gno_layers = int(os.environ.get("NUM_LAYERS", "10"))
width = int(os.environ.get("WIDTH", "48"))
max_mode = int(os.environ.get("MAX_MODE", "64"))
dataset_largest_mode = int(os.environ.get("DATASET_LARGEST_MODE", "150"))
output_dim = 4

# Optional monitor process
# MONITOR_CMD can be: none | tegrastats | nvidia-smi
MONITOR_CMD = os.environ.get("MONITOR_CMD", "none").strip().lower()
MONITOR_INTERVAL_MS = int(os.environ.get("MONITOR_INTERVAL_MS", "200"))
MONITOR_LOG = os.path.join(LOG_DIR, f"tegrastats_{RUN_TS}.log")

# AMP / precision control
AMP_MODE = os.environ.get("AMP_MODE", "off").strip().lower()   # off | fp16 | bf16 | native_fp16
AMP_FALLBACK = os.environ.get("AMP_FALLBACK", "fp16").strip().lower()  # off | fp16

PRECISION_MODE = os.environ.get("PRECISION_MODE", "").strip().lower()

if PRECISION_MODE:
    if PRECISION_MODE == "tf32":
        AMP_MODE = "off"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif PRECISION_MODE == "fp16_native":
        AMP_MODE = "native_fp16"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        raise RuntimeError(f"Only missing VIRSO modes are supported here: tf32 or fp16_native, got {PRECISION_MODE}")


# ============================================================
# Helpers
# ============================================================
def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def load_raw_data():
    grid_path = os.path.join(DATA_DIR, "share", "trunk.npz")
    branch_path = os.path.join(DATA_DIR, "branch.npz")
    target_path = os.path.join(DATA_DIR, "target.npy")

    require_file(grid_path)
    require_file(branch_path)
    require_file(target_path)
    require_file(model_file)

    grid_pos = torch.from_numpy(np.load(grid_path)["trunk"]).float()

    branch = np.load(branch_path)
    heat_prof = torch.from_numpy(branch["branch2"]).float()
    inlet = torch.from_numpy(branch["branch1"]).float()

    # preserve original channel selection exactly
    target = torch.from_numpy(np.load(target_path)).float()[:, :, [1, 3, 4, 5, 6]]

    return grid_pos, heat_prof, inlet, target


def split_data(inlet, heat_prof, target):
    trainval_inlet, final_test_inlet, trainval_heat, final_test_heat, trainval_target, final_test_target = train_test_split(
        inlet, heat_prof, target,
        test_size=0.2,
        random_state=random_state,
        shuffle=True
    )

    train_inlet, val_inlet, train_heat, val_heat, train_target, val_target = train_test_split(
        trainval_inlet, trainval_heat, trainval_target,
        test_size=0.2,
        random_state=random_state + 1,
        shuffle=True
    )

    return {
        "train_inlet": train_inlet,
        "val_inlet": val_inlet,
        "train_heat": train_heat,
        "val_heat": val_heat,
        "train_target": train_target,
        "val_target": val_target,
        "final_test_inlet": final_test_inlet,
        "final_test_heat": final_test_heat,
        "final_test_target": final_test_target,
    }


def build_normalizers(train_inlet, train_heat, train_target):
    inlet_normalizer = GaussianNormalizer(train_inlet, False)
    heat_normalizer = GaussianNormalizer(train_heat, True)
    output_normalizer_cpu = RangeNormalizer(train_target[:, :, :-1], True, low=-1, high=1)

    # Separate GPU copy for decode only
    output_normalizer_gpu = RangeNormalizer(train_target[:, :, :-1], True, low=-1, high=1)
    if device.type == "cuda":
        output_normalizer_gpu.cuda()

    return inlet_normalizer, heat_normalizer, output_normalizer_cpu, output_normalizer_gpu


def start_monitor():
    if MONITOR_CMD == "none":
        log("Monitor disabled")
        return None

    if MONITOR_CMD == "tegrastats":
        exe = shutil.which("tegrastats")
        if exe is None:
            log("tegrastats not found; monitor disabled")
            return None

        log(f"Starting tegrastats monitor -> {MONITOR_LOG}")
        fp = open(MONITOR_LOG, "w", encoding="utf-8")

        # Preserve the old Makefile behavior: timestamp|tegrastats_line
        cmd = (
            f"stdbuf -oL -eL {exe} --interval {MONITOR_INTERVAL_MS} 2>&1 | "
            "while IFS= read -r line; do "
            "printf \"%s|%s\\n\" \"$(date +%s.%3N)\" \"$line\"; "
            "done"
        )

        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=fp,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        proc._log_fp = fp
        return proc

    if MONITOR_CMD == "nvidia-smi":
        exe = shutil.which("nvidia-smi")
        if exe is None:
            log("nvidia-smi not found; monitor disabled")
            return None

        log(f"Starting nvidia-smi monitor -> {MONITOR_LOG}")
        fp = open(MONITOR_LOG, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                exe,
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                "--format=csv",
                "-lms", "200",
            ],
            stdout=fp,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        proc._log_fp = fp
        return proc

    log(f"Unknown MONITOR_CMD={MONITOR_CMD}; monitor disabled")
    return None


def stop_monitor(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        pass
    try:
        if hasattr(proc, "_log_fp"):
            proc._log_fp.close()
    except Exception:
        pass


def init_model(input_dim):
    # preserve original call semantics exactly
    torch.manual_seed(model_random_state)
    model = GraphFNO(
        num_sp2gno_layers,
        input_dim,
        width,
        max_mode,
        device,
        output_dim,
        16,   # lip_nodes
        2,    # domain_dim
        embed,
        collab_skip,
        spectral_skip,
        spectral,
        spatial,
    )
    return model


def load_model(model):
    log(f"Loading checkpoint: {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model

def resolve_amp_dtype():
    if device.type != "cuda":
        log("AMP disabled: CUDA not available")
        return None

    if AMP_MODE == "off":
        log("AMP disabled: AMP_MODE=off")
        return None

    if AMP_MODE == "native_fp16":
        log("Native FP16 requested: model converted to half precision without autocast")
        return None

    if AMP_MODE == "fp16":
        log("AMP enabled: float16")
        return torch.float16

    if AMP_MODE == "bf16":
        log("AMP requested: bfloat16")
        try:
            # lightweight runtime probe
            a = torch.randn(64, 64, device=device, dtype=torch.float32)
            b = torch.randn(64, 64, device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                c = a @ b
            del a, b, c
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log("AMP enabled: bfloat16 probe succeeded")
            return torch.bfloat16
        except Exception as e:
            log(f"BF16 probe failed: {repr(e)}")
            if AMP_FALLBACK == "fp16":
                log("Falling back to float16 autocast")
                return torch.float16
            log("No AMP fallback; running in FP32")
            return None

    log(f"Unknown AMP_MODE={AMP_MODE}; running in FP32")
    return None

def run_model_forward(model, el, amp_dtype):
    x = el.input.to(device, non_blocking=False)
    U = el.U.to(device, non_blocking=False)
    edge_index = el.edge_index.to(device, non_blocking=False)
    edge_weight = el.edge_weight.to(device, non_blocking=False)
    grid = el.grid_pos.to(device, non_blocking=False)
    lip = el.lipschitz_embedding.to(device, non_blocking=False)

    if amp_dtype is None:
        out = model(x, U, edge_index, edge_weight, grid, lip, 1)
    else:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(x, U, edge_index, edge_weight, grid, lip, 1)
    return out


def profile_flops_torch(model, el, amp_dtype):
    """
    Returns total FLOPs for one inference sample using torch.profiler.
    """
    if device.type == "cuda":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]

    # warmup
    with torch.inference_mode():
        for _ in range(PROFILE_WARMUP):
            out = run_model_forward(model, el, amp_dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            del out

    total_flops = 0

    with torch.inference_mode():
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            for _ in range(PROFILE_ITERS):
                out = run_model_forward(model, el, amp_dtype)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                del out

    # Sum FLOPs across all recorded events
    for evt in prof.key_averages():
        evt_flops = getattr(evt, "flops", 0)
        if evt_flops is not None:
            total_flops += evt_flops

    # average per single inference
    if PROFILE_ITERS > 0:
        total_flops = total_flops / PROFILE_ITERS

    return float(total_flops)


def estimate_flops_analytic(num_nodes, num_edges, width, max_mode, num_layers, spectral, spatial):
    """
    Professor-provided asymptotic estimate:
      spectral: O(m * n * d_v^2) per layer
      spatial : O(|E| * d_v^2) per layer

    We use:
      m = max_mode
      n = num_nodes
      d_v = width
      |E| = edge count in edge_index
    """
    dv2 = width * width
    per_layer = 0

    if spectral:
        per_layer += max_mode * num_nodes * dv2

    if spatial:
        per_layer += num_edges * dv2

    total = num_layers * per_layer
    return float(total)

# ============================================================
# Main
# ============================================================
def main():
    log("=" * 80)
    log("VirSO Jetson inference start")
    log("=" * 80)
    log(f"ROOT: {ROOT}")
    log(f"RUN_TS: {RUN_TS}")
    log(f"RUN_DIR: {RUN_DIR}")
    log(f"LOG_PATH: {LOG_PATH}")
    log(f"torch version: {torch.__version__}")
    log(f"cuda available: {torch.cuda.is_available()}")
    log(f"device: {device}")
    if torch.cuda.is_available():
        log(f"gpu name: {torch.cuda.get_device_name(0)}")
        log(f"cuda version: {torch.version.cuda}")
    log(f"MONITOR_CMD: {MONITOR_CMD}")

    log(f"AMP_MODE: {AMP_MODE}")
    log(f"AMP_FALLBACK: {AMP_FALLBACK}")

    log("=" * 80)
    log("Loading raw data")
    log("=" * 80)

    grid_pos, heat_prof, inlet, target = load_raw_data()
    num_nodes = grid_pos.shape[0]
    input_dim = heat_prof.shape[1] + inlet.shape[1]

    log(f"grid_pos shape : {tuple(grid_pos.shape)} dtype={grid_pos.dtype}")
    log(f"heat_prof shape: {tuple(heat_prof.shape)} dtype={heat_prof.dtype}")
    log(f"inlet shape    : {tuple(inlet.shape)} dtype={inlet.dtype}")
    log(f"target shape   : {tuple(target.shape)} dtype={target.dtype}")
    log(f"input_dim={input_dim}, output_dim={output_dim}, num_nodes={num_nodes}")

    splits = split_data(inlet, heat_prof, target)

    log("=" * 80)
    log("Data split")
    log("=" * 80)
    log(f"train_target shape     : {tuple(splits['train_target'].shape)}")
    log(f"val_target shape       : {tuple(splits['val_target'].shape)}")
    log(f"final_test_target shape: {tuple(splits['final_test_target'].shape)}")

    # IMPORTANT:
    # Keep CPU normalizers for GraphDataset creation, otherwise Jetson device mismatch occurs.
    inlet_normalizer, heat_normalizer, output_normalizer_cpu, output_normalizer_gpu = build_normalizers(
        splits["train_inlet"],
        splits["train_heat"],
        splits["train_target"],
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.join(ROOT, "results")
    os.makedirs(base_folder, exist_ok=True)

    log("=" * 80)
    log("Building test dataset (same flow as original)")
    log("=" * 80)

    t_dataset0 = time.time()

    # SAME FLOW as original code:
    # GraphDataset over full final_test_target, repeated_grid=True by default.
    test_dataset = GraphDataset(
        splits["final_test_target"],
        grid_pos,
        splits["final_test_heat"],
        splits["final_test_inlet"],
        inlet_normalizer,
        heat_normalizer,
        output_normalizer_cpu,
        False,
        base_folder,
        timestamp,
        largest_mode=dataset_largest_mode,   # preserve original script
        custom_name="delete",
        custom_data_name="final",
        graph_type=graph_type,
        k=k,
        k_lower=k_lower,
        r=r,
        factor_of_lower=factor_of_lower,
        start_radius=start_radius,
    )

    t_dataset1 = time.time()

    log(f"dataset len           : {len(test_dataset)}")
    log(f"dataset build time (s): {t_dataset1 - t_dataset0:.3f}")

    sample0 = test_dataset[0]
    log(f"sample input shape    : {tuple(sample0.input.shape)}")
    log(f"sample output shape   : {tuple(sample0.output.shape)}")
    log(f"sample U shape        : {tuple(sample0.U.shape)}")
    log(f"sample edge_index     : {tuple(sample0.edge_index.shape)}")
    log(f"sample edge_weight    : {tuple(sample0.edge_weight.shape)}")
    log(f"sample grid_pos       : {tuple(sample0.grid_pos.shape)}")
    log(f"sample lip shape      : {tuple(sample0.lipschitz_embedding.shape)}")

    log("=" * 80)
    log("Model init / load")
    log("=" * 80)

    model = init_model(input_dim=input_dim)
    model = load_model(model)

    if AMP_MODE == "native_fp16":
        log("Converting model parameters/buffers to FP16 for native-FP16 feasibility test")
        model = model.half()

    amp_dtype = resolve_amp_dtype()
    log(f"Resolved AMP dtype: {amp_dtype}")
    log(f"PRECISION_MODE: {PRECISION_MODE}")
    log(f"TF32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}")
    log(f"TF32 cuDNN allowed : {torch.backends.cudnn.allow_tf32}")

    profiled_flops = None
    estimated_flops = None
    macs_per_sample = None
    num_edges = int(sample0.edge_index.shape[1])

    if PROFILE_FLOPS:
        log("=" * 80)
        log("Profiling FLOPs on single 3,977-node Heat Exchanger sample")
        log("=" * 80)

        try:
            profiled_flops = profile_flops_torch(model, sample0, amp_dtype)
            log(f"profiled FLOPs per sample: {profiled_flops:.3e}")
        except Exception as e:
            log(f"torch.profiler FLOPs profiling failed: {repr(e)}")
            profiled_flops = ""

        estimated_flops = estimate_flops_analytic(
            num_nodes=num_nodes,
            num_edges=num_edges,
            width=width,
            max_mode=max_mode,
            num_layers=num_sp2gno_layers,
            spectral=spectral,
            spatial=spatial,
        )
        log(f"analytic FLOPs estimate per sample: {estimated_flops:.3e}")

        # convention: 1 MAC ~= 2 FLOPs
        flops_for_macs = profiled_flops if profiled_flops != "" else estimated_flops
        if flops_for_macs != "":
            macs_per_sample = float(flops_for_macs) / 2.0
            log(f"MACs per sample: {macs_per_sample:.3e}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {param_count}")

    loss_fn = LpLoss(size_average=True)

    # Preserve original accumulation behavior
    total_loss = 0.0
    indiv_loss = np.array([0.0] * output_dim, dtype=np.float64)

    latencies_ms = []
    pred_list = []
    gold_list = []

    dry_pred_path = os.path.join(OUT_DIR, "virso_single_pred.npy")
    dry_gold_path = os.path.join(OUT_DIR, "virso_single_gold.npy")
    pred_all_path = os.path.join(OUT_DIR, "virso_predictions.npy")
    gold_all_path = os.path.join(OUT_DIR, "virso_targets.npy")

    monitor_proc = None
    inference_start_wall_s = None
    inference_end_wall_s = None

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        log("=" * 80)
        log("Inference")
        log("=" * 80)

        # Make sure FLOPs profiling / prior GPU work is done before power logging starts
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        monitor_proc = start_monitor()

        # Give tegrastats a brief moment to start cleanly
        if monitor_proc is not None:
            time.sleep(0.25)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        inference_start_wall_s = start_time

        with torch.inference_mode():
            for idx, el in enumerate(tqdm(test_dataset, desc="VirSO inference")):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()

                x = el.input.to(device, non_blocking=False)
                U = el.U.to(device, non_blocking=False)
                edge_index = el.edge_index.to(device, non_blocking=False)
                edge_weight = el.edge_weight.to(device, non_blocking=False)
                grid = el.grid_pos.to(device, non_blocking=False)
                lip = el.lipschitz_embedding.to(device, non_blocking=False)
                y_norm = el.output.to(device, non_blocking=False)

                if amp_dtype is None:
                    out = model(x, U, edge_index, edge_weight, grid, lip, 1)
                else:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = model(x, U, edge_index, edge_weight, grid, lip, 1)

                out_fp32 = out.float()
                y_norm_fp32 = y_norm.float()

                batch_preds = output_normalizer_gpu.decode(
                    out_fp32.view(1, num_nodes, output_dim)
                )
                batch_output = output_normalizer_gpu.decode(
                    y_norm_fp32.view(1, num_nodes, output_dim + 1)[:, :, :-1]
                )

                assert batch_output.shape[0] == 1
                assert batch_output.shape[1] == num_nodes
                assert batch_output.shape[2] == output_dim

                for dim in range(output_dim):
                    lss = loss_fn(
                        batch_preds[:, :, dim].view(1, -1),
                        batch_output[:, :, dim].view(1, -1),
                    )
                    indiv_loss[dim] += float(lss)
                    total_loss += float(lss)

                mag_loss = loss_fn(
                    torch.sum(batch_preds[:, :, 1:] ** 2, dim=2).view(1, -1),
                    (y_norm_fp32.view(1, num_nodes, output_dim + 1)[:, :, -1] ** 2).view(1, -1),
                )
                total_loss += float(lambda_mag * mag_loss)

                pred_list.append(batch_preds.detach().float().cpu())
                gold_list.append(batch_output.detach().float().cpu())

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.time()

                latency_ms = (t1 - t0) * 1000.0
                latencies_ms.append(latency_ms)
                
                if idx == 0:
                    np.save(dry_pred_path, batch_preds.detach().float().cpu().numpy())
                    np.save(dry_gold_path, batch_output.detach().float().cpu().numpy())
                    log(f"Saved dry prediction: {dry_pred_path}")
                    log(f"Saved dry target    : {dry_gold_path}")

                del x, U, edge_index, edge_weight, grid, lip, y_norm, out, out_fp32, y_norm_fp32, batch_preds, batch_output

                if ((idx + 1) % 10 == 0) or ((idx + 1) == len(test_dataset)):
                    log(f"[{idx + 1:4d}/{len(test_dataset)}] done")
                    log_gpu_mem(prefix="Progress:")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        inference_end_wall_s = end_time

    finally:
        stop_monitor(monitor_proc)

    num_samples = len(test_dataset)
    total_time_s = end_time - start_time
    avg_latency_ms = float(np.mean(latencies_ms)) if len(latencies_ms) > 0 else None
    p50_latency_ms = float(np.percentile(latencies_ms, 50)) if len(latencies_ms) > 0 else None
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if len(latencies_ms) > 0 else None

    avg_total_loss = total_loss / num_samples
    avg_indiv_loss = indiv_loss / num_samples

    pred_shape = ""
    gold_shape = ""

    if len(pred_list) > 0:
        pred_all = torch.cat(pred_list, dim=0).numpy()
        np.save(pred_all_path, pred_all)
        pred_shape = str(pred_all.shape)
        log(f"Saved predictions: {pred_all_path}")
    else:
        pred_all = None

    if len(gold_list) > 0:
        gold_all = torch.cat(gold_list, dim=0).numpy()
        np.save(gold_all_path, gold_all)
        gold_shape = str(gold_all.shape)
        log(f"Saved targets    : {gold_all_path}")
    else:
        gold_all = None

    peak_gpu_mem_mb = ""
    if torch.cuda.is_available():
        peak_gpu_mem_mb = safe_float(torch.cuda.max_memory_allocated() / (1024 ** 2))

    log("")
    log("=" * 80)
    log("Done")
    log("=" * 80)
    log(f"Final Test MSE: (total: {avg_total_loss}), p,v_x,v_y,v_z: {avg_indiv_loss})")
    log(f"Latency (s per iteration): {total_time_s / num_samples}")
    log(f"avg latency (ms): {avg_latency_ms}")
    log(f"p50 latency (ms): {p50_latency_ms}")
    log(f"p95 latency (ms): {p95_latency_ms}")
    log(f"total inference time (s): {total_time_s:.6f}")
    log(f"peak gpu mem (MB): {peak_gpu_mem_mb}")
    if MONITOR_CMD != "none":
        log(f"monitor log: {MONITOR_LOG}")

    summary_rows = [
        ("run_ts", RUN_TS),
        ("run_dir", RUN_DIR),
        ("report_dir", REPORT_DIR),

        ("device", str(device)),
        ("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        ("torch_version", torch.__version__),
        ("cuda_version", torch.version.cuda if torch.cuda.is_available() else ""),
        ("model_checkpoint", model_file),
        ("model_checkpoint_name", os.path.basename(model_file)),
        ("model_params", param_count),
        ("num_edges", num_edges),
        ("profiled_flops_per_sample", profiled_flops),
        ("analytic_flops_per_sample", estimated_flops),
        ("macs_per_sample", macs_per_sample),
        ("graph_type", graph_type),
        ("k", k),
        ("r", r),
        ("k_lower", k_lower),
        ("start_radius", start_radius),
        ("factor_of_lower", factor_of_lower),
        ("dataset_largest_mode", dataset_largest_mode),
        ("model_max_mode", max_mode),
        ("num_nodes", num_nodes),
        ("num_samples", num_samples),
        ("input_dim", input_dim),
        ("output_dim", output_dim),

        ("num_layers", num_sp2gno_layers),
        ("width", width),
        ("max_mode", max_mode),
        ("k_neighbors", k),

        ("embed", int(embed)),
        ("spectral", int(spectral)),
        ("spatial", int(spatial)),
        ("collab_skip", int(collab_skip)),
        ("spectral_skip", int(spectral_skip)),

        ("dataset_build_time_s", safe_float(t_dataset1 - t_dataset0)),
        ("single_sample_latency_ms", safe_float(latencies_ms[0]) if len(latencies_ms) > 0 else ""),
        ("avg_latency_ms", avg_latency_ms),
        ("p50_latency_ms", p50_latency_ms),
        ("p95_latency_ms", p95_latency_ms),
        ("total_inference_time_s", safe_float(total_time_s)),
        ("inference_start_wall_s", safe_float(inference_start_wall_s) if inference_start_wall_s is not None else ""),
        ("inference_end_wall_s", safe_float(inference_end_wall_s) if inference_end_wall_s is not None else ""),
        ("inference_elapsed_wall_s", safe_float(inference_end_wall_s - inference_start_wall_s)
            if (inference_start_wall_s is not None and inference_end_wall_s is not None) else ""),
        ("latency_s_per_iteration", safe_float(total_time_s / num_samples)),
        ("avg_total_loss", safe_float(avg_total_loss)),
        ("avg_pressure_loss", safe_float(avg_indiv_loss[0])),
        ("avg_vx_loss", safe_float(avg_indiv_loss[1])),
        ("avg_vy_loss", safe_float(avg_indiv_loss[2])),
        ("avg_vz_loss", safe_float(avg_indiv_loss[3])),
        ("peak_gpu_mem_mb", peak_gpu_mem_mb),
        ("predictions_path", pred_all_path),
        ("targets_path", gold_all_path),
        ("predictions_shape", pred_shape),
        ("targets_shape", gold_shape),
        ("dry_prediction_path", dry_pred_path),
        ("dry_target_path", dry_gold_path),
        ("monitor_cmd", MONITOR_CMD),
        ("monitor_log", MONITOR_LOG if MONITOR_CMD != "none" else ""),
        ("log_file", LOG_PATH),
        ("precision_mode", PRECISION_MODE),
        ("tf32_matmul_allowed", int(torch.backends.cuda.matmul.allow_tf32)),
        ("tf32_cudnn_allowed", int(torch.backends.cudnn.allow_tf32)),
        ("amp_mode", AMP_MODE),
        ("amp_resolved_dtype", str(amp_dtype)),
    ]
    write_summary_csv(SUMMARY_CSV, summary_rows)
    log(f"Saved summary CSV: {SUMMARY_CSV}")
    log(f"Finished successfully. Full log saved to: {LOG_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error during VirSO inference run")
        err_path = os.path.join(LOG_DIR, f"virso_inference_error_{RUN_TS}.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("Fatal error during VirSO inference run\n")
            f.write(f"Exception: {repr(e)}\n\n")
            f.write(traceback.format_exc())

        print(f"\n[ERROR] Exception log saved to: {err_path}", file=sys.stderr)
        print(f"[ERROR] Main log saved to: {LOG_PATH}", file=sys.stderr)
        raise