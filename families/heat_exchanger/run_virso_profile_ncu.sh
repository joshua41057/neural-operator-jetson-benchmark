#!/usr/bin/env bash
# Run with sudo (ncu requires root on this Jetson):
#   sudo bash run_virso_profile_ncu.sh
#
# Two attempts before this one both hit nvmap resource exhaustion
# (NvMapMemAllocInternalTagged .../ NvMapMemHandleAlloc error) under
# --replay-mode kernel. Root cause, confirmed by reading the harness:
# run_virso_inference_jetson.py's timed loop (line ~790,
# `for idx, el in enumerate(tqdm(test_dataset, ...))`) always iterates the
# FULL 310-sample test set -- PROFILE_WARMUP/PROFILE_ITERS only gate a
# separate FLOPS-estimation code path, not this loop. So ncu's kernel-replay
# mode, which snapshots/restores each targeted kernel's device memory to
# system RAM on every one of its ~18 detailed-set passes (the exact overhead
# its own warning flags: "Backing up device memory in system memory...
# Consider using --replay-mode application"), was being asked to do that
# across a run generating thousands of kernel launches, not the single
# forward pass the sp2gno reference script gets via its own --rep 1
# --min_duration_s 0.001 flags. Two fixes, both taken from mechanisms
# already present in this exact codebase:
#   1. MEM_AUDIT_ONLY=1 / AUDIT_MAX_SAMPLES=3 -- the same slicing the
#      repo's own successful allocaudit_* single-sample memory probes use
#      (see run_virso_inference_jetson.py:586-591) -- bounds the run to a
#      few samples instead of all 310.
#   2. --replay-mode application instead of kernel -- avoids the per-kernel
#      device-memory backup/restore entirely (re-runs the whole -- now short
#      -- app per pass-group instead), per ncu's own suggestion.
# VIRSO_ALLOCATOR_AUDIT=1 (the allocator-cache fix needed for latency/energy
# correctness) is deliberately NOT set here: ncu's roofline metrics (compute/
# memory throughput %, occupancy) are properties of the kernel's own launch
# config and memory access pattern, invariant to whether the surrounding
# allocator caches blocks between calls -- and the uncached path is what the
# reference sp2gno ncu script itself runs under, and is what survived further
# (~100+ kernels) before the first crash here.
#
# Kernel-name regex is unchanged from
# /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026/scripts_profile/run_sp2gno_profile_ncu.sh
# -- nsys confirms VirSO's Sp2GNO forward pass launches the identical kernel
# set (CatArrayBatchedCopy, indexSelectLargeIndex, scatter_gather_elementwise_kernel,
# vectorized_layer_norm_kernel, ampere_sgemm*, gemvx), since it's the same
# model code (sp2gno_model.py).
set -euo pipefail

export HOME=/home/jetson
ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
cd "$ROOT" || exit 1

PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"
NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
OUTDIR="$ROOT/inference_runs/virso_ncu"
mkdir -p "${OUTDIR}"

run_ncu () {
  local tag="$1"
  shift

  if [[ -s "${OUTDIR}/${tag}.ncu-rep" ]]; then
    echo "[SKIP] ${tag} (already exists)"
    return 0
  fi

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  "${NCU_BIN}" \
    --force-overwrite \
    --target-processes all \
    --replay-mode application \
    --cache-control all \
    --clock-control none \
    --set detailed \
    --kernel-name "regex:CatArrayBatchedCopy|indexSelectLargeIndex|scatter_gather_elementwise_kernel|vectorized_layer_norm_kernel|ampere_sgemm|gemvx" \
    --launch-count 150 \
    --export "${OUTDIR}/${tag}" \
    --log-file "${OUTDIR}/${tag}.log" \
    "$@"

  if [[ ! -f "${OUTDIR}/${tag}.ncu-rep" ]]; then
    echo "[FAIL] ${tag} -- missing report, tail of log:"
    tail -60 "${OUTDIR}/${tag}.log" || true
  fi
}

run_case () {
  local tag="$1"
  local model_path="$2"
  local num_layers="$3"
  local width="$4"
  local max_mode="$5"
  local spectral="$6"
  local spatial="$7"

  MODEL_PATH="$model_path" \
  AMP_MODE="off" \
  AMP_FALLBACK="fp16" \
  TF32_MODE="strict" \
  NUM_LAYERS="$num_layers" \
  WIDTH="$width" \
  MAX_MODE="$max_mode" \
  K_NEIGHBORS="30" \
  EMBED="1" \
  SPECTRAL="$spectral" \
  SPATIAL="$spatial" \
  COLLAB_SKIP="1" \
  SPECTRAL_SKIP="1" \
  MONITOR_CMD="none" \
  PROFILE_FLOPS="0" \
  PROFILE_WARMUP="0" \
  PROFILE_ITERS="1" \
  MEM_AUDIT_ONLY="1" \
  AUDIT_MAX_SAMPLES="3" \
  run_ncu "${tag}" "${PY}" "${SCRIPT}"
}

# Three Sp2GNO Heat Exchanger variants, FP32 strict -- same case set as the
# nsys pass (full/spectral/layer2), matching the paper's per-family ncu
# coverage (one representative fp32 case per architectural configuration).
run_case full_fp32     "$ROOT/sp2gno_final.pth"                10 48 64 1 1
run_case spectral_fp32 "$HOME/VirSO/best_model_spectral.pth"   10 48 64 1 0
run_case layer2_fp32   "$HOME/VirSO/best_model_2_layer.pth"     2 48 40 1 1

echo "[DONE] VirSO Sp2GNO ncu profiling (3 cases)"
echo "Now run: sudo chown -R jetson:jetson ${OUTDIR}"
