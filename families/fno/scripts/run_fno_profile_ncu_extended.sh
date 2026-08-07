#!/usr/bin/env bash
# Run with sudo (ncu requires root on this Jetson):
#   sudo bash scripts/run_fno_profile_ncu_extended.sh
set -euo pipefail

export HOME=/home/jetson
cd /home/jetson/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

# NVMAP_IOC_GET_FD / "LaunchFailed" crashes on this Jetson are NOT about the CUDA
# caching allocator (PYTORCH_NO_CUDA_MEMORY_CACHING does nothing here -- verified:
# no code in this repo reads it, and stock PyTorch doesn't use it to gate the
# allocator either). Log evidence from two independent crashes (this FNO run and
# WNO's) both died at kernel-launch index ~480 within one ncu session -- an
# apparent nvmap/driver resource ceiling tied to *total kernel launches profiled*,
# not kernel type. Fix: keep each ncu session to exactly one forward pass
# (--num-warmup 0 --num-iters 1 below), well under that ceiling.

PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
OUTDIR="results/jetson_fno_profile_ncu_extended"
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
    --replay-mode kernel \
    --cache-control all \
    --clock-control none \
    --set detailed \
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
  local ckpt="$2"
  local ts="$3"
  local bank="$4"

  run_ncu "${tag}" \
    "${PYBIN}" -m src.eval.profile_inference \
      --mode torchscript \
      --checkpoint "${ckpt}" \
      --torchscript "${ts}" \
      --input-bank "${bank}" \
      --precision fp32 \
      --batch-size 1 \
      --num-warmup 0 \
      --num-iters 1 \
      --device cuda \
      --results-dir "${OUTDIR}" \
      --result-tag "${tag}" \
      --notes "ncu resolution-coverage extension"
}

# Darcy: r141/r211 already succeeded (skip-logic above will skip them if present).
run_case "darcy_r141_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r141_seed0_best.pt \
  artifacts/torchscript/darcy_fno_base_r141_seed0.ts \
  artifacts/benchmark_inputs/darcy_r141_bank.pt

run_case "darcy_r211_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r211_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r211_seed1.ts \
  artifacts/benchmark_inputs/darcy_r211_bank.pt

# Clean base (non-frontier) high-resolution Darcy point.
run_case "darcy_r421_base_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r421_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r421_seed1.ts \
  artifacts/benchmark_inputs/darcy_r421_bank.pt

# Burgers: zero prior ncu kernel-level coverage.
run_case "burgers_r2048_ts_fp32" \
  artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
  artifacts/torchscript/burgers_fno_base_seed3.ts \
  artifacts/benchmark_inputs/burgers_r2048_bank.pt

run_case "burgers_r8192_ts_fp32" \
  artifacts/checkpoints/burgers_fno_base_r8192_seed1_best.pt \
  artifacts/torchscript/burgers_fno_base_r8192_seed1.ts \
  artifacts/benchmark_inputs/burgers_r8192_bank.pt

echo "[DONE] FNO ncu profiling (5 target cases)"
echo "Now run: sudo chown -R jetson:jetson results/jetson_fno_profile_ncu_extended"
