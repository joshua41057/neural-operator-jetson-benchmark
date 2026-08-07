#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1

PYTHON_BIN="$(which python)"
NCU_BIN="$(readlink -f "$(which ncu)")"
export PYTHONPATH="$PWD"

OUTDIR="results/jetson_fno_profile_ncu_frontier"
mkdir -p "${OUTDIR}"

run_ncu () {
  local tag="$1"
  shift

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  sudo -E "${NCU_BIN}" \
    --target-processes all \
    --set launchstats \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --force-overwrite \
    --export "${OUTDIR}/${tag}" \
    "$@"
}

# 200ms frontier case only
run_ncu "darcy_large_on421_ts_fp32_ncu" \
  "${PYTHON_BIN}" -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_large_seed0.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 1 \
    --num-iters 1 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag darcy_large_on421_ts_fp32_ncu \
    --notes "frontier ncu same method as prior successful runs"