#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

OUTDIR="results/jetson_fno_profile_module"
mkdir -p "${OUTDIR}"

run_one () {
  local tag="$1"
  shift

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  python -m src.eval.module_profile_eager "$@" \
    --results-dir "${OUTDIR}" \
    --result-tag "${tag}"
}

run_one "darcy_r85_eager_module_profile" \
  --checkpoint artifacts/checkpoints/darcy_fno_base_r85_seed2_best.pt \
  --input-bank artifacts/benchmark_inputs/darcy_r85_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 5 \
  --num-iters 20 \
  --device cuda

run_one "darcy_r281_eager_module_profile" \
  --checkpoint artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  --input-bank artifacts/benchmark_inputs/darcy_r281_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 3 \
  --num-iters 10 \
  --device cuda

run_one "darcy_large_on421_eager_module_profile" \
  --checkpoint artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 2 \
  --num-iters 5 \
  --device cuda