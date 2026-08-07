#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

OUTDIR="results/jetson_fno_profile_ncu"
mkdir -p "${OUTDIR}"

run_ncu () {
  local tag="$1"
  shift

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  ncu \
    --set full \
    --target-processes all \
    --export "${OUTDIR}/${tag}" \
    "$@"
}

run_ncu "darcy_r85_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_base_r85_seed2_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_base_r85_seed2.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r85_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 20 \
    --num-iters 30 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag darcy_r85_ts_fp32 \
    --notes "ncu low-res"

run_ncu "darcy_r281_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r281_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 20 \
    --num-iters 30 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag darcy_r281_ts_fp32 \
    --notes "ncu high-res"