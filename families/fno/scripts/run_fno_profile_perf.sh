#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

OUTDIR="results/jetson_fno_profile_perf"
mkdir -p "${OUTDIR}"

run_perf () {
  local tag="$1"
  shift

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  perf stat -d -d -d \
    -o "${OUTDIR}/${tag}_perf.txt" \
    "$@"
}

run_perf "burgers_base_eager_fp32" \
  python -m src.eval.profile_inference \
    --mode eager \
    --checkpoint artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
    --input-bank artifacts/benchmark_inputs/burgers_r2048_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 20 \
    --num-iters 100 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag burgers_base_eager_fp32 \
    --notes "perf eager"

run_perf "burgers_base_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
    --torchscript artifacts/torchscript/burgers_fno_base_seed3.ts \
    --input-bank artifacts/benchmark_inputs/burgers_r2048_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 20 \
    --num-iters 100 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag burgers_base_ts_fp32 \
    --notes "perf torchscript"