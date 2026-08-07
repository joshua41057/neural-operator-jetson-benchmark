#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
mkdir -p results/jetson_fno_sustained

run_sustained () {
  local tag="$1"
  shift

  local tegra_log="results/jetson_fno_sustained/${tag}_tegrastats.log"
  local out_json="results/jetson_fno_sustained/${tag}.json"

  tegrastats --interval 200 --logfile "${tegra_log}" &
  TPID=$!

  set +e
  python -m src.eval.benchmark_inference "$@"
  STATUS=$?
  set -e

  kill ${TPID} 2>/dev/null || true
  wait ${TPID} 2>/dev/null || true

  if [[ ${STATUS} -eq 0 ]]; then
    echo "[ OK ] ${tag}"
  else
    echo "[FAIL] ${tag} (exit=${STATUS})"
  fi
}

run_sustained "burgers_base_ts_fp32_sustained" \
  --mode torchscript \
  --checkpoint artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
  --torchscript artifacts/torchscript/burgers_fno_base_seed3.ts \
  --input-bank artifacts/benchmark_inputs/burgers_r2048_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 100 \
  --num-iters 5000 \
  --device cuda \
  --results-dir results/jetson_fno_sustained \
  --result-tag burgers_base_ts_fp32_sustained

run_sustained "darcy_base_ts_fp32_sustained" \
  --mode torchscript \
  --checkpoint artifacts/checkpoints/darcy_fno_base_seed0_best.pt \
  --torchscript artifacts/torchscript/darcy_fno_base_seed0.ts \
  --input-bank artifacts/benchmark_inputs/darcy_r141_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 100 \
  --num-iters 5000 \
  --device cuda \
  --results-dir results/jetson_fno_sustained \
  --result-tag darcy_base_ts_fp32_sustained

run_sustained "darcy_large_ts_fp32_sustained" \
  --mode torchscript \
  --checkpoint artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  --torchscript artifacts/torchscript/darcy_fno_large_seed0.ts \
  --input-bank artifacts/benchmark_inputs/darcy_r141_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 100 \
  --num-iters 5000 \
  --device cuda \
  --results-dir results/jetson_fno_sustained \
  --result-tag darcy_large_ts_fp32_sustained