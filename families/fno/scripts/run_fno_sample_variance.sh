#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
mkdir -p results/jetson_fno_samples

run_sample () {
  local tag="$1"
  local sample_idx="$2"
  shift 2

  local tegra_log="results/jetson_fno_samples/${tag}_s${sample_idx}_tegrastats.log"

  tegrastats --interval 100 --logfile "${tegra_log}" &
  TPID=$!

  set +e
  python -m src.eval.benchmark_inference "$@" --sample-index "${sample_idx}"
  STATUS=$?
  set -e

  kill ${TPID} 2>/dev/null || true
  wait ${TPID} 2>/dev/null || true

  if [[ ${STATUS} -eq 0 ]]; then
    echo "[ OK ] ${tag} sample=${sample_idx}"
  else
    echo "[FAIL] ${tag} sample=${sample_idx} exit=${STATUS}"
  fi
}

for S in 0 1 2 3 4 5 6 7; do
  run_sample "burgers_base_ts_fp32" "${S}" \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
    --torchscript artifacts/torchscript/burgers_fno_base_seed3.ts \
    --input-bank artifacts/benchmark_inputs/burgers_r2048_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 20 \
    --num-iters 100 \
    --device cuda \
    --results-dir results/jetson_fno_samples \
    --result-tag "burgers_base_ts_fp32_s${S}"
done

for S in 0 1 2 3 4 5 6 7; do
  run_sample "darcy_base_ts_fp32" "${S}" \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_base_seed0_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_base_seed0.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r141_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 20 \
    --num-iters 100 \
    --device cuda \
    --results-dir results/jetson_fno_samples \
    --result-tag "darcy_base_ts_fp32_s${S}"
done