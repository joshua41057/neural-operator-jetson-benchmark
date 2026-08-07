#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
mkdir -p results/jetson_fno_frontier

run_one () {
  local tag="$1"
  shift

  local tegra_log="results/jetson_fno_frontier/${tag}_tegrastats.log"
  local out_json="results/jetson_fno_frontier/${tag}.json"

  if [[ -f "${out_json}" ]]; then
    echo "[SKIP] ${tag}"
    return 0
  fi

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  tegrastats --interval 100 --logfile "${tegra_log}" &
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

# ------------------------------------------------------------------
# Native/frontier stress points
# ------------------------------------------------------------------

# Burgers: use largest trained checkpoint(s), run at native 8192
run_one "burgers_base_r4096_on_8192_eager_fp32" \
  --mode eager \
  --checkpoint artifacts/checkpoints/burgers_fno_base_r4096_seed0_best.pt \
  --input-bank artifacts/benchmark_inputs/burgers_r8192_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 20 \
  --num-iters 100 \
  --device cuda \
  --results-dir results/jetson_fno_frontier \
  --result-tag burgers_base_r4096_on_8192_eager_fp32

run_one "burgers_base_r4096_on_8192_torchscript_fp32" \
  --mode torchscript \
  --checkpoint artifacts/checkpoints/burgers_fno_base_r4096_seed0_best.pt \
  --torchscript artifacts/torchscript/burgers_fno_base_r4096_seed0.ts \
  --input-bank artifacts/benchmark_inputs/burgers_r8192_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 20 \
  --num-iters 100 \
  --device cuda \
  --results-dir results/jetson_fno_frontier \
  --result-tag burgers_base_r4096_on_8192_torchscript_fp32

# Darcy: use largest trained base checkpoint(s), run at native 421
run_one "darcy_base_r281_on_421_eager_fp32" \
  --mode eager \
  --checkpoint artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 20 \
  --num-iters 100 \
  --device cuda \
  --results-dir results/jetson_fno_frontier \
  --result-tag darcy_base_r281_on_421_eager_fp32

run_one "darcy_base_r281_on_421_torchscript_fp32" \
  --mode torchscript \
  --checkpoint artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  --torchscript artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
  --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 20 \
  --num-iters 100 \
  --device cuda \
  --results-dir results/jetson_fno_frontier \
  --result-tag darcy_base_r281_on_421_torchscript_fp32

# Larger-model stress too
run_one "darcy_large_r141_on_421_eager_fp32" \
  --mode eager \
  --checkpoint artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 20 \
  --num-iters 100 \
  --device cuda \
  --results-dir results/jetson_fno_frontier \
  --result-tag darcy_large_r141_on_421_eager_fp32

run_one "darcy_large_r141_on_421_torchscript_fp32" \
  --mode torchscript \
  --checkpoint artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  --torchscript artifacts/torchscript/darcy_fno_large_seed0.ts \
  --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
  --precision fp32 \
  --batch-size 1 \
  --num-warmup 20 \
  --num-iters 100 \
  --device cuda \
  --results-dir results/jetson_fno_frontier \
  --result-tag darcy_large_r141_on_421_torchscript_fp32