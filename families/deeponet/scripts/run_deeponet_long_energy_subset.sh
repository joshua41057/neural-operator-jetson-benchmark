#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"

RESULTS_DIR="results/jetson_deeponet_long_energy"
mkdir -p "$RESULTS_DIR" logs results/artifacts

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_deeponet_long_energy_subset_${STAMP}.log"
SUCCESS="logs/run_deeponet_long_energy_subset_${STAMP}_success.csv"
FAIL="logs/run_deeponet_long_energy_subset_${STAMP}_fail.csv"

echo "experiment_name,backend,precision,result_tag" > "$SUCCESS"
echo "experiment_name,backend,precision,result_tag,exit_code" > "$FAIL"

DURATION_SEC="${DURATION_SEC:-120}"
WARMUP_SEC="${WARMUP_SEC:-10}"

echo "=== DeepONet long-run energy subset ===" | tee "$LOG"
echo "PWD=$PWD" | tee -a "$LOG"
echo "DURATION_SEC=$DURATION_SEC" | tee -a "$LOG"
echo "WARMUP_SEC=$WARMUP_SEC" | tee -a "$LOG"
date | tee -a "$LOG"

python - <<'PY' | tee -a "$LOG"
import torch, src
print("src:", src.__file__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("matmul_tf32_initial:", torch.backends.cuda.matmul.allow_tf32)
print("cudnn_tf32_initial:", torch.backends.cudnn.allow_tf32)
PY

run_case () {
  local exp="$1"
  local ckpt="$2"
  local ts="$3"
  local bank="$4"
  local precision="$5"

  local backend="torchscript"
  local tag="${exp}_${backend}_${precision}_${DURATION_SEC}s"

  echo | tee -a "$LOG"
  echo "=== LONG RUN $tag ===" | tee -a "$LOG"

  set +e
  python -m src.eval.benchmark_sustained_inference \
    --mode "$backend" \
    --checkpoint "$ckpt" \
    --torchscript "$ts" \
    --input-bank "$bank" \
    --precision "$precision" \
    --batch-size 1 \
    --warmup-sec "$WARMUP_SEC" \
    --duration-sec "$DURATION_SEC" \
    --device cuda \
    --tegrastats-interval-ms 1000 \
    --results-dir "$RESULTS_DIR" \
    --result-tag "$tag" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "$exp,$backend,$precision,$tag" >> "$SUCCESS"
  else
    echo "$exp,$backend,$precision,$tag,$rc" >> "$FAIL"
  fi
}

run_all_precisions () {
  local exp="$1"
  local ckpt="$2"
  local ts="$3"
  local bank="$4"

  run_case "$exp" "$ckpt" "$ts" "$bank" fp32_strict
  run_case "$exp" "$ckpt" "$ts" "$bank" tf32
  run_case "$exp" "$ckpt" "$ts" "$bank" bf16_autocast
  run_case "$exp" "$ckpt" "$ts" "$bank" fp16_autocast
  run_case "$exp" "$ckpt" "$ts" "$bank" fp16_native
}

# Representative subset for main paper:
# 1D base, 2D base, 2D high-resolution base, 2D large controlled model.
run_all_precisions \
  burgers_deeponet_base \
  artifacts/checkpoints/burgers_deeponet_base_seed2_best.pt \
  artifacts/torchscript/burgers_deeponet_base_seed2.ts \
  artifacts/benchmark_inputs/burgers_r2048_bank.pt

run_all_precisions \
  darcy_deeponet_base \
  artifacts/checkpoints/darcy_deeponet_base_seed2_best.pt \
  artifacts/torchscript/darcy_deeponet_base_seed2.ts \
  artifacts/benchmark_inputs/darcy_r141_bank.pt

run_all_precisions \
  darcy_deeponet_base_r281 \
  artifacts/checkpoints/darcy_deeponet_base_r281_seed0_best.pt \
  artifacts/torchscript/darcy_deeponet_base_r281_seed0.ts \
  artifacts/benchmark_inputs/darcy_r281_bank.pt

run_all_precisions \
  darcy_deeponet_large \
  artifacts/checkpoints/darcy_deeponet_large_seed1_best.pt \
  artifacts/torchscript/darcy_deeponet_large_seed1.ts \
  artifacts/benchmark_inputs/darcy_r141_bank.pt

run_all_precisions \
  burgers_deeponet_base_r4096 \
  artifacts/checkpoints/burgers_deeponet_base_r4096_seed2_best.pt \
  artifacts/torchscript/burgers_deeponet_base_r4096_seed2.ts \
  artifacts/benchmark_inputs/burgers_r4096_bank.pt

run_all_precisions \
  burgers_deeponet_base_r8192 \
  artifacts/checkpoints/burgers_deeponet_base_r8192_seed0_best.pt \
  artifacts/torchscript/burgers_deeponet_base_r8192_seed0.ts \
  artifacts/benchmark_inputs/burgers_r8192_bank.pt

run_all_precisions \
  darcy_deeponet_base_r421 \
  artifacts/checkpoints/darcy_deeponet_base_r421_seed2_best.pt \
  artifacts/torchscript/darcy_deeponet_base_r421_seed2.ts \
  artifacts/benchmark_inputs/darcy_r421_bank.pt

echo | tee -a "$LOG"
echo "Done." | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "Success: $SUCCESS" | tee -a "$LOG"
echo "Fail: $FAIL" | tee -a "$LOG"
echo "JSON count:" | tee -a "$LOG"
find "$RESULTS_DIR" -name '*.json' | wc -l | tee -a "$LOG"
