#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
PYTHONBIN="${PYTHONBIN:-$(which python)}"
NCUBIN="${NCUBIN:-/usr/local/cuda-12.6/bin/ncu}"

OUTDIR="results/profiles/deeponet_ncu_detailed"
EXPDIR="results/profiles/deeponet_ncu_detailed_exports"

rm -rf "$OUTDIR" "$EXPDIR"
mkdir -p "$OUTDIR" "$EXPDIR"

sudo -v

run_case () {
  local CASE_ID="$1"
  local PRECISION="$2"
  local CHECKPOINT="$3"
  local TORCHSCRIPT="$4"
  local INPUT_BANK="$5"

  echo
  echo "=== NCU detailed canonical case: ${CASE_ID} ==="
  echo "precision=${PRECISION}"
  echo "checkpoint=${CHECKPOINT}"
  echo "torchscript=${TORCHSCRIPT}"
  echo "input_bank=${INPUT_BANK}"

  test -f "$CHECKPOINT"
  test -f "$TORCHSCRIPT"
  test -f "$INPUT_BANK"

  sudo env \
    PYTHONPATH="$PWD" \
    PYTHONNOUSERSITE=1 \
    "$NCUBIN" \
      --force-overwrite \
      --target-processes all \
      --profile-from-start off \
      --replay-mode kernel \
      --cache-control all \
      --clock-control none \
      --set detailed \
      --export "${OUTDIR}/${CASE_ID}" \
      --log-file "${OUTDIR}/${CASE_ID}.log" \
      "$PYTHONBIN" src/eval/profile_forward_ncu.py \
        --mode torchscript \
        --checkpoint "$CHECKPOINT" \
        --torchscript "$TORCHSCRIPT" \
        --input-bank "$INPUT_BANK" \
        --precision "$PRECISION" \
        --device cuda \
        --batch-size 1 \
        --warmup 10 \
        --profile-iters 1 \
        --result-json "${OUTDIR}/${CASE_ID}_forward.json"

  sudo chown -R "$USER:$USER" "$OUTDIR"

  local REP="${OUTDIR}/${CASE_ID}.ncu-rep"

  if [ ! -f "$REP" ]; then
    echo "ERROR: missing report: $REP"
    tail -120 "${OUTDIR}/${CASE_ID}.log" || true
    exit 2
  fi

  if grep -q "No kernels were profiled" "${OUTDIR}/${CASE_ID}.log"; then
    echo "ERROR: NCU produced report path check but profiled no kernels: ${CASE_ID}"
    tail -120 "${OUTDIR}/${CASE_ID}.log" || true
    exit 3
  fi

  "$NCUBIN" \
    --import "$REP" \
    --csv \
    --page raw \
    --print-units base \
    > "${EXPDIR}/${CASE_ID}_raw.csv"

  "$NCUBIN" \
    --import "$REP" \
    --csv \
    --page details \
    --print-details all \
    --print-metric-name name \
    --print-units base \
    > "${EXPDIR}/${CASE_ID}_details.csv"

  "$NCUBIN" \
    --import "$REP" \
    --csv \
    --page details \
    --print-summary per-kernel \
    --print-details all \
    --print-metric-name name \
    --print-units base \
    > "${EXPDIR}/${CASE_ID}_per_kernel.csv"

  echo "DONE: ${CASE_ID}"
}

run_case \
  "burgers_base_r2048_ts_fp32" \
  "fp32_strict" \
  "artifacts/checkpoints/burgers_deeponet_base_r2048_seed2_best.pt" \
  "artifacts/torchscript/burgers_deeponet_base_r2048_seed2.ts" \
  "artifacts/benchmark_inputs/burgers_r2048_bank.pt"

run_case \
  "darcy_base_r141_ts_fp32" \
  "fp32_strict" \
  "artifacts/checkpoints/darcy_deeponet_base_r141_seed2_best.pt" \
  "artifacts/torchscript/darcy_deeponet_base_r141_seed2.ts" \
  "artifacts/benchmark_inputs/darcy_r141_bank.pt"

run_case \
  "darcy_base_r281_ts_fp32" \
  "fp32_strict" \
  "artifacts/checkpoints/darcy_deeponet_base_r281_seed0_best.pt" \
  "artifacts/torchscript/darcy_deeponet_base_r281_seed0.ts" \
  "artifacts/benchmark_inputs/darcy_r281_bank.pt"

run_case \
  "darcy_base_r281_ts_fp16_native" \
  "fp16_native" \
  "artifacts/checkpoints/darcy_deeponet_base_r281_seed0_best.pt" \
  "artifacts/torchscript/darcy_deeponet_base_r281_seed0.ts" \
  "artifacts/benchmark_inputs/darcy_r281_bank.pt"

run_case \
  "darcy_large_r141_ts_fp32" \
  "fp32_strict" \
  "artifacts/checkpoints/darcy_deeponet_large_seed1_best.pt" \
  "artifacts/torchscript/darcy_deeponet_large_seed1.ts" \
  "artifacts/benchmark_inputs/darcy_r141_bank.pt"

echo
echo "=== report files ==="
ls -lh "$OUTDIR"/*.ncu-rep

echo
echo "=== export files ==="
ls -lh "$EXPDIR"/*.csv

echo
echo "=== error scan ==="
grep -R "No kernels were profiled\|ERROR\|Insufficient\|Failed" "$OUTDIR" "$EXPDIR" || true
