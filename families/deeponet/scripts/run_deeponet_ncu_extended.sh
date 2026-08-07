#!/usr/bin/env bash
# Run with sudo (ncu requires root on this Jetson):
#   sudo bash scripts/run_deeponet_ncu_extended.sh
set -euo pipefail

cd /home/jetson/jjyoo3/EDCNO_DeepONet || exit 1

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
PYTHONBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
NCUBIN="/usr/local/cuda-12.6/bin/ncu"

OUTDIR="results/profiles/deeponet_ncu_extended"
EXPDIR="results/profiles/deeponet_ncu_extended_exports"
mkdir -p "$OUTDIR" "$EXPDIR"

run_case () {
  local CASE_ID="$1"
  local PRECISION="$2"
  local CHECKPOINT="$3"
  local TORCHSCRIPT="$4"
  local INPUT_BANK="$5"

  if [[ -s "${OUTDIR}/${CASE_ID}.ncu-rep" ]]; then
    echo "[SKIP] ${CASE_ID} (already exists)"
    return 0
  fi

  echo
  echo "=== NCU case: ${CASE_ID} ==="

  test -f "$CHECKPOINT"
  test -f "$TORCHSCRIPT"
  test -f "$INPUT_BANK"

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

  local REP="${OUTDIR}/${CASE_ID}.ncu-rep"
  if [ ! -f "$REP" ]; then
    echo "ERROR: missing report: $REP"
    tail -120 "${OUTDIR}/${CASE_ID}.log" || true
    return 1
  fi

  "$NCUBIN" --import "$REP" --csv --page raw --print-units base \
    > "${EXPDIR}/${CASE_ID}_raw.csv"
  "$NCUBIN" --import "$REP" --csv --page details --print-details all --print-metric-name name --print-units base \
    > "${EXPDIR}/${CASE_ID}_details.csv"
  "$NCUBIN" --import "$REP" --csv --page details --print-summary per-kernel --print-details all --print-metric-name name --print-units base \
    > "${EXPDIR}/${CASE_ID}_per_kernel.csv"

  echo "DONE: ${CASE_ID}"
}

run_case \
  "burgers_base_r8192_ts_fp32" \
  "fp32_strict" \
  "artifacts/checkpoints/burgers_deeponet_base_r8192_seed0_best.pt" \
  "artifacts/torchscript/burgers_deeponet_base_r8192_seed0.ts" \
  "artifacts/benchmark_inputs/burgers_r8192_bank.pt"

run_case \
  "darcy_base_r421_ts_fp32" \
  "fp32_strict" \
  "artifacts/checkpoints/darcy_deeponet_base_r421_seed2_best.pt" \
  "artifacts/torchscript/darcy_deeponet_base_r421_seed2.ts" \
  "artifacts/benchmark_inputs/darcy_r421_bank.pt"

echo "[DONE] DeepONet ncu profiling (2 extended cases)"
echo "Now run: sudo chown -R jetson:jetson results/profiles/deeponet_ncu_extended results/profiles/deeponet_ncu_extended_exports"
