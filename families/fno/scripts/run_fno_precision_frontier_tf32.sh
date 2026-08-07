#!/usr/bin/env bash
set -u
set -o pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

RESULTS_DIR="results/jetson_fno_precision_frontier"
mkdir -p "${RESULTS_DIR}"

NUM_WARMUP="${NUM_WARMUP:-20}"
NUM_ITERS="${NUM_ITERS:-100}"
REPEATS="${REPEATS:-3}"
OVERWRITE="${OVERWRITE:-0}"

run_case () {
  local tag="$1"
  local ckpt="$2"
  local ts="$3"
  local bank="$4"
  local mode="$5"
  local precision="$6"

  local out="${RESULTS_DIR}/${tag}_${mode}_${precision}.json"
  if [[ -f "${out}" && "${OVERWRITE}" != "1" ]]; then
    echo "[SKIP] ${tag}_${mode}_${precision}"
    return 0
  fi

  echo "================================================================================"
  echo "[RUN] ${tag}_${mode}_${precision}"
  echo "================================================================================"

  local cmd=(
    python -m src.eval.benchmark_precision_inference
    --mode "${mode}"
    --checkpoint "${ckpt}"
    --input-bank "${bank}"
    --sample-index 0
    --batch-size 1
    --precision-mode "${precision}"
    --num-warmup "${NUM_WARMUP}"
    --num-iters "${NUM_ITERS}"
    --repeats "${REPEATS}"
    --device cuda
    --results-dir "${RESULTS_DIR}"
    --result-tag "${tag}_${mode}_${precision}"
  )

  if [[ "${mode}" == "torchscript" ]]; then
    cmd+=(--torchscript "${ts}")
  fi

  "${cmd[@]}"
}

# Frontier-relevant cases only.
# Do not run bf16/fp16 here; controlled matrix already established incompatibility.
CASES=(
  "burgers_base_r4096_on_8192|artifacts/checkpoints/burgers_fno_base_r4096_seed0_best.pt|artifacts/torchscript/burgers_fno_base_r4096_seed0.ts|artifacts/benchmark_inputs/burgers_r8192_bank.pt"
  "darcy_base_r281_on_421|artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt|artifacts/torchscript/darcy_fno_base_r281_seed1.ts|artifacts/benchmark_inputs/darcy_r421_bank.pt"
  "darcy_large_r141_on_421|artifacts/checkpoints/darcy_fno_large_seed0_best.pt|artifacts/torchscript/darcy_fno_large_seed0.ts|artifacts/benchmark_inputs/darcy_r421_bank.pt"
)

for c in "${CASES[@]}"; do
  IFS='|' read -r tag ckpt ts bank <<< "${c}"
  for mode in eager torchscript; do
    for precision in fp32_strict tf32; do
      run_case "${tag}" "${ckpt}" "${ts}" "${bank}" "${mode}" "${precision}"
      sleep 2
    done
  done
done

echo "Done. Results in ${RESULTS_DIR}"
