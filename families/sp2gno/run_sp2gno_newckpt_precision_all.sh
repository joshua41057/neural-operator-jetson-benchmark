#!/usr/bin/env bash
set -u

cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026

unset PYTORCH_NO_CUDA_MEMORY_CACHING
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NEW="${NEW:-/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints}"
DATA_DIR="${DATA_DIR:-/home/jetson/data}"
BURGERS_SPLIT="${BURGERS_SPLIT:-Jetson_data/burgers_split.json}"

RESULTS_ROOT="${RESULTS_ROOT:-inference_runs}"
RUN_TAG="${RUN_TAG:-sp2gno_exact_precision_all_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${RESULTS_ROOT}/${RUN_TAG}"

WARMUP="${WARMUP:-20}"
DURATION="${DURATION:-120}"
REPS="${REPS:-3}"

mkdir -p "${SUITE_ROOT}"

run_one () {
  local case_id="$1"
  local dataset="$2"
  local ckpt="$3"
  local width="$4"
  local sub="$5"
  local r="$6"
  local k="$7"
  local precision="$8"
  local rep="$9"
  local ref="${10:-}"

  local run_name="${case_id}_${precision}_rep${rep}"

  local common_args=(
    --case_id "${case_id}"
    --run_name "${run_name}"
    --suite_root "${SUITE_ROOT}"
    --dataset "${dataset}"
    --data_dir "${DATA_DIR}"
    --cache_dir cache
    --ckpt "${ckpt}"
    --width "${width}"
    --n_layers 6
    --num_freq 64
    --k "${k}"
    --precision "${precision}"
    --warmup "${WARMUP}"
    --min_duration_s "${DURATION}"
    --min_cycles 1
    --rep "${rep}"
    --save_outputs
  )

  if [[ "${dataset}" == "burgers" ]]; then
    common_args+=(--sub "${sub}" --burgers_split "${BURGERS_SPLIT}")
  else
    common_args+=(--r "${r}" --ntrain 900 --nval 100 --ntest 200)
  fi

  if [[ -n "${ref}" ]]; then
    common_args+=(--fp32_ref_predictions "${ref}")
  fi

  echo
  echo "[RUN] ${run_name}"

  if python3 bench_sp2gno_jetson_exact.py "${common_args[@]}"; then
    echo "[OK] ${run_name}"
  else
    echo "[FAILED] ${run_name}"
  fi
}

run_case () {
  local case_id="$1"
  local dataset="$2"
  local ckpt="$3"
  local width="$4"
  local sub="$5"
  local r="$6"
  local k="$7"

  echo
  echo "============================================================"
  echo "[CASE] ${case_id}"
  echo "============================================================"

  for rep in $(seq 1 "${REPS}"); do
    run_one "${case_id}" "${dataset}" "${ckpt}" "${width}" "${sub}" "${r}" "${k}" "fp32_strict" "${rep}" ""
  done

  local ref="${SUITE_ROOT}/${case_id}_fp32_strict_rep1/outputs/predictions.npy"

  for precision in tf32 bf16_autocast fp16_autocast fp16_native; do
    for rep in $(seq 1 "${REPS}"); do
      run_one "${case_id}" "${dataset}" "${ckpt}" "${width}" "${sub}" "${r}" "${k}" "${precision}" "${rep}" "${ref}"
    done
  done
}

# Burgers: 8192 / sub = resolution.
run_case sp2gno_burgers_small_s2048 burgers "${NEW}/sp2gno_burgers_small_s2048.pth" 13 4 0 8
run_case sp2gno_burgers_base_s2048  burgers "${NEW}/sp2gno_burgers_base_s2048.pth"  24 4 0 8
run_case sp2gno_burgers_large_s2048 burgers "${NEW}/sp2gno_burgers_large_s2048.pth" 45 4 0 8
run_case sp2gno_burgers_base_s4096  burgers "${NEW}/sp2gno_burgers_base_s4096.pth"  24 2 0 8

# Darcy: 421 / r = resolution.
run_case sp2gno_darcy_small_r141 darcy "${NEW}/sp2gno_darcy_small_r141.pth" 13 0 3 20
run_case sp2gno_darcy_base_r141  darcy "${NEW}/sp2gno_darcy_base_r141.pth"  24 0 3 20
run_case sp2gno_darcy_large_r141 darcy "${NEW}/sp2gno_darcy_large_r141.pth" 45 0 3 20
run_case sp2gno_darcy_base_r211  darcy "${NEW}/sp2gno_darcy_base_r211.pth"  24 0 2 20

echo
echo "[DONE] suite_root=${SUITE_ROOT}"
