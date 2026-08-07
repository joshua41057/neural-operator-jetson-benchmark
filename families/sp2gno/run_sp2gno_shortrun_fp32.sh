#!/usr/bin/env bash
# Short-run timing class for Sp2GNO, FP32 strict.
#
# Adds the measurement class that the paper's cross-family comparisons
# (Fig. 2, Table 5, Table 10) require: fixed-iteration window on a single
# request, no concurrent telemetry, matching the FNO/DeepONet/WNO short-run
# protocol (30 warmup / 100 timed iterations, batch size one). The 120 s
# sustained runs already collected under inference_runs/ are left untouched.
set -u

cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026

unset PYTORCH_NO_CUDA_MEMORY_CACHING
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NEW="${NEW:-/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints}"
DATA_DIR="${DATA_DIR:-/home/jetson/data}"
BURGERS_SPLIT="${BURGERS_SPLIT:-Jetson_data/burgers_split.json}"

RESULTS_ROOT="${RESULTS_ROOT:-inference_runs}"
RUN_TAG="${RUN_TAG:-sp2gno_shortrun_fp32}"
SUITE_ROOT="${RESULTS_ROOT}/${RUN_TAG}"

WARMUP="${WARMUP:-30}"
NUM_ITERS="${NUM_ITERS:-100}"
VALIDITY_SAMPLES="${VALIDITY_SAMPLES:-8}"
REPS="${REPS:-3}"
PYBIN="${PYBIN:-python3}"

mkdir -p "${SUITE_ROOT}"
LOG="${SUITE_ROOT}/run_matrix.log"

run_one () {
  local case_id="$1" dataset="$2" ckpt="$3" width="$4" sub="$5" res="$6" k="$7" rep="$8"
  local run_name="${case_id}_fp32_strict_rep${rep}"

  local a=(
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
    --precision fp32_strict
    --warmup "${WARMUP}"
    --timing_class short_run
    --num_iters "${NUM_ITERS}"
    --validity_samples "${VALIDITY_SAMPLES}"
    --sample_index 0
    --rep "${rep}"
  )

  if [[ "${dataset}" == "burgers" ]]; then
    a+=(--sub "${sub}" --burgers_split "${BURGERS_SPLIT}")
  else
    a+=(--res "${res}" --ntrain 900 --nval 100 --ntest 200)
  fi

  echo "[RUN] ${run_name}" | tee -a "${LOG}"
  if "${PYBIN}" bench_sp2gno_jetson_exact.py "${a[@]}" >> "${LOG}" 2>&1; then
    echo "[ OK ] ${run_name}" | tee -a "${LOG}"
  else
    echo "[FAIL] ${run_name}" | tee -a "${LOG}"
  fi
  sleep 2
}

run_case () {
  for rep in $(seq 1 "${REPS}"); do
    run_one "$1" "$2" "$3" "$4" "$5" "$6" "$7" "${rep}"
  done
}

echo "RUN_TAG=${RUN_TAG} timing_class=short_run warmup=${WARMUP} iters=${NUM_ITERS} reps=${REPS}" | tee -a "${LOG}"

#         case_id                     dataset  ckpt                                  width sub res  k
run_case sp2gno_burgers_small_s2048 burgers "${NEW}/sp2gno_burgers_small_s2048.pth" 13 4  0 8
run_case sp2gno_burgers_base_s2048  burgers "${NEW}/sp2gno_burgers_base_s2048.pth"  24 4  0 8
run_case sp2gno_burgers_large_s2048 burgers "${NEW}/sp2gno_burgers_large_s2048.pth" 45 4  0 8
run_case sp2gno_burgers_base_r512   burgers "${NEW}/sp2gno_burgers_base_r512.pth"   24 16 0 8
run_case sp2gno_burgers_base_r1024  burgers "${NEW}/sp2gno_burgers_base_r1024.pth"  24 8  0 8
run_case sp2gno_burgers_base_s4096  burgers "${NEW}/sp2gno_burgers_base_s4096.pth"  24 2  0 8
run_case sp2gno_burgers_base_r8192  burgers "${NEW}/sp2gno_burgers_base_r8192.pth"  24 1  0 8

run_case sp2gno_darcy_small_r141 darcy "${NEW}/sp2gno_darcy_small_r141.pth" 13 0 141 20
run_case sp2gno_darcy_base_r141  darcy "${NEW}/sp2gno_darcy_base_r141.pth"  24 0 141 20
run_case sp2gno_darcy_large_r141 darcy "${NEW}/sp2gno_darcy_large_r141.pth" 45 0 141 20
run_case sp2gno_darcy_base_r85   darcy "${NEW}/sp2gno_darcy_base_r85.pth"   24 0  85 20
run_case sp2gno_darcy_base_r211  darcy "${NEW}/sp2gno_darcy_base_r211.pth"  24 0 211 20
run_case sp2gno_darcy_base_r281  darcy "${NEW}/sp2gno_darcy_base_r281.pth"  24 0 281 20
run_case sp2gno_darcy_base_r421  darcy "${NEW}/sp2gno_darcy_base_r421.pth"  24 0 421 20

echo "[DONE] suite_root=${SUITE_ROOT}" | tee -a "${LOG}"
