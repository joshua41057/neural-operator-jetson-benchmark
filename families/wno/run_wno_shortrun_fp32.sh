#!/usr/bin/env bash
# Short-run timing class for WNO, FP32 strict.
#
# Adds the measurement class that the paper's cross-family comparisons
# (Fig. 2, Table 5, Table 10) require: fixed-iteration window, no concurrent
# telemetry, matching the FNO/DeepONet protocol (30 warmup / 100 timed
# iterations, batch size one). The 120 s sustained runs already collected
# under results/jetson_wno_exact are left untouched.
set -uo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-results/jetson_wno_exact}"
RUN_TAG="${RUN_TAG:-wno_shortrun_fp32_$(date +%Y%m%d_%H%M%S)}"
REPS="${REPS:-3}"
NUM_WARMUP="${NUM_WARMUP:-30}"
NUM_ITERS="${NUM_ITERS:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
PYBIN="${PYBIN:-/home/jetson/miniforge3/envs/extra_bench/bin/python}"

OUT_DIR="${RESULTS_ROOT}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
MASTER_LOG="${OUT_DIR}/run_matrix.log"
SUCCESS_LOG="${OUT_DIR}/success.txt"
FAIL_LOG="${OUT_DIR}/fail.txt"
: > "${SUCCESS_LOG}"; : > "${FAIL_LOG}"

BANKS="/home/jetson/data/wno_inference_banks_exact"

CASES=(
  "wno_burgers_small_r2048|burgers|checkpoints/wno_burgers_small_r2048.pth|${BANKS}/burgers_r2048_bank.pt"
  "wno_burgers_base_r2048|burgers|checkpoints/wno_burgers_base_r2048.pth|${BANKS}/burgers_r2048_bank.pt"
  "wno_burgers_large_r2048|burgers|checkpoints/wno_burgers_large_r2048.pth|${BANKS}/burgers_r2048_bank.pt"
  "wno_burgers_base_r512|burgers|checkpoints/wno_burgers_base_r512.pth|${BANKS}/burgers_r512_bank.pt"
  "wno_burgers_base_r1024|burgers|checkpoints/wno_burgers_base_r1024.pth|${BANKS}/burgers_r1024_bank.pt"
  "wno_burgers_base_r4096|burgers|checkpoints/wno_burgers_base_r4096.pth|${BANKS}/burgers_r4096_bank.pt"
  "wno_burgers_base_r8192|burgers|checkpoints/wno_burgers_base_r8192.pth|${BANKS}/burgers_r8192_bank.pt"
  "wno_darcy_small_r141|darcy|checkpoints/wno_darcy_small_r141.pth|${BANKS}/darcy_r141_bank.pt"
  "wno_darcy_base_r141|darcy|checkpoints/wno_darcy_base_r141.pth|${BANKS}/darcy_r141_bank.pt"
  "wno_darcy_large_r141|darcy|checkpoints/wno_darcy_large_r141.pth|${BANKS}/darcy_r141_bank.pt"
  "wno_darcy_base_r85|darcy|checkpoints/wno_darcy_base_r85.pth|${BANKS}/darcy_r85_bank.pt"
  "wno_darcy_base_r211|darcy|checkpoints/wno_darcy_base_r211.pth|${BANKS}/darcy_r211_bank.pt"
  "wno_darcy_base_r281|darcy|checkpoints/wno_darcy_base_r281.pth|${BANKS}/darcy_r281_bank.pt"
  "wno_darcy_base_r421|darcy|checkpoints/wno_darcy_base_r421.pth|${BANKS}/darcy_r421_bank.pt"
)

echo "RUN_TAG=${RUN_TAG} timing_class=short_run warmup=${NUM_WARMUP} iters=${NUM_ITERS} reps=${REPS}" \
  | tee -a "${MASTER_LOG}"

for row in "${CASES[@]}"; do
  IFS='|' read -r case_id dataset ckpt bank <<< "${row}"
  for rep in $(seq 1 "${REPS}"); do
    run_case_id="${case_id}_rep${rep}"
    echo "[RUN] ${run_case_id} fp32_strict" | tee -a "${MASTER_LOG}"

    PYTHONNOUSERSITE=1 "${PYBIN}" bench_wno_jetson_exact.py \
      --case-id "${run_case_id}" \
      --dataset "${dataset}" \
      --checkpoint "${ckpt}" \
      --bank "${bank}" \
      --precision-mode fp32_strict \
      --sample-index 0 \
      --batch-size 1 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --timing-class short_run \
      --num-warmup "${NUM_WARMUP}" \
      --num-iters "${NUM_ITERS}" \
      --device cuda \
      --results-root "${RESULTS_ROOT}" \
      --run-tag "${RUN_TAG}" \
      --compute-full-eval 0 \
      --compute-perturbation 0 \
      >> "${MASTER_LOG}" 2>&1

    rc=$?
    if [ "${rc}" -eq 0 ]; then
      echo "${run_case_id},fp32_strict" >> "${SUCCESS_LOG}"
      echo "[ OK ] ${run_case_id}" | tee -a "${MASTER_LOG}"
    else
      echo "${run_case_id},fp32_strict" >> "${FAIL_LOG}"
      echo "[FAIL] ${run_case_id} rc=${rc}" | tee -a "${MASTER_LOG}"
    fi
    sleep 2
  done
done

echo "DONE ${RUN_TAG}" | tee -a "${MASTER_LOG}"
