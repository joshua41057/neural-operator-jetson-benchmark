#!/usr/bin/env bash
set -u
set -o pipefail

cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks || exit 1

if [ -n "${CONDA_EXE:-}" ]; then
  source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
fi
conda activate vs_wno 2>/dev/null || true

export PYTHONPATH="$PWD/sample_codes:$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

RESULTS_ROOT="${RESULTS_ROOT:-results/jetson_wno_exact}"
RUN_TAG="${RUN_TAG:-wno_exact_new_r1024_r85_r211_$(date +%Y%m%d_%H%M%S)}"

WARMUP_SECONDS="${WARMUP_SECONDS:-20}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
REPS="${REPS:-3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${RESULTS_ROOT}/${RUN_TAG}"

MASTER_LOG="${RESULTS_ROOT}/${RUN_TAG}/run_matrix.log"
SUCCESS_LOG="${RESULTS_ROOT}/${RUN_TAG}/success.txt"
FAIL_LOG="${RESULTS_ROOT}/${RUN_TAG}/fail.txt"

touch "${MASTER_LOG}" "${SUCCESS_LOG}" "${FAIL_LOG}"

run_one () {
  local case_id="$1"
  local dataset="$2"
  local ckpt="$3"
  local bank="$4"
  local precision="$5"
  local rep="$6"

  local run_case_id="${case_id}_rep${rep}"
  local out_json="${RESULTS_ROOT}/${RUN_TAG}/${run_case_id}/${precision}/result.json"

  if [ -f "${out_json}" ] && [ "${OVERWRITE}" != "1" ]; then
    echo "[SKIP] ${run_case_id} ${precision}" | tee -a "${MASTER_LOG}"
    return 0
  fi

  echo "====================================================================================================" | tee -a "${MASTER_LOG}"
  echo "[RUN] case=${run_case_id} dataset=${dataset} precision=${precision}" | tee -a "${MASTER_LOG}"
  echo "ckpt=${ckpt}" | tee -a "${MASTER_LOG}"
  echo "bank=${bank}" | tee -a "${MASTER_LOG}"
  echo "====================================================================================================" | tee -a "${MASTER_LOG}"

  python3 bench_wno_jetson_exact.py \
    --case-id "${run_case_id}" \
    --dataset "${dataset}" \
    --checkpoint "${ckpt}" \
    --bank "${bank}" \
    --precision-mode "${precision}" \
    --sample-index 0 \
    --batch-size 1 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --warmup-seconds "${WARMUP_SECONDS}" \
    --measure-seconds "${MEASURE_SECONDS}" \
    --device cuda \
    --results-root "${RESULTS_ROOT}" \
    --run-tag "${RUN_TAG}" \
    --compute-full-eval 1 \
    --compute-perturbation 1 \
    2>&1 | tee -a "${MASTER_LOG}"

  local rc=${PIPESTATUS[0]}

  if [ "${rc}" -eq 0 ]; then
    echo "${run_case_id},${precision}" >> "${SUCCESS_LOG}"
    echo "[ OK ] ${run_case_id} ${precision}" | tee -a "${MASTER_LOG}"
  else
    echo "${run_case_id},${precision}" >> "${FAIL_LOG}"
    echo "[FAIL] ${run_case_id} ${precision} rc=${rc}" | tee -a "${MASTER_LOG}"
  fi

  sleep 3
}

cases=(
  "wno_burgers_base_r1024|burgers|checkpoints/wno_burgers_base_r1024.pth|/home/jetson/data/wno_inference_banks_exact/burgers_r1024_bank.pt"
  "wno_darcy_base_r85|darcy|checkpoints/wno_darcy_base_r85.pth|/home/jetson/data/wno_inference_banks_exact/darcy_r85_bank.pt"
  "wno_darcy_base_r211|darcy|checkpoints/wno_darcy_base_r211.pth|/home/jetson/data/wno_inference_banks_exact/darcy_r211_bank.pt"
)

precisions=(
  "fp32_strict"
  "tf32"
  "bf16_autocast"
  "fp16_autocast"
  "fp16_native"
)

echo "RUN_TAG=${RUN_TAG}" | tee -a "${MASTER_LOG}"

for row in "${cases[@]}"; do
  IFS='|' read -r case_id dataset ckpt bank <<< "${row}"
  for precision in "${precisions[@]}"; do
    for rep in $(seq 1 "${REPS}"); do
      run_one "${case_id}" "${dataset}" "${ckpt}" "${bank}" "${precision}" "${rep}"
    done
  done
done

echo "DONE RUN_TAG=${RUN_TAG}" | tee -a "${MASTER_LOG}"
echo "Success log: ${SUCCESS_LOG}" | tee -a "${MASTER_LOG}"
echo "Fail log   : ${FAIL_LOG}" | tee -a "${MASTER_LOG}"
