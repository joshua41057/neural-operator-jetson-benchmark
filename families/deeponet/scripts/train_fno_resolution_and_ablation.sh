#!/usr/bin/env bash
set -u

# Usage:
#   bash scripts/train_fno_resolution_and_ablation.sh [device] [seeds]
#
# Example:
#   bash scripts/train_fno_resolution_and_ablation.sh cuda "0 1 2"
#
# Recommended:
#   1) 먼저 main controlled suite 실행
#      bash scripts/train_all_fno.sh cuda "0 1 2 3 4"
#   2) 끝난 뒤 이 스크립트 실행
#      bash scripts/train_fno_resolution_and_ablation.sh cuda "0 1 2"

DEVICE=${1:-cuda}
SEEDS=${2:-"0 1 2"}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/train_fno_resolution_and_ablation_${RUN_TS}.log"
SUCCESS_LOG="${LOG_DIR}/train_fno_resolution_and_ablation_${RUN_TS}_success.txt"
FAIL_LOG="${LOG_DIR}/train_fno_resolution_and_ablation_${RUN_TS}_fail.txt"

touch "${MASTER_LOG}" "${SUCCESS_LOG}" "${FAIL_LOG}"

echo "==================================================" | tee -a "${MASTER_LOG}"
echo "FNO resolution + ablation training launcher" | tee -a "${MASTER_LOG}"
echo "Root dir : ${ROOT_DIR}" | tee -a "${MASTER_LOG}"
echo "Device   : ${DEVICE}" | tee -a "${MASTER_LOG}"
echo "Seeds    : ${SEEDS}" | tee -a "${MASTER_LOG}"
echo "Started  : $(date)" | tee -a "${MASTER_LOG}"
echo "==================================================" | tee -a "${MASTER_LOG}"

# ---------------------------------------
# Utility functions
# ---------------------------------------

config_to_experiment_name() {
  local cfg="$1"
  python - "$cfg" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['name'])
PY
}

experiment_done_for_all_seeds() {
  local exp_name="$1"
  local seeds_str="$2"

  for s in ${seeds_str}; do
    local summary_path="checkpoints/${exp_name}/seed${s}/summary.json"
    if [[ ! -f "${summary_path}" ]]; then
      return 1
    fi
  done
  return 0
}

run_one_config() {
  local cfg="$1"
  local exp_name
  exp_name="$(config_to_experiment_name "${cfg}")"

  echo "" | tee -a "${MASTER_LOG}"
  echo "--------------------------------------------------" | tee -a "${MASTER_LOG}"
  echo "Config      : ${cfg}" | tee -a "${MASTER_LOG}"
  echo "Experiment  : ${exp_name}" | tee -a "${MASTER_LOG}"
  echo "--------------------------------------------------" | tee -a "${MASTER_LOG}"

  if experiment_done_for_all_seeds "${exp_name}" "${SEEDS}"; then
    echo "[SKIP] ${exp_name} already has summary.json for all requested seeds: ${SEEDS}" | tee -a "${MASTER_LOG}"
    echo "${cfg}" >> "${SUCCESS_LOG}"
    return 0
  fi

  local run_log="${LOG_DIR}/${exp_name}_${RUN_TS}.log"

  echo "[RUN ] bash scripts/train_fno_seeds.sh ${cfg} ${DEVICE} \"${SEEDS}\"" | tee -a "${MASTER_LOG}"
  bash scripts/train_fno_seeds.sh "${cfg}" "${DEVICE}" "${SEEDS}" 2>&1 | tee "${run_log}"
  local status=${PIPESTATUS[0]}

  if [[ ${status} -ne 0 ]]; then
    echo "[FAIL] ${exp_name} (exit=${status})" | tee -a "${MASTER_LOG}"
    echo "${cfg}" >> "${FAIL_LOG}"
    return ${status}
  fi

  # Aggregate again explicitly for safety
  python scripts/aggregate_summaries.py --experiment-dir "checkpoints/${exp_name}" >> "${MASTER_LOG}" 2>&1
  local agg_status=$?

  if [[ ${agg_status} -ne 0 ]]; then
    echo "[WARN] training succeeded but aggregation failed for ${exp_name}" | tee -a "${MASTER_LOG}"
    echo "${cfg}" >> "${FAIL_LOG}"
    return ${agg_status}
  fi

  echo "[ OK ] ${exp_name}" | tee -a "${MASTER_LOG}"
  echo "${cfg}" >> "${SUCCESS_LOG}"
  return 0
}

run_group() {
  local group_name="$1"
  shift
  local configs=("$@")

  echo "" | tee -a "${MASTER_LOG}"
  echo "==================================================" | tee -a "${MASTER_LOG}"
  echo "GROUP: ${group_name}" | tee -a "${MASTER_LOG}"
  echo "==================================================" | tee -a "${MASTER_LOG}"

  local cfg
  for cfg in "${configs[@]}"; do
    run_one_config "${cfg}" || true
  done
}

# ---------------------------------------
# Config groups
# ---------------------------------------

BURGERS_RESOLUTION_CONFIGS=(
  "configs/resolution/burgers_fno_base_r512.yaml"
  "configs/resolution/burgers_fno_base_r1024.yaml"
  "configs/resolution/burgers_fno_base_r2048.yaml"
  "configs/resolution/burgers_fno_base_r4096.yaml"
)

DARCY_RESOLUTION_CONFIGS=(
  "configs/resolution/darcy_fno_base_r85.yaml"
  "configs/resolution/darcy_fno_base_r141.yaml"
  "configs/resolution/darcy_fno_base_r211.yaml"
  "configs/resolution/darcy_fno_base_r281.yaml"
)

BURGERS_ABLATION_CONFIGS=(
  "configs/ablations/burgers/burgers_fno_base_modes12.yaml"
  "configs/ablations/burgers/burgers_fno_base_modes16.yaml"
  "configs/ablations/burgers/burgers_fno_base_modes32.yaml"
  "configs/ablations/burgers/burgers_fno_base_nocoords.yaml"
  "configs/ablations/burgers/burgers_fno_base_pad0.yaml"
  "configs/ablations/burgers/burgers_fno_base_pad40.yaml"
)

DARCY_ABLATION_CONFIGS=(
  "configs/ablations/darcy/darcy_fno_base_modes12.yaml"
  "configs/ablations/darcy/darcy_fno_base_modes24.yaml"
  "configs/ablations/darcy/darcy_fno_base_nocoords.yaml"
  "configs/ablations/darcy/darcy_fno_base_pad0.yaml"
  "configs/ablations/darcy/darcy_fno_base_pad15.yaml"
)

# ---------------------------------------
# Pre-flight checks
# ---------------------------------------

NEEDED_FILES=(
  "scripts/train_fno_seeds.sh"
  "scripts/aggregate_summaries.py"
  "configs/resolution/burgers_fno_base_r512.yaml"
  "configs/resolution/burgers_fno_base_r1024.yaml"
  "configs/resolution/burgers_fno_base_r2048.yaml"
  "configs/resolution/burgers_fno_base_r4096.yaml"
  "configs/resolution/darcy_fno_base_r85.yaml"
  "configs/resolution/darcy_fno_base_r141.yaml"
  "configs/resolution/darcy_fno_base_r211.yaml"
  "configs/resolution/darcy_fno_base_r281.yaml"
  "configs/ablations/burgers/burgers_fno_base_modes12.yaml"
  "configs/ablations/burgers/burgers_fno_base_modes16.yaml"
  "configs/ablations/burgers/burgers_fno_base_modes32.yaml"
  "configs/ablations/burgers/burgers_fno_base_nocoords.yaml"
  "configs/ablations/burgers/burgers_fno_base_pad0.yaml"
  "configs/ablations/burgers/burgers_fno_base_pad40.yaml"
  "configs/ablations/darcy/darcy_fno_base_modes12.yaml"
  "configs/ablations/darcy/darcy_fno_base_modes24.yaml"
  "configs/ablations/darcy/darcy_fno_base_nocoords.yaml"
  "configs/ablations/darcy/darcy_fno_base_pad0.yaml"
  "configs/ablations/darcy/darcy_fno_base_pad15.yaml"
)

for f in "${NEEDED_FILES[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing required file: ${f}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
done

# ---------------------------------------
# Execution order
# ---------------------------------------

# 1) Resolution sweep first
# This is higher-priority for the paper's scaling/deployability claims.
run_group "BURGERS RESOLUTION SWEEP" "${BURGERS_RESOLUTION_CONFIGS[@]}"
run_group "DARCY RESOLUTION SWEEP" "${DARCY_RESOLUTION_CONFIGS[@]}"

# 2) Then ablations
run_group "BURGERS ABLATIONS" "${BURGERS_ABLATION_CONFIGS[@]}"
run_group "DARCY ABLATIONS" "${DARCY_ABLATION_CONFIGS[@]}"

echo "" | tee -a "${MASTER_LOG}"
echo "==================================================" | tee -a "${MASTER_LOG}"
echo "Finished : $(date)" | tee -a "${MASTER_LOG}"
echo "Master log   : ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "Success list : ${SUCCESS_LOG}" | tee -a "${MASTER_LOG}"
echo "Fail list    : ${FAIL_LOG}" | tee -a "${MASTER_LOG}"
echo "==================================================" | tee -a "${MASTER_LOG}"

echo ""
echo "========== SUCCESS CONFIGS =========="
cat "${SUCCESS_LOG}" 2>/dev/null || true

echo ""
echo "========== FAILED CONFIGS =========="
cat "${FAIL_LOG}" 2>/dev/null || true