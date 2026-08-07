#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: bash scripts/train_fno_seeds.sh <config> [device] [seeds]}
DEVICE=${2:-cuda}
SEEDS=${3:-"0 1 2"}

echo "Config: ${CONFIG}"
echo "Device: ${DEVICE}"
echo "Seeds: ${SEEDS}"

EXPERIMENT_NAME=$(
python - "${CONFIG}" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['name'])
PY
)

for SEED in ${SEEDS}; do
  SUMMARY_PATH="checkpoints/${EXPERIMENT_NAME}/seed${SEED}/summary.json"

  if [[ -f "${SUMMARY_PATH}" ]]; then
    echo "=== Skipping seed ${SEED} (found ${SUMMARY_PATH}) ==="
    continue
  fi

  echo "=== Training seed ${SEED} ==="
  python -m src.train.train_fno --config "${CONFIG}" --seed "${SEED}" --device "${DEVICE}"
done

python scripts/aggregate_summaries.py --experiment-dir "checkpoints/${EXPERIMENT_NAME}"