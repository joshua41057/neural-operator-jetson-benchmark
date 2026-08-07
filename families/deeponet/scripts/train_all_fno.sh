#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda}
SEEDS=${2:-"0 1 2 3 4"}

CONFIGS=(
  configs/burgers_fno_small.yaml
  configs/burgers_fno_base.yaml
  configs/burgers_fno_large.yaml
  configs/darcy_fno_small.yaml
  configs/darcy_fno_base.yaml
  configs/darcy_fno_large.yaml
)

for CFG in "${CONFIGS[@]}"; do
  echo "########################################"
  echo "Running ${CFG}"
  echo "########################################"
  bash scripts/train_fno_seeds.sh "${CFG}" "${DEVICE}" "${SEEDS}"
done