#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

CONFIGS=(
  configs/deeponet/resolution/burgers_deeponet_base_r512.yaml
  configs/deeponet/resolution/burgers_deeponet_base_r1024.yaml
  configs/deeponet/resolution/burgers_deeponet_base_r2048.yaml
  configs/deeponet/resolution/burgers_deeponet_base_r4096.yaml
  configs/deeponet/resolution/darcy_deeponet_base_r85.yaml
  configs/deeponet/resolution/darcy_deeponet_base_r141.yaml
  configs/deeponet/resolution/darcy_deeponet_base_r211.yaml
  configs/deeponet/resolution/darcy_deeponet_base_r281.yaml
)

SEEDS=(0 1 2)

SUCCESS="logs/train_deeponet_resolution_${STAMP}_success.txt"
FAIL="logs/train_deeponet_resolution_${STAMP}_fail.txt"
: > "$SUCCESS"
: > "$FAIL"

for cfg in "${CONFIGS[@]}"; do
  exp="$(basename "$cfg" .yaml)"
  for seed in "${SEEDS[@]}"; do
    log="logs/${exp}_seed${seed}_${STAMP}.log"
    echo "=== RUN $exp seed=$seed ===" | tee "$log"
    if python -m src.train.train_deeponet --config "$cfg" --seed "$seed" --device "$DEVICE" 2>&1 | tee -a "$log"; then
      echo "$exp,seed${seed}" >> "$SUCCESS"
    else
      echo "$exp,seed${seed}" >> "$FAIL"
    fi
  done
done

echo "Success list: $SUCCESS"
echo "Fail list:    $FAIL"
