#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

CONFIGS=(
  configs/deeponet/burgers_deeponet_small.yaml
  configs/deeponet/burgers_deeponet_base.yaml
  configs/deeponet/burgers_deeponet_large.yaml
  configs/deeponet/darcy_deeponet_small.yaml
  configs/deeponet/darcy_deeponet_base.yaml
  configs/deeponet/darcy_deeponet_large.yaml
)

SEEDS=(0 1 2 3 4)

SUCCESS="logs/train_deeponet_seeds_${STAMP}_success.txt"
FAIL="logs/train_deeponet_seeds_${STAMP}_fail.txt"
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
