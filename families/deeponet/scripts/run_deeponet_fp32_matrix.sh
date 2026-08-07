#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs"
RESULTS_DIR="results/jetson_deeponet_fp32"
mkdir -p "$LOGDIR" "$RESULTS_DIR" results/artifacts

LOG="$LOGDIR/run_deeponet_fp32_matrix_${STAMP}.log"
FAIL="$LOGDIR/run_deeponet_fp32_matrix_${STAMP}_fail.csv"
SUCCESS="$LOGDIR/run_deeponet_fp32_matrix_${STAMP}_success.csv"

echo "experiment_name,backend,precision,result_tag" > "$SUCCESS"
echo "experiment_name,backend,precision,result_tag,exit_code" > "$FAIL"

{
  echo "=== DeepONet FP32 matrix ==="
  date
  pwd
  python - <<'PY'
import torch, src
print("src", src.__file__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
} | tee "$LOG"

python - <<'PY' | while IFS=',' read -r exp ckpt ts bank backend tag; do
import csv
from pathlib import Path

manifest = Path("manifests/deeponet_jetson_manifest.csv")
for r in csv.DictReader(manifest.open()):
    exp = r["experiment_name"]
    ckpt = r["checkpoint_path"]
    ts = r["torchscript_path"]
    bank = r["input_bank_path"]
    for backend in ["eager", "torchscript"]:
        tag = f"{exp}_{backend}_fp32"
        print(",".join([exp, ckpt, ts, bank, backend, tag]))
PY
  echo
  echo "=== RUN $tag ===" | tee -a "$LOG"

  cmd=(
    python -m src.eval.benchmark_inference
    --mode "$backend"
    --checkpoint "$ckpt"
    --input-bank "$bank"
    --precision fp32
    --batch-size 1
    --num-warmup 30
    --num-iters 100
    --device cuda
    --results-dir "$RESULTS_DIR"
    --result-tag "$tag"
  )

  if [[ "$backend" == "torchscript" ]]; then
    cmd+=(--torchscript "$ts")
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "$exp,$backend,fp32,$tag" >> "$SUCCESS"
  else
    echo "$exp,$backend,fp32,$tag,$rc" >> "$FAIL"
  fi
done

echo "Done. Log: $LOG"
echo "Success: $SUCCESS"
echo "Fail: $FAIL"
echo "JSON count:"
find "$RESULTS_DIR" -name '*.json' | wc -l
