#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"

PLAN="${PLAN:-results/artifacts/deeponet_profile_plan.csv}"
OUTDIR="results/profiles/deeponet_nsys"
mkdir -p "$OUTDIR" logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_deeponet_nsys_${STAMP}.log"
SUCCESS="logs/run_deeponet_nsys_${STAMP}_success.csv"
FAIL="logs/run_deeponet_nsys_${STAMP}_fail.csv"

echo "profile_id,status,rep,sqlite,json" > "$SUCCESS"
echo "profile_id,status,error" > "$FAIL"

NSYS_WARMUP="${NSYS_WARMUP:-20}"
NSYS_ITERS="${NSYS_ITERS:-80}"

echo "PLAN=$PLAN" | tee "$LOG"
echo "NSYS_WARMUP=$NSYS_WARMUP NSYS_ITERS=$NSYS_ITERS" | tee -a "$LOG"

tail -n +2 "$PLAN" | while IFS=, read -r profile_id case_role precision checkpoint torchscript input_bank main_or_appendix; do
  echo | tee -a "$LOG"
  echo "=== NSYS profile_id=$profile_id role=$case_role precision=$precision class=$main_or_appendix ===" | tee -a "$LOG"

  outbase="$OUTDIR/${profile_id}"
  json_out="$OUTDIR/${profile_id}_forward.json"

  rm -f "${outbase}.nsys-rep" "${outbase}.sqlite" "${outbase}.qdstrm" "$json_out"

  set +e
  nsys profile \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --cuda-memory-usage=false \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    --stop-on-exit=true \
    --output="$outbase" \
    python -m src.eval.profile_forward_nvtx \
      --mode torchscript \
      --checkpoint "$checkpoint" \
      --torchscript "$torchscript" \
      --input-bank "$input_bank" \
      --precision "$precision" \
      --batch-size 1 \
      --warmup "$NSYS_WARMUP" \
      --profile-iters "$NSYS_ITERS" \
      --device cuda \
      --result-json "$json_out" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  rep="${outbase}.nsys-rep"
  sqlite="${outbase}.sqlite"

  if [[ "$rc" -ne 0 ]]; then
    echo "${profile_id},fail,nsys_profile_failed_rc_${rc}" >> "$FAIL"
    continue
  fi

  if [[ ! -f "$rep" ]]; then
    echo "${profile_id},fail,missing_nsys_rep" >> "$FAIL"
    continue
  fi

  if [[ ! -f "$sqlite" ]]; then
    nsys export \
      --type sqlite \
      --force-overwrite=true \
      --output="$sqlite" \
      "$rep" 2>&1 | tee -a "$LOG" || true
  fi

  if [[ ! -f "$json_out" ]]; then
    echo "${profile_id},fail,missing_forward_json" >> "$FAIL"
    continue
  fi

  echo "${profile_id},success,${rep},${sqlite},${json_out}" >> "$SUCCESS"
done

echo "NSYS done"
echo "SUCCESS=$SUCCESS"
echo "FAIL=$FAIL"
