#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

IN_DIR="results/profiles/deeponet_nsys"
OUT_DIR="results/profiles/deeponet_nsys_stats"
mkdir -p "$OUT_DIR" logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/extract_deeponet_nsys_stats_${STAMP}.log"
STATUS="logs/extract_deeponet_nsys_stats_${STAMP}_status.csv"

echo "profile_id,report,status,output,error" > "$STATUS"

for rep in "$IN_DIR"/*.nsys-rep; do
  base="$(basename "$rep" .nsys-rep)"
  echo
  echo "=== Extract NSYS stats: $base ===" | tee -a "$LOG"

  out="$OUT_DIR/${base}_cuda_gpu_kern_sum"
  echo "--- cuda_gpu_kern_sum: $base ---" | tee -a "$LOG"
  if nsys stats \
      --force-export=true \
      --report cuda_gpu_kern_sum \
      --format csv \
      --output "$out" \
      "$rep" >> "$LOG" 2>&1; then
    echo "$base,cuda_gpu_kern_sum,success,$out," >> "$STATUS"
  else
    echo "$base,cuda_gpu_kern_sum,fail,$out,nsys_stats_failed" >> "$STATUS"
  fi

  out="$OUT_DIR/${base}_cuda_api_sum"
  echo "--- cuda_api_sum: $base ---" | tee -a "$LOG"
  if nsys stats \
      --force-export=true \
      --report cuda_api_sum \
      --format csv \
      --output "$out" \
      "$rep" >> "$LOG" 2>&1; then
    echo "$base,cuda_api_sum,success,$out," >> "$STATUS"
  else
    echo "$base,cuda_api_sum,fail,$out,nsys_stats_failed" >> "$STATUS"
  fi

  out="$OUT_DIR/${base}_nvtx_sum"
  echo "--- nvtx_sum: $base ---" | tee -a "$LOG"
  if nsys stats \
      --force-export=true \
      --report nvtx_sum \
      --format csv \
      --output "$out" \
      "$rep" >> "$LOG" 2>&1; then
    echo "$base,nvtx_sum,success,$out," >> "$STATUS"
  else
    echo "$base,nvtx_sum,fail,$out,nsys_stats_failed" >> "$STATUS"
  fi

  out="$OUT_DIR/${base}_forward_cuda_gpu_kern_sum"
  echo "--- forward cuda_gpu_kern_sum: $base ---" | tee -a "$LOG"
  if nsys stats \
      --force-export=true \
      --filter-nvtx deeponet_profile_forward \
      --report cuda_gpu_kern_sum \
      --format csv \
      --output "$out" \
      "$rep" >> "$LOG" 2>&1; then
    echo "$base,forward_cuda_gpu_kern_sum,success,$out," >> "$STATUS"
  else
    echo "$base,forward_cuda_gpu_kern_sum,fail,$out,filter_or_stats_failed" >> "$STATUS"
  fi
done

echo "Wrote status: $STATUS"
echo "Wrote stats to: $OUT_DIR"
