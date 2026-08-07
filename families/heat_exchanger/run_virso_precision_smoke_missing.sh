#!/usr/bin/env bash
set -u

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"

SWEEP_ROOT="$ROOT/inference_runs/virso_precision_smoke_$(date +%Y%m%d_%H%M%S)"
REG="$SWEEP_ROOT/virso_precision_smoke_registry.csv"
mkdir -p "$SWEEP_ROOT"

echo "run_ts,case,model_path,amp_mode,amp_fallback,num_layers,width,max_mode,spectral,spatial,exit_status" > "$REG"

run_one () {
  local CASE="$1"
  local MODEL_PATH="$2"
  local AMP_MODE="$3"
  local AMP_FALLBACK="$4"
  local NUM_LAYERS="$5"
  local WIDTH="$6"
  local MAX_MODE="$7"
  local SPECTRAL="$8"
  local SPATIAL="$9"

  local RUN_TS="precision_${CASE}_${AMP_MODE}_$(date +%Y%m%d_%H%M%S)"
  local RUN_DIR="$SWEEP_ROOT/$RUN_TS"
  local OUT_DIR="$RUN_DIR/outputs"
  local LOG_DIR="$RUN_DIR/logs"
  local REPORT_DIR="$RUN_DIR/reports"
  local EDGE_CSV="$REPORT_DIR/virso_edge_summary_${RUN_TS}.csv"
  local STATUS_TXT="$LOG_DIR/run_status_${RUN_TS}.txt"

  mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

  echo
  echo "================================================================================"
  echo "RUN_TS=$RUN_TS"
  echo "CASE=$CASE"
  echo "AMP_MODE=$AMP_MODE AMP_FALLBACK=$AMP_FALLBACK"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "LAYERS=$NUM_LAYERS WIDTH=$WIDTH MAX_MODE=$MAX_MODE SPECTRAL=$SPECTRAL SPATIAL=$SPATIAL"
  echo "================================================================================"

  {
    echo "RUN_TS=$RUN_TS"
    echo "CASE=$CASE"
    echo "ROOT=$ROOT"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "AMP_MODE=$AMP_MODE"
    echo "AMP_FALLBACK=$AMP_FALLBACK"
    echo "NUM_LAYERS=$NUM_LAYERS"
    echo "WIDTH=$WIDTH"
    echo "MAX_MODE=$MAX_MODE"
    echo "K_NEIGHBORS=30"
    echo "EMBED=1"
    echo "SPECTRAL=$SPECTRAL"
    echo "SPATIAL=$SPATIAL"
    echo "COLLAB_SKIP=1"
    echo "SPECTRAL_SKIP=1"
    echo "MONITOR_CMD=none"
    echo "PROFILE_FLOPS=0"
    date
    uname -a
  } > "$RUN_DIR/provenance.txt"

  set +e
  RUN_TS="$RUN_TS" \
  RUN_DIR="$RUN_DIR" \
  OUT_DIR="$OUT_DIR" \
  LOG_DIR="$LOG_DIR" \
  REPORT_DIR="$REPORT_DIR" \
  EDGE_CSV="$EDGE_CSV" \
  MODEL_PATH="$MODEL_PATH" \
  MONITOR_CMD="none" \
  MONITOR_INTERVAL_MS="200" \
  PROFILE_FLOPS="0" \
  PROFILE_WARMUP="2" \
  PROFILE_ITERS="1" \
  AMP_MODE="$AMP_MODE" \
  AMP_FALLBACK="$AMP_FALLBACK" \
  NUM_LAYERS="$NUM_LAYERS" \
  WIDTH="$WIDTH" \
  MAX_MODE="$MAX_MODE" \
  K_NEIGHBORS="30" \
  EMBED="1" \
  SPECTRAL="$SPECTRAL" \
  SPATIAL="$SPATIAL" \
  COLLAB_SKIP="1" \
  SPECTRAL_SKIP="1" \
  "$PY" "$SCRIPT" > "$LOG_DIR/stdout_${RUN_TS}.log" 2> "$LOG_DIR/stderr_${RUN_TS}.log"
  STATUS=$?
  set -e

  echo "$STATUS" > "$STATUS_TXT"
  echo "${RUN_TS},${CASE},${MODEL_PATH},${AMP_MODE},${AMP_FALLBACK},${NUM_LAYERS},${WIDTH},${MAX_MODE},${SPECTRAL},${SPATIAL},${STATUS}" >> "$REG"

  echo "EXIT_STATUS=$STATUS"
  if [ -f "$EDGE_CSV" ]; then
    echo "EDGE_SUMMARY=$EDGE_CSV"
    cat "$EDGE_CSV"
  else
    echo "NO_EDGE_SUMMARY"
    echo "--- STDERR tail ---"
    tail -80 "$LOG_DIR/stderr_${RUN_TS}.log" || true
    echo "--- STDOUT tail ---"
    tail -80 "$LOG_DIR/stdout_${RUN_TS}.log" || true
  fi

  sync
  sleep 10
}

# Missing VIRSO precision cases.
# BF16 is run with AMP_FALLBACK=off so it cannot silently become FP16.
run_one "full"     "$ROOT/sp2gno_final.pth"             "bf16" "off"  10 48 64 1 1
run_one "spectral" "$HOME/VirSO/best_model_spectral.pth" "fp16" "fp16" 10 48 64 1 0
run_one "spectral" "$HOME/VirSO/best_model_spectral.pth" "bf16" "off"  10 48 64 1 0
run_one "layer2"   "$HOME/VirSO/best_model_2_layer.pth"  "fp16" "fp16" 2  48 40 1 1
run_one "layer2"   "$HOME/VirSO/best_model_2_layer.pth"  "bf16" "off"  2  48 40 1 1

echo
echo "DONE"
echo "SWEEP_ROOT=$SWEEP_ROOT"
echo "REGISTRY=$REG"
cat "$REG"
