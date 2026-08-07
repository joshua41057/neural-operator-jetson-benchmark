#!/usr/bin/env bash
set -u

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson_tf32_native_only.py"

SWEEP_ROOT="$ROOT/inference_runs/virso_missing_tf32_native_$(date +%Y%m%d_%H%M%S)"
REG="$SWEEP_ROOT/virso_missing_tf32_native_registry.csv"
mkdir -p "$SWEEP_ROOT"

echo "run_ts,path,precision_mode,model_path,num_layers,width,max_mode,spectral,spatial,exit_status" > "$REG"

run_one () {
  local PATH_NAME="$1"
  local MODEL_PATH="$2"
  local NUM_LAYERS="$3"
  local WIDTH="$4"
  local MAX_MODE="$5"
  local SPECTRAL="$6"
  local SPATIAL="$7"
  local PREC="$8"

  local RUN_TS="virso_${PATH_NAME}_${PREC}_$(date +%Y%m%d_%H%M%S)"
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
  echo "PATH=$PATH_NAME"
  echo "PRECISION_MODE=$PREC"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "NUM_LAYERS=$NUM_LAYERS WIDTH=$WIDTH MAX_MODE=$MAX_MODE SPECTRAL=$SPECTRAL SPATIAL=$SPATIAL"
  echo "================================================================================"

  set +e
  RUN_TS="$RUN_TS" \
  RUN_DIR="$RUN_DIR" \
  OUT_DIR="$OUT_DIR" \
  LOG_DIR="$LOG_DIR" \
  REPORT_DIR="$REPORT_DIR" \
  EDGE_CSV="$EDGE_CSV" \
  MODEL_PATH="$MODEL_PATH" \
  PRECISION_MODE="$PREC" \
  MONITOR_CMD="none" \
  MONITOR_INTERVAL_MS="200" \
  PROFILE_FLOPS="0" \
  PROFILE_WARMUP="2" \
  PROFILE_ITERS="1" \
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
  echo "${RUN_TS},${PATH_NAME},${PREC},${MODEL_PATH},${NUM_LAYERS},${WIDTH},${MAX_MODE},${SPECTRAL},${SPATIAL},${STATUS}" >> "$REG"

  echo "EXIT_STATUS=$STATUS"

  if [ -f "$EDGE_CSV" ]; then
    echo "EDGE_SUMMARY=$EDGE_CSV"
    grep -E "^(precision_mode|tf32_matmul_allowed|tf32_cudnn_allowed|amp_mode|amp_resolved_dtype|p50_latency_ms|p95_latency_ms|latency_s_per_iteration|avg_total_loss)," "$EDGE_CSV" || true
  else
    echo "NO_EDGE_SUMMARY"
    echo "--- STDERR tail ---"
    tail -120 "$LOG_DIR/stderr_${RUN_TS}.log" || true
    echo "--- STDOUT tail ---"
    tail -120 "$LOG_DIR/stdout_${RUN_TS}.log" || true
  fi

  sync
  sleep 10
}

# Missing mode 1: TF32
run_one "full"     "$ROOT/sp2gno_final.pth"              10 48 64 1 1 "tf32"
run_one "spectral" "$HOME/VirSO/best_model_spectral.pth" 10 48 64 1 0 "tf32"
run_one "layer2"   "$HOME/VirSO/best_model_2_layer.pth"   2 48 40 1 1 "tf32"

# Missing mode 2: native FP16
run_one "full"     "$ROOT/sp2gno_final.pth"              10 48 64 1 1 "fp16_native"
run_one "spectral" "$HOME/VirSO/best_model_spectral.pth" 10 48 64 1 0 "fp16_native"
run_one "layer2"   "$HOME/VirSO/best_model_2_layer.pth"   2 48 40 1 1 "fp16_native"

echo
echo "DONE"
echo "SWEEP_ROOT=$SWEEP_ROOT"
echo "REGISTRY=$REG"
cat "$REG"
