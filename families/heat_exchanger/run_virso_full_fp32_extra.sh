#!/usr/bin/env bash
set -u

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"
PARSE="$ROOT/scripts/parse_tegrastats.py"
MERGE="$ROOT/scripts/merge_reports.py"
DIRECT_REG="$ROOT/inference_runs/virso_direct_benchmark_registry.csv"

run_full_fp32 () {
  local REP="$1"
  local RUN_TS="bench2_full_fp32_r${REP}"
  local CASE="full_fp32"
  local MODEL_PATH="$ROOT/sp2gno_final.pth"
  local RUN_DIR="$ROOT/inference_runs/$RUN_TS"
  local OUT_DIR="$RUN_DIR/outputs"
  local LOG_DIR="$RUN_DIR/logs"
  local REPORT_DIR="$RUN_DIR/reports"

  local TEGRA_LOG="$LOG_DIR/tegrastats_${RUN_TS}.log"
  local POWER_CSV="$REPORT_DIR/jetson_power_summary_${RUN_TS}.csv"
  local EDGE_CSV="$REPORT_DIR/virso_edge_summary_${RUN_TS}.csv"
  local FINAL_CSV="$REPORT_DIR/jetson_reviewer_report_${RUN_TS}.csv"
  local STATUS_TXT="$LOG_DIR/run_status_${RUN_TS}.txt"

  mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

  echo
  echo "================================================================================"
  echo "RUN_TS=$RUN_TS"
  echo "CASE=$CASE"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "AMP_MODE=off"
  echo "NUM_LAYERS=10 WIDTH=48 MAX_MODE=64 SPECTRAL=1 SPATIAL=1"
  echo "================================================================================"

  {
    echo "RUN_TS=$RUN_TS"
    echo "CASE=$CASE"
    echo "ROOT=$ROOT"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "AMP_MODE=off"
    echo "NUM_LAYERS=10"
    echo "WIDTH=48"
    echo "MAX_MODE=64"
    echo "K_NEIGHBORS=30"
    echo "EMBED=1"
    echo "SPECTRAL=1"
    echo "SPATIAL=1"
    echo "COLLAB_SKIP=1"
    echo "SPECTRAL_SKIP=1"
    echo "MONITOR_CMD=tegrastats"
    echo "MONITOR_INTERVAL_MS=200"
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
  MONITOR_CMD="tegrastats" \
  MONITOR_INTERVAL_MS="200" \
  PROFILE_FLOPS="0" \
  PROFILE_WARMUP="2" \
  PROFILE_ITERS="1" \
  AMP_MODE="off" \
  AMP_FALLBACK="fp16" \
  NUM_LAYERS="10" \
  WIDTH="48" \
  MAX_MODE="64" \
  K_NEIGHBORS="30" \
  EMBED="1" \
  SPECTRAL="1" \
  SPATIAL="1" \
  COLLAB_SKIP="1" \
  SPECTRAL_SKIP="1" \
  "$PY" "$SCRIPT" > "$LOG_DIR/stdout_${RUN_TS}.log" 2> "$LOG_DIR/stderr_${RUN_TS}.log"
  STATUS=$?
  set -e

  echo "$STATUS" > "$STATUS_TXT"
  echo "${RUN_TS},${CASE},${MODEL_PATH},off,10,48,64,1,1,${STATUS}" >> "$DIRECT_REG"

  echo "EXIT_STATUS=$STATUS"

  if [ -f "$TEGRA_LOG" ]; then
    "$PY" "$PARSE" "$TEGRA_LOG" "$POWER_CSV" || true
  fi

  if [ -f "$EDGE_CSV" ] && [ -f "$POWER_CSV" ]; then
    "$PY" "$MERGE" "$EDGE_CSV" "$POWER_CSV" "$FINAL_CSV" "$STATUS_TXT" || true
  fi

  find "$RUN_DIR" -maxdepth 3 -type f | sort
  sync
  sleep 20
}

run_full_fp32 2
run_full_fp32 3
