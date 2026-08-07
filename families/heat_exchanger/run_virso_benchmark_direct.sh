#!/usr/bin/env bash
set -u

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"
PARSE="$ROOT/scripts/parse_tegrastats.py"
MERGE="$ROOT/scripts/merge_reports.py"

DIRECT_REG="$ROOT/inference_runs/virso_direct_benchmark_registry.csv"

mkdir -p "$ROOT/inference_runs"

echo "run_ts,case,model_path,amp_mode,num_layers,width,max_mode,spectral,spatial,exit_status" > "$DIRECT_REG"

run_one() {
  local CASE="$1"
  local MODEL_PATH="$2"
  local AMP_MODE="$3"
  local NUM_LAYERS="$4"
  local WIDTH="$5"
  local MAX_MODE="$6"
  local SPECTRAL="$7"
  local SPATIAL="$8"
  local REP="$9"

  local RUN_TS="bench2_${CASE}_r${REP}"
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
  echo "AMP_MODE=$AMP_MODE"
  echo "NUM_LAYERS=$NUM_LAYERS WIDTH=$WIDTH MAX_MODE=$MAX_MODE SPECTRAL=$SPECTRAL SPATIAL=$SPATIAL"
  echo "================================================================================"

  {
    echo "RUN_TS=$RUN_TS"
    echo "CASE=$CASE"
    echo "ROOT=$ROOT"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "AMP_MODE=$AMP_MODE"
    echo "NUM_LAYERS=$NUM_LAYERS"
    echo "WIDTH=$WIDTH"
    echo "MAX_MODE=$MAX_MODE"
    echo "K_NEIGHBORS=30"
    echo "EMBED=1"
    echo "SPECTRAL=$SPECTRAL"
    echo "SPATIAL=$SPATIAL"
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
  AMP_MODE="$AMP_MODE" \
  AMP_FALLBACK="fp16" \
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
  echo "${RUN_TS},${CASE},${MODEL_PATH},${AMP_MODE},${NUM_LAYERS},${WIDTH},${MAX_MODE},${SPECTRAL},${SPATIAL},${STATUS}" >> "$DIRECT_REG"

  echo "EXIT_STATUS=$STATUS"

  if [ -f "$TEGRA_LOG" ]; then
    "$PY" "$PARSE" "$TEGRA_LOG" "$POWER_CSV" || true
  else
    echo "WARNING: missing tegrastats log: $TEGRA_LOG"
  fi

  if [ -f "$EDGE_CSV" ] && [ -f "$POWER_CSV" ]; then
    "$PY" "$MERGE" "$EDGE_CSV" "$POWER_CSV" "$FINAL_CSV" "$STATUS_TXT" || true
  else
    echo "WARNING: missing EDGE_CSV or POWER_CSV"
    echo "EDGE_CSV=$EDGE_CSV"
    echo "POWER_CSV=$POWER_CSV"
  fi

  echo "RUN_DIR=$RUN_DIR"
  find "$RUN_DIR" -maxdepth 3 -type f | sort

  sync
  sleep 20
}

# Feasibility probe only. If this OOMs, keep it as a failure record.
run_one "full_fp32_feas" "$ROOT/sp2gno_final.pth" "off" 10 48 64 1 1 1

# Main measured deployment rows.
for r in 1 2 3; do
  run_one "full_ampfp16" "$ROOT/sp2gno_final.pth" "fp16" 10 48 64 1 1 "$r"
done

for r in 1 2 3; do
  run_one "spectral_fp32" "$HOME/VirSO/best_model_spectral.pth" "off" 10 48 64 1 0 "$r"
done

for r in 1 2 3; do
  run_one "layer2_fp32" "$HOME/VirSO/best_model_2_layer.pth" "off" 2 48 40 1 1 "$r"
done

echo
echo "DONE. Direct registry:"
cat "$DIRECT_REG"
