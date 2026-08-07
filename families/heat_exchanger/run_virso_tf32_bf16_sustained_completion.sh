#!/usr/bin/env bash
set -u

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"

SCRIPT_BF16="$ROOT/run_virso_inference_jetson.py"
SCRIPT_TF32="$ROOT/run_virso_inference_jetson_tf32_native_only.py"

PARSE="$ROOT/scripts/parse_tegrastats.py"
MERGE="$ROOT/scripts/merge_reports.py"
REG="$ROOT/inference_runs/virso_tf32_bf16_sustained_completion_registry.csv"

mkdir -p "$ROOT/inference_runs"

if [ ! -f "$REG" ]; then
  echo "run_ts,case,precision,model_path,num_layers,width,max_mode,spectral,spatial,exit_status" > "$REG"
fi

run_one () {
  local CASE="$1"        # full / spectral / layer2
  local PREC="$2"        # tf32 / bf16
  local REP="$3"
  local MODEL_PATH="$4"
  local NUM_LAYERS="$5"
  local WIDTH="$6"
  local MAX_MODE="$7"
  local SPECTRAL="$8"
  local SPATIAL="$9"

  local RUN_TS="completion_${CASE}_${PREC}_r${REP}"
  local RUN_DIR="$ROOT/inference_runs/$RUN_TS"
  local OUT_DIR="$RUN_DIR/outputs"
  local LOG_DIR="$RUN_DIR/logs"
  local REPORT_DIR="$RUN_DIR/reports"

  local TEGRA_LOG="$LOG_DIR/tegrastats_${RUN_TS}.log"
  local POWER_CSV="$REPORT_DIR/jetson_power_summary_${RUN_TS}.csv"
  local EDGE_CSV="$REPORT_DIR/virso_edge_summary_${RUN_TS}.csv"
  local FINAL_CSV="$REPORT_DIR/jetson_reviewer_report_${RUN_TS}.csv"
  local STATUS_TXT="$LOG_DIR/run_status_${RUN_TS}.txt"

  if [ -d "$RUN_DIR" ]; then
    echo "SKIP existing: $RUN_DIR"
    return 0
  fi

  mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

  echo
  echo "================================================================================"
  echo "RUN_TS=$RUN_TS"
  echo "CASE=$CASE PREC=$PREC REP=$REP"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "NUM_LAYERS=$NUM_LAYERS WIDTH=$WIDTH MAX_MODE=$MAX_MODE SPECTRAL=$SPECTRAL SPATIAL=$SPATIAL"
  echo "================================================================================"

  {
    echo "RUN_TS=$RUN_TS"
    echo "CASE=$CASE"
    echo "PRECISION=$PREC"
    echo "ROOT=$ROOT"
    echo "MODEL_PATH=$MODEL_PATH"
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

  if [ "$PREC" = "tf32" ]; then
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
    PRECISION_MODE="tf32" \
    NUM_LAYERS="$NUM_LAYERS" \
    WIDTH="$WIDTH" \
    MAX_MODE="$MAX_MODE" \
    K_NEIGHBORS="30" \
    EMBED="1" \
    SPECTRAL="$SPECTRAL" \
    SPATIAL="$SPATIAL" \
    COLLAB_SKIP="1" \
    SPECTRAL_SKIP="1" \
    "$PY" "$SCRIPT_TF32" > "$LOG_DIR/stdout_${RUN_TS}.log" 2> "$LOG_DIR/stderr_${RUN_TS}.log"
    STATUS=$?
  else
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
    AMP_MODE="bf16" \
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
    "$PY" "$SCRIPT_BF16" > "$LOG_DIR/stdout_${RUN_TS}.log" 2> "$LOG_DIR/stderr_${RUN_TS}.log"
    STATUS=$?
  fi

  set -e

  echo "$STATUS" > "$STATUS_TXT"
  echo "${RUN_TS},${CASE},${PREC},${MODEL_PATH},${NUM_LAYERS},${WIDTH},${MAX_MODE},${SPECTRAL},${SPATIAL},${STATUS}" >> "$REG"

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

  if [ "$STATUS" -ne 0 ]; then
    echo "[FAIL] $RUN_TS"
    tail -100 "$LOG_DIR/stderr_${RUN_TS}.log" || true
  else
    echo "[OK] $RUN_TS"
  fi

  sync
  sleep 20
}

# Full spectral-spatial: n = 5
for r in 1 2 3 4 5; do
  run_one "full" "tf32" "$r" "$ROOT/sp2gno_final.pth" 10 48 64 1 1
done
for r in 1 2 3 4 5; do
  run_one "full" "bf16" "$r" "$ROOT/sp2gno_final.pth" 10 48 64 1 1
done

# Spectral-only: n = 3
for r in 1 2 3; do
  run_one "spectral" "tf32" "$r" "$HOME/VirSO/best_model_spectral.pth" 10 48 64 1 0
done
for r in 1 2 3; do
  run_one "spectral" "bf16" "$r" "$HOME/VirSO/best_model_spectral.pth" 10 48 64 1 0
done

# Two-layer spectral-spatial: n = 3
for r in 1 2 3; do
  run_one "layer2" "tf32" "$r" "$HOME/VirSO/best_model_2_layer.pth" 2 48 40 1 1
done
for r in 1 2 3; do
  run_one "layer2" "bf16" "$r" "$HOME/VirSO/best_model_2_layer.pth" 2 48 40 1 1
done

echo
echo "DONE:"
cat "$REG"
