#!/usr/bin/env bash
set -u

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"
PROFILE_ROOT="$ROOT/inference_runs/virso_profiles_existing_path"
DIRECT_REG="$PROFILE_ROOT/virso_profile_registry.csv"

mkdir -p "$PROFILE_ROOT"
cd "$ROOT"

echo "run_ts,case,model_path,amp_mode,num_layers,width,max_mode,spectral,spatial,exit_status,profile_mode" > "$DIRECT_REG"

run_case () {
  local RUN_TS="$1"
  local CASE="$2"
  local MODEL_PATH="$3"
  local AMP_MODE="$4"
  local NUM_LAYERS="$5"
  local WIDTH="$6"
  local MAX_MODE="$7"
  local SPECTRAL="$8"
  local SPATIAL="$9"
  local COLLAB_SKIP="${10}"

  local RUN_DIR="$PROFILE_ROOT/$RUN_TS"
  local OUT_DIR="$RUN_DIR/outputs"
  local LOG_DIR="$RUN_DIR/logs"
  local REPORT_DIR="$RUN_DIR/reports"
  local EDGE_CSV="$REPORT_DIR/virso_edge_summary_${RUN_TS}.csv"
  local STATUS_TXT="$LOG_DIR/run_status_${RUN_TS}.txt"

  mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

  echo
  echo "================================================================================"
  echo "PROFILE RUN_TS=$RUN_TS"
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
    echo "COLLAB_SKIP=$COLLAB_SKIP"
    echo "SPECTRAL_SKIP=1"
    echo "MONITOR_CMD=none"
    echo "PROFILE_FLOPS=1"
    echo "PROFILE_WARMUP=2"
    echo "PROFILE_ITERS=1"
    date
    uname -a
    "$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
  } > "$RUN_DIR/provenance.txt"

  set +e

  if command -v nsys >/dev/null 2>&1; then
    echo "Using Nsight Systems for $RUN_TS"
    RUN_TS="$RUN_TS" \
    RUN_DIR="$RUN_DIR" \
    OUT_DIR="$OUT_DIR" \
    LOG_DIR="$LOG_DIR" \
    REPORT_DIR="$REPORT_DIR" \
    EDGE_CSV="$EDGE_CSV" \
    MODEL_PATH="$MODEL_PATH" \
    MONITOR_CMD="none" \
    MONITOR_INTERVAL_MS="200" \
    PROFILE_FLOPS="1" \
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
    COLLAB_SKIP="$COLLAB_SKIP" \
    SPECTRAL_SKIP="1" \
    nsys profile \
      --force-overwrite=true \
      --trace=cuda,nvtx,osrt,cublas,cudnn \
      --sample=none \
      --cpuctxsw=none \
      -o "$RUN_DIR/nsys_${RUN_TS}" \
      "$PY" "$SCRIPT" \
      > "$LOG_DIR/stdout_${RUN_TS}.log" \
      2> "$LOG_DIR/stderr_${RUN_TS}.log"
    STATUS=$?
    PROFILE_MODE="nsys+script_profile"
  else
    echo "nsys not found; using script PROFILE_FLOPS only for $RUN_TS"
    RUN_TS="$RUN_TS" \
    RUN_DIR="$RUN_DIR" \
    OUT_DIR="$OUT_DIR" \
    LOG_DIR="$LOG_DIR" \
    REPORT_DIR="$REPORT_DIR" \
    EDGE_CSV="$EDGE_CSV" \
    MODEL_PATH="$MODEL_PATH" \
    MONITOR_CMD="none" \
    MONITOR_INTERVAL_MS="200" \
    PROFILE_FLOPS="1" \
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
    COLLAB_SKIP="$COLLAB_SKIP" \
    SPECTRAL_SKIP="1" \
    "$PY" "$SCRIPT" \
      > "$LOG_DIR/stdout_${RUN_TS}.log" \
      2> "$LOG_DIR/stderr_${RUN_TS}.log"
    STATUS=$?
    PROFILE_MODE="script_profile_only"
  fi

  set -e

  echo "$STATUS" > "$STATUS_TXT"
  echo "${RUN_TS},${CASE},${MODEL_PATH},${AMP_MODE},${NUM_LAYERS},${WIDTH},${MAX_MODE},${SPECTRAL},${SPATIAL},${STATUS},${PROFILE_MODE}" >> "$DIRECT_REG"

  echo "EXIT_STATUS=$STATUS"
  find "$RUN_DIR" -maxdepth 3 -type f | sort

  sync
  sleep 20
}

# Required representative profiles.
run_case "profile_full_fp32_existing_path" \
  "full_fp32" \
  "$ROOT/sp2gno_final.pth" \
  "off" \
  10 48 64 1 1 1

run_case "profile_spectral_fp32_existing_path" \
  "spectral_fp32" \
  "$HOME/VirSO/best_model_spectral.pth" \
  "off" \
  10 48 64 1 0 0

run_case "profile_layer2_fp32_existing_path" \
  "layer2_fp32" \
  "$HOME/VirSO/best_model_2_layer.pth" \
  "off" \
  2 48 40 1 1 1

# Optional. Keep for appendix/provenance only unless clean and useful.
run_case "profile_full_ampfp16_existing_path_optional" \
  "full_ampfp16" \
  "$ROOT/sp2gno_final.pth" \
  "fp16" \
  10 48 64 1 1 1

echo
echo "DONE. Registry:"
cat "$DIRECT_REG"
