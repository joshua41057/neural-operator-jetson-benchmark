#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"
PARSE="$ROOT/scripts/parse_tegrastats.py"
MERGE="$ROOT/scripts/merge_reports.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
OFFICIAL_ROOT="$ROOT/inference_runs/official_virso_cache_enabled_${STAMP}"
REG="$OFFICIAL_ROOT/official_virso_registry.csv"

mkdir -p "$OFFICIAL_ROOT"

echo "run_ts,case,model_path,amp_mode,tf32_mode,num_layers,width,max_mode,spectral,spatial,rep,exit_status" > "$REG"

run_one() {
  local CASE="$1"
  local MODEL_PATH="$2"
  local AMP_MODE="$3"
  local TF32_MODE="$4"
  local NUM_LAYERS="$5"
  local WIDTH="$6"
  local MAX_MODE="$7"
  local SPECTRAL="$8"
  local SPATIAL="$9"
  local REP="${10}"

  local RUN_TS="official_${CASE}_r${REP}"
  local RUN_DIR="$OFFICIAL_ROOT/$RUN_TS"
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
  echo "AMP_MODE=$AMP_MODE TF32_MODE=$TF32_MODE"
  echo "NUM_LAYERS=$NUM_LAYERS WIDTH=$WIDTH MAX_MODE=$MAX_MODE SPECTRAL=$SPECTRAL SPATIAL=$SPATIAL"
  echo "================================================================================"

  {
    echo "RUN_TS=$RUN_TS"
    echo "CASE=$CASE"
    echo "ROOT=$ROOT"
    echo "OFFICIAL_ROOT=$OFFICIAL_ROOT"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "AMP_MODE=$AMP_MODE"
    echo "TF32_MODE=$TF32_MODE"
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
    echo "VIRSO_ALLOCATOR_AUDIT=1"
    echo "PYTORCH_NO_CUDA_MEMORY_CACHING=UNSET"
    echo "MEM_AUDIT_ONLY=0"
    echo "SAVE_OUTPUTS=1"
    echo "PROFILE_FLOPS=0"
    date
    uname -a
  } > "$RUN_DIR/provenance.txt"

  unset PYTORCH_NO_CUDA_MEMORY_CACHING
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

  export RUN_TS="$RUN_TS"
  export RUN_DIR="$RUN_DIR"
  export OUT_DIR="$OUT_DIR"
  export LOG_DIR="$LOG_DIR"
  export REPORT_DIR="$REPORT_DIR"
  export EDGE_CSV="$EDGE_CSV"
  export MODEL_PATH="$MODEL_PATH"

  export MONITOR_CMD="tegrastats"
  export MONITOR_INTERVAL_MS="200"

  export VIRSO_ALLOCATOR_AUDIT="1"
  export MEM_AUDIT_ONLY="0"
  export SAVE_OUTPUTS="1"

  export PROFILE_FLOPS="0"
  export PROFILE_WARMUP="2"
  export PROFILE_ITERS="1"

  export TF32_MODE="$TF32_MODE"
  export AMP_MODE="$AMP_MODE"
  export AMP_FALLBACK="off"

  export NUM_LAYERS="$NUM_LAYERS"
  export WIDTH="$WIDTH"
  export MAX_MODE="$MAX_MODE"
  export K_NEIGHBORS="30"
  export EMBED="1"
  export SPECTRAL="$SPECTRAL"
  export SPATIAL="$SPATIAL"
  export COLLAB_SKIP="1"
  export SPECTRAL_SKIP="1"

  set +e
  "$PY" "$SCRIPT" > "$LOG_DIR/stdout_${RUN_TS}.log" 2> "$LOG_DIR/stderr_${RUN_TS}.log"
  STATUS=$?
  set -e

  echo "$STATUS" > "$STATUS_TXT"
  echo "${RUN_TS},${CASE},${MODEL_PATH},${AMP_MODE},${TF32_MODE},${NUM_LAYERS},${WIDTH},${MAX_MODE},${SPECTRAL},${SPATIAL},${REP},${STATUS}" >> "$REG"

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

# -------------------------------------------------------------------
# Official VIRSO cache-enabled sustained telemetry matrix.
# 3 variants × 4 executable precision modes × 3 reps = 36 runs.
# -------------------------------------------------------------------

for r in 1 2 3; do
  run_one "full_fp32" "$ROOT/sp2gno_final.pth" "off" "strict" 10 48 64 1 1 "$r"
done

for r in 1 2 3; do
  run_one "full_tf32" "$ROOT/sp2gno_final.pth" "off" "tf32" 10 48 64 1 1 "$r"
done

for r in 1 2 3; do
  run_one "full_bf16" "$ROOT/sp2gno_final.pth" "bf16" "strict" 10 48 64 1 1 "$r"
done

for r in 1 2 3; do
  run_one "full_fp16_autocast" "$ROOT/sp2gno_final.pth" "fp16" "strict" 10 48 64 1 1 "$r"
done

for r in 1 2 3; do
  run_one "spectral_fp32" "$HOME/VirSO/best_model_spectral.pth" "off" "strict" 10 48 64 1 0 "$r"
done

for r in 1 2 3; do
  run_one "spectral_tf32" "$HOME/VirSO/best_model_spectral.pth" "off" "tf32" 10 48 64 1 0 "$r"
done

for r in 1 2 3; do
  run_one "spectral_bf16" "$HOME/VirSO/best_model_spectral.pth" "bf16" "strict" 10 48 64 1 0 "$r"
done

for r in 1 2 3; do
  run_one "spectral_fp16_autocast" "$HOME/VirSO/best_model_spectral.pth" "fp16" "strict" 10 48 64 1 0 "$r"
done

for r in 1 2 3; do
  run_one "layer2_fp32" "$HOME/VirSO/best_model_2_layer.pth" "off" "strict" 2 48 40 1 1 "$r"
done

for r in 1 2 3; do
  run_one "layer2_tf32" "$HOME/VirSO/best_model_2_layer.pth" "off" "tf32" 2 48 40 1 1 "$r"
done

for r in 1 2 3; do
  run_one "layer2_bf16" "$HOME/VirSO/best_model_2_layer.pth" "bf16" "strict" 2 48 40 1 1 "$r"
done

for r in 1 2 3; do
  run_one "layer2_fp16_autocast" "$HOME/VirSO/best_model_2_layer.pth" "fp16" "strict" 2 48 40 1 1 "$r"
done

echo
echo "DONE. Official root:"
echo "$OFFICIAL_ROOT"
echo
echo "Registry:"
cat "$REG"