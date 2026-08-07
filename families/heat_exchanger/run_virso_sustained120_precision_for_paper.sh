#!/usr/bin/env bash
set -euo pipefail

# Same 12-case x 3-rep matrix as run_virso_official_cache_enabled_for_paper.sh,
# with SUSTAIN_MIN_DURATION_S=120 added so each run cycles the full 310-sample
# test set repeatedly until >=120s elapsed, matching the sustained-window
# protocol used for the Burgers/Darcy Sp2GNO precision table (bench_sp2gno_
# jetson_exact.py --min_duration_s 120). Everything else (VIRSO_ALLOCATOR_
# AUDIT=1, tegrastats monitor wrapping the whole loop, cache-enabled
# allocator) is unchanged from the already-verified official script.

ROOT="$HOME/VirSO/For_Jetson/For_Jetson"
PY="$HOME/miniforge3/envs/wave_gpu_310_test/bin/python"
SCRIPT="$ROOT/run_virso_inference_jetson.py"
PARSE="$ROOT/scripts/parse_tegrastats.py"
MERGE="$ROOT/scripts/merge_reports.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
OFFICIAL_ROOT="$ROOT/inference_runs/sustained120_virso_${STAMP}"
REG="$OFFICIAL_ROOT/sustained120_virso_registry.csv"

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

  local RUN_TS="sustained120_${CASE}_r${REP}"
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
  echo "RUN_TS=$RUN_TS  CASE=$CASE  AMP_MODE=$AMP_MODE TF32_MODE=$TF32_MODE"
  echo "================================================================================"

  {
    echo "RUN_TS=$RUN_TS"; echo "CASE=$CASE"; echo "MODEL_PATH=$MODEL_PATH"
    echo "AMP_MODE=$AMP_MODE"; echo "TF32_MODE=$TF32_MODE"
    echo "NUM_LAYERS=$NUM_LAYERS"; echo "WIDTH=$WIDTH"; echo "MAX_MODE=$MAX_MODE"
    echo "SPECTRAL=$SPECTRAL"; echo "SPATIAL=$SPATIAL"
    echo "SUSTAIN_MIN_DURATION_S=120"; echo "SUSTAIN_MIN_CYCLES=1"
    echo "VIRSO_ALLOCATOR_AUDIT=1"
    date; uname -a
  } > "$RUN_DIR/provenance.txt"

  unset PYTORCH_NO_CUDA_MEMORY_CACHING
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

  export RUN_TS="$RUN_TS" RUN_DIR="$RUN_DIR" OUT_DIR="$OUT_DIR" LOG_DIR="$LOG_DIR" \
         REPORT_DIR="$REPORT_DIR" EDGE_CSV="$EDGE_CSV" MODEL_PATH="$MODEL_PATH"

  export MONITOR_CMD="tegrastats" MONITOR_INTERVAL_MS="200"
  export VIRSO_ALLOCATOR_AUDIT="1" MEM_AUDIT_ONLY="0" SAVE_OUTPUTS="1"
  export PROFILE_FLOPS="0" PROFILE_WARMUP="2" PROFILE_ITERS="1"
  export SUSTAIN_MIN_DURATION_S="120" SUSTAIN_MIN_CYCLES="1"

  export TF32_MODE="$TF32_MODE" AMP_MODE="$AMP_MODE" AMP_FALLBACK="off"
  export NUM_LAYERS="$NUM_LAYERS" WIDTH="$WIDTH" MAX_MODE="$MAX_MODE"
  export K_NEIGHBORS="30" EMBED="1" SPECTRAL="$SPECTRAL" SPATIAL="$SPATIAL"
  export COLLAB_SKIP="1" SPECTRAL_SKIP="1"

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
  fi

  echo "RUN_DIR=$RUN_DIR"
  sync
  sleep 20
}

for r in 1 2 3; do run_one "full_fp32"            "$ROOT/sp2gno_final.pth"              "off" "strict" 10 48 64 1 1 "$r"; done
for r in 1 2 3; do run_one "full_tf32"            "$ROOT/sp2gno_final.pth"              "off" "tf32"   10 48 64 1 1 "$r"; done
for r in 1 2 3; do run_one "full_bf16"            "$ROOT/sp2gno_final.pth"              "bf16" "strict" 10 48 64 1 1 "$r"; done
for r in 1 2 3; do run_one "full_fp16_autocast"   "$ROOT/sp2gno_final.pth"              "fp16" "strict" 10 48 64 1 1 "$r"; done

for r in 1 2 3; do run_one "spectral_fp32"          "$HOME/VirSO/best_model_spectral.pth" "off" "strict" 10 48 64 1 0 "$r"; done
for r in 1 2 3; do run_one "spectral_tf32"          "$HOME/VirSO/best_model_spectral.pth" "off" "tf32"   10 48 64 1 0 "$r"; done
for r in 1 2 3; do run_one "spectral_bf16"          "$HOME/VirSO/best_model_spectral.pth" "bf16" "strict" 10 48 64 1 0 "$r"; done
for r in 1 2 3; do run_one "spectral_fp16_autocast" "$HOME/VirSO/best_model_spectral.pth" "fp16" "strict" 10 48 64 1 0 "$r"; done

for r in 1 2 3; do run_one "layer2_fp32"          "$HOME/VirSO/best_model_2_layer.pth"  "off" "strict" 2 48 40 1 1 "$r"; done
for r in 1 2 3; do run_one "layer2_tf32"          "$HOME/VirSO/best_model_2_layer.pth"  "off" "tf32"   2 48 40 1 1 "$r"; done
for r in 1 2 3; do run_one "layer2_bf16"          "$HOME/VirSO/best_model_2_layer.pth"  "bf16" "strict" 2 48 40 1 1 "$r"; done
for r in 1 2 3; do run_one "layer2_fp16_autocast" "$HOME/VirSO/best_model_2_layer.pth"  "fp16" "strict" 2 48 40 1 1 "$r"; done

echo
echo "DONE. Official root: $OFFICIAL_ROOT"
echo "Registry:"
cat "$REG"
