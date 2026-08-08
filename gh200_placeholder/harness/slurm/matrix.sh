#!/bin/bash
# GH200 FP32 inference benchmark matrix (27 cases).
#
# Families: fno | deeponet | wno | sp2gno | hx | all
# Usage:    bash matrix.sh <family>
#
# Env knobs (defaults chosen to mirror the Jetson protocol):
#   RUN_TAG        run label / output subdir        (default gh200_fp32_<date via caller>)
#   RESULTS_ROOT   results root                     (<REPO>/results)
#   REPS           repetitions per case             (3)
#   DURATION       sustained measure window (s)     (120)
#   WARMUP         warmup window (s)                (20)
#   INTERVAL_MS    nvidia-smi sample interval (ms)  (200)
set -u

RESULTS_ROOT="${RESULTS_ROOT:-<REPO>/results}"
RUN_TAG="${RUN_TAG:-gh200_fp32}"
REPS="${REPS:-3}"
DURATION="${DURATION:-120}"
WARMUP="${WARMUP:-20}"
INTERVAL_MS="${INTERVAL_MS:-200}"

JHPC=<JETSON_HPC>
EDGE=<WORK>/edge
GHB=<REPO>
OUT="$RESULTS_ROOT/$RUN_TAG"
mkdir -p "$OUT"

FNO_DIR="$JHPC/EDCNO"
DON_DIR="$JHPC/EDCNO_DeepONet"
WNO_DIR="$JHPC/WNO_Sp2GNO_Benchmarks"
SP2_DIR="$JHPC/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026"
HX_DIR="$JHPC/VirSO/For_Jetson/For_Jetson"
SP2_SPLIT="$SP2_DIR/Jetson_data/burgers_split.json"

log() { echo "[$(date '+%T')] $*"; }

# ---------------------------------------------------------------- FNO
run_fno() {
  cd "$FNO_DIR" || exit 2
  export PYTHONPATH="$FNO_DIR"
  local RES="$OUT/fno"
  # case  ckpt  bank  dataset
  local cases=(
    "fno_burgers_small_r2048 burgers_fno_small_seed2_best burgers_r2048_bank burgers"
    "fno_burgers_base_r2048  burgers_fno_base_seed3_best  burgers_r2048_bank burgers"
    "fno_burgers_large_r2048 burgers_fno_large_seed0_best burgers_r2048_bank burgers"
    "fno_darcy_small_r141    darcy_fno_small_seed4_best   darcy_r141_bank    darcy"
    "fno_darcy_base_r141     darcy_fno_base_seed0_best    darcy_r141_bank    darcy"
    "fno_darcy_large_r141    darcy_fno_large_seed0_best   darcy_r141_bank    darcy"
  )
  for row in "${cases[@]}"; do
    set -- $row; local case=$1 ck=$2 bank=$3
    for r in $(seq 1 "$REPS"); do
      log "FNO $case rep$r"
      python -m src.eval.benchmark_energy_inference_gh200 \
        --mode eager \
        --checkpoint "artifacts/checkpoints/$ck.pt" \
        --input-bank "artifacts/benchmark_inputs/$bank.pt" \
        --precision-mode fp32_strict \
        --warmup-seconds "$WARMUP" --measure-seconds "$DURATION" \
        --power-interval-ms "$INTERVAL_MS" \
        --results-dir "$RES" --result-tag "${case}_rep${r}" || log "FNO $case rep$r FAILED"
    done
  done
}

# ---------------------------------------------------------------- DeepONet
run_deeponet() {
  cd "$DON_DIR" || exit 2
  export PYTHONPATH="$DON_DIR"
  local RES="$OUT/deeponet"
  local cases=(
    "deeponet_burgers_small_r2048 burgers_deeponet_small_seed1_best burgers_r2048_bank"
    "deeponet_burgers_base_r2048  burgers_deeponet_base_seed2_best  burgers_r2048_bank"
    "deeponet_burgers_large_r2048 burgers_deeponet_large_seed3_best burgers_r2048_bank"
    "deeponet_darcy_small_r141    darcy_deeponet_small_seed2_best   darcy_r141_bank"
    "deeponet_darcy_base_r141     darcy_deeponet_base_seed2_best    darcy_r141_bank"
    "deeponet_darcy_large_r141    darcy_deeponet_large_seed1_best   darcy_r141_bank"
  )
  for row in "${cases[@]}"; do
    set -- $row; local case=$1 ck=$2 bank=$3
    for r in $(seq 1 "$REPS"); do
      log "DeepONet $case rep$r"
      python -m src.eval.benchmark_sustained_inference_gh200 \
        --mode eager \
        --checkpoint "artifacts/checkpoints/$ck.pt" \
        --input-bank "artifacts/benchmark_inputs/$bank.pt" \
        --precision fp32_strict --batch-size 1 \
        --warmup-sec 10 --duration-sec "$DURATION" \
        --tegrastats-interval-ms "$INTERVAL_MS" \
        --results-dir "$RES" --result-tag "${case}_rep${r}" || log "DeepONet $case rep$r FAILED"
    done
  done
}

# ---------------------------------------------------------------- WNO
run_wno() {
  cd "$WNO_DIR" || exit 2
  export PYTHONPATH="$WNO_DIR"
  local RES="$OUT/wno"
  local cases=(
    "wno_burgers_small_r2048 burgers wno_burgers_small_r2048 $GHB/banks/burgers_r2048_bank.pt"
    "wno_burgers_base_r2048  burgers wno_burgers_base_r2048  $GHB/banks/burgers_r2048_bank.pt"
    "wno_burgers_large_r2048 burgers wno_burgers_large_r2048 $GHB/banks/burgers_r2048_bank.pt"
    "wno_darcy_small_r141    darcy   wno_darcy_small_r141    $GHB/banks/darcy_r141_bank.pt"
    "wno_darcy_base_r141     darcy   wno_darcy_base_r141     $GHB/banks/darcy_r141_bank.pt"
    "wno_darcy_large_r141    darcy   wno_darcy_large_r141    $GHB/banks/darcy_r141_bank.pt"
  )
  for row in "${cases[@]}"; do
    set -- $row; local case=$1 ds=$2 ck=$3 bank=$4
    for r in $(seq 1 "$REPS"); do
      log "WNO $case rep$r"
      python bench_wno_gh200.py \
        --case-id "${case}_rep${r}" --dataset "$ds" \
        --checkpoint "checkpoints/$ck.pth" --bank "$bank" \
        --precision-mode fp32_strict --batch-size 1 --eval-batch-size 10 \
        --warmup-seconds "$WARMUP" --measure-seconds "$DURATION" \
        --tegrastats-interval-ms "$INTERVAL_MS" \
        --device cuda --results-root "$RES" --run-tag "$RUN_TAG" \
        --compute-full-eval 1 --compute-perturbation 0 || log "WNO $case rep$r FAILED"
    done
  done
}

# ---------------------------------------------------------------- Sp2GNO burgers/darcy
run_sp2gno() {
  cd "$SP2_DIR" || exit 2
  export PYTHONPATH="$SP2_DIR"
  local RES="$OUT/sp2gno"
  # case  dataset  width  ckpt(edge)  extra(--sub/--res + --k)
  local cases=(
    "sp2gno_burgers_small_r2048 burgers 13 sp2gno_burgers_small_s2048 --sub|4|--k|8"
    "sp2gno_burgers_base_r2048  burgers 24 sp2gno_burgers_base_s2048  --sub|4|--k|8"
    "sp2gno_burgers_large_r2048 burgers 45 sp2gno_burgers_large_s2048 --sub|4|--k|8"
    "sp2gno_darcy_small_r141    darcy   13 sp2gno_darcy_small_r141    --res|141|--k|20"
    "sp2gno_darcy_base_r141     darcy   24 sp2gno_darcy_base_r141     --res|141|--k|20"
    "sp2gno_darcy_large_r141    darcy   45 sp2gno_darcy_large_r141    --res|141|--k|20"
  )
  for row in "${cases[@]}"; do
    set -- $row; local case=$1 ds=$2 w=$3 ck=$4 extra=$5
    IFS='|' read -ra EX <<< "$extra"
    for r in $(seq 1 "$REPS"); do
      log "Sp2GNO $case rep$r"
      python bench_sp2gno_gh200.py \
        --dataset "$ds" --width "$w" --n_layers 6 --num_freq 64 "${EX[@]}" \
        --ckpt "$EDGE/checkpoints/$ck.pth" \
        --data_dir "$EDGE/Jetson_data" --burgers_split "$SP2_SPLIT" \
        --cache_dir "$SP2_DIR/cache" \
        --precision fp32_strict --warmup 20 --min_duration_s "$DURATION" \
        --tegrastats_interval_ms "$INTERVAL_MS" \
        --suite_root "$RES" --case_id "$case" \
        --run_name "${case}_fp32_strict_rep${r}" --rep "$r" || log "Sp2GNO $case rep$r FAILED"
    done
  done
}

# ---------------------------------------------------------------- Heat Exchanger
run_hx() {
  cd "$HX_DIR" || exit 2
  export PYTHONPATH="$HX_DIR"
  local RES="$OUT/hx"
  for v in full spectral layer2; do
    for r in $(seq 1 "$REPS"); do
      log "HX $v rep$r"
      python run_virso_inference_gh200.py \
        --variant "$v" --run-dir "$RES/${v}_rep${r}" \
        --duration "$DURATION" --interval-ms "$INTERVAL_MS" || log "HX $v rep$r FAILED"
    done
  done
}

FAMILY="${1:-all}"
log "matrix start family=$FAMILY RUN_TAG=$RUN_TAG REPS=$REPS DURATION=$DURATION -> $OUT"
case "$FAMILY" in
  fno) run_fno;;
  deeponet) run_deeponet;;
  wno) run_wno;;
  sp2gno) run_sp2gno;;
  hx) run_hx;;
  all) run_fno; run_deeponet; run_wno; run_sp2gno; run_hx;;
  *) echo "unknown family $FAMILY"; exit 2;;
esac
log "matrix done family=$FAMILY"
