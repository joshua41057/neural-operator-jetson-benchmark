#!/usr/bin/env bash
# Unified cross-family sustained sweep — the rows that appear in Table 9 and the abstract.
#
# One protocol for all four families:
#   preloaded device-resident batch-size-one request  (staging outside the timed window)
#   20 s time-based warmup   |   exactly 120 s window   |   R = 3   |   tegrastats 100 ms
#
# The within-family precision sweeps (Tables 30/34/38, Fig. 3) and the frontier sweeps
# keep their family-native configuration: they are FP32-relative ratios inside one family,
# so protocol differences cancel there.
set -u
PY_FD="${PY_FD:-/home/jetson/miniforge3/envs/vs_wno/bin/python}"   # FNO / DeepONet env (torch+CUDA)
export PY_FD
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1

REPS="${REPS:-3}"
WARM="${WARM:-20}"
DUR="${DUR:-120}"
TEGRA_MS="${TEGRA_MS:-100}"
LOG="/home/jetson/jjyoo3/unified_sustained.log"
: > "${LOG}"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

say "protocol: warmup=${WARM}s window=${DUR}s reps=${REPS} tegrastats=${TEGRA_MS}ms"

# --------------------------------------------------------------------- FNO
say "=== FNO sustained ==="
cd /home/jetson/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
mkdir -p results/jetson_fno_unified_sustained
fno () {  # tag ckpt ts bank
  for rep in $(seq 1 "${REPS}"); do
    local t="$1_rep${rep}"
    [ -f "results/jetson_fno_unified_sustained/${t}_energy.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    tegrastats --interval "${TEGRA_MS}" \
      --logfile "results/jetson_fno_unified_sustained/${t}_tegrastats.log" & local tp=$!
    sleep 0.3
    "${PY_FD}" -m src.eval.benchmark_energy_inference \
      --mode torchscript --checkpoint "$2" --torchscript "$3" --input-bank "$4" \
      --sample-index 0 --batch-size 1 --precision-mode fp32_strict \
      --warmup-seconds "${WARM}" --measure-seconds "${DUR}" \
      --results-dir results/jetson_fno_unified_sustained --result-tag "${t}" >> "${LOG}" 2>&1
    local rc=$?; kill "${tp}" 2>/dev/null; wait "${tp}" 2>/dev/null
    [ $rc -eq 0 ] && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 2
  done
}
A=artifacts
fno burgers_base_r2048 $A/checkpoints/burgers_fno_base_seed3_best.pt $A/torchscript/burgers_fno_base_seed3.ts $A/benchmark_inputs/burgers_r2048_bank.pt
fno darcy_base_r281 $A/checkpoints/darcy_fno_base_r281_seed1_best.pt $A/torchscript/darcy_fno_base_r281_seed1.ts $A/benchmark_inputs/darcy_r281_bank.pt

# ---------------------------------------------------------------- DeepONet
say "=== DeepONet sustained ==="
cd /home/jetson/jjyoo3/EDCNO_DeepONet || exit 1
export PYTHONPATH=$PWD
mkdir -p results/jetson_deeponet_unified_sustained
don () {  # tag ckpt ts bank precision
  for rep in $(seq 1 "${REPS}"); do
    local t="$1_rep${rep}"
    [ -f "results/jetson_deeponet_unified_sustained/${t}.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    "${PY_FD}" -m src.eval.benchmark_sustained_inference \
      --mode torchscript --checkpoint "$2" --torchscript "$3" --input-bank "$4" \
      --precision "$5" --batch-size 1 --unified-protocol 1 \
      --warmup-sec "${WARM}" --duration-sec "${DUR}" --tegrastats-interval-ms "${TEGRA_MS}" \
      --results-dir results/jetson_deeponet_unified_sustained --result-tag "${t}" >> "${LOG}" 2>&1 \
      && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 2
  done
}
M=manifests/deeponet_jetson_manifest.csv
eval "$("${PY_FD}" - "$M" <<'PY'
import csv,sys
w={"burgers_deeponet_base":"B","darcy_deeponet_base_r281":"D281"}
for r in csv.DictReader(open(sys.argv[1])):
    if r["experiment_name"] in w:
        k=w[r["experiment_name"]]
        print(f'{k}_CK="{r["checkpoint_path"]}"; {k}_TS="{r["torchscript_path"]}"; {k}_BK="{r["input_bank_path"]}"')
PY
)"
don burgers_base_fp32   "$B_CK"    "$B_TS"    "$B_BK"    fp32_strict
don darcy_r281_fp32     "$D281_CK" "$D281_TS" "$D281_BK" fp32_strict
don darcy_r281_fp16nat  "$D281_CK" "$D281_TS" "$D281_BK" fp16_native

# --------------------------------------------------------------------- WNO
say "=== WNO sustained ==="
cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks || exit 1
PY_EB=/home/jetson/miniforge3/envs/extra_bench/bin/python
BK=/home/jetson/data/wno_inference_banks_exact
wno () {  # case ds ckpt bank
  for rep in $(seq 1 "${REPS}"); do
    local t="$1_rep${rep}"
    [ -f "results/jetson_wno_exact/wno_unified_sustained/${t}/fp32_strict/result.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    PYTHONNOUSERSITE=1 "${PY_EB}" bench_wno_jetson_exact.py \
      --case-id "${t}" --dataset "$2" --checkpoint "$3" --bank "$4" \
      --precision-mode fp32_strict --sample-index 0 --batch-size 1 --eval-batch-size 10 \
      --timing-class sustained --warmup-seconds "${WARM}" --measure-seconds "${DUR}" \
      --tegrastats-interval-ms "${TEGRA_MS}" --device cuda \
      --results-root results/jetson_wno_exact --run-tag wno_unified_sustained \
      --compute-full-eval 0 --compute-perturbation 0 >> "${LOG}" 2>&1 \
      && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 2
  done
}
wno wno_burgers_base_r2048 burgers checkpoints/wno_burgers_base_r2048.pth $BK/burgers_r2048_bank.pt
wno wno_darcy_base_r281    darcy   checkpoints/wno_darcy_base_r281.pth    $BK/darcy_r281_bank.pt

# ------------------------------------------------------------------ Sp2GNO
say "=== Sp2GNO sustained (unified prestaged) ==="
cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026 || exit 1
NEW=/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints
SUITE=inference_runs/sp2gno_unified_sustained
sp () {  # case dataset ckpt width sub res k
  for rep in $(seq 1 "${REPS}"); do
    local rn="$1_fp32_strict_rep${rep}"
    [ -f "${SUITE}/${rn}/reports/sp2gno_edge_summary_${rn}.csv" ] && { say "[SKIP] ${rn}"; continue; }
    local a=(--case_id "$1" --run_name "${rn}" --suite_root "${SUITE}" --dataset "$2"
             --data_dir /home/jetson/data --cache_dir cache --ckpt "${NEW}/$3"
             --width "$4" --n_layers 6 --num_freq 64 --k "$7" --precision fp32_strict
             --unified_protocol 1 --warmup_seconds "${WARM}" --min_duration_s "${DUR}"
             --tegrastats_interval_ms "${TEGRA_MS}" --validity_samples 8
             --sample_index 0 --rep "${rep}")
    if [ "$2" = "burgers" ]; then a+=(--sub "$5" --burgers_split Jetson_data/burgers_split.json)
    else a+=(--res "$6" --ntrain 900 --nval 100 --ntest 200); fi
    say "[RUN] ${rn}"
    PYTHONNOUSERSITE=1 "${PY_EB}" bench_sp2gno_jetson_exact.py "${a[@]}" >> "${LOG}" 2>&1 \
      && say "[ OK ] ${rn}" || say "[FAIL] ${rn}"
    sleep 2
  done
}
sp sp2gno_burgers_base_s2048 burgers sp2gno_burgers_base_s2048.pth 24 4 0 8
sp sp2gno_darcy_base_r141    darcy   sp2gno_darcy_base_r141.pth    24 0 141 20

say "=== UNIFIED SUSTAINED DONE ==="
