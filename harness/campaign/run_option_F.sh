#!/usr/bin/env bash
# Option F — bring every remaining table onto the unified protocol.
#
# Stages are ordered so that partial completion is still useful:
#   A  Heat Exchanger, all precisions          -> Table 9 (3 rows), Table 37, Table 38 heat rows
#   B  FNO sustained, all admitted precisions  -> Table 22
#   C  DeepONet sustained, all precisions      -> Table 29
#   D  WNO sustained, all precisions           -> Table 34
#   E  Sp2GNO sustained, all precisions        -> Table 38
#   F  short-run eager + ablations             -> Tables 17, 18, 26
#
# Every stage skips work already on disk, so the script is safe to re-run.
set -u
export PY_FD="${PY_FD:-/home/jetson/miniforge3/envs/vs_wno/bin/python}"
PY_EB=/home/jetson/miniforge3/envs/extra_bench/bin/python
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1

REPS="${REPS:-3}"; WARM="${WARM:-20}"; DUR="${DUR:-120}"; TEGRA_MS="${TEGRA_MS:-100}"
LOG=/home/jetson/jjyoo3/option_F.log; : > "$LOG"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "Option F start | reps=$REPS warmup=${WARM}s window=${DUR}s tegrastats=${TEGRA_MS}ms"

# =================================================== A. Heat Exchanger
say "===== STAGE A: Heat Exchanger (unified, all precisions) ====="
cd /home/jetson/VirSO/For_Jetson/For_Jetson || exit 1
sed 's/^# for r in 1 2 3; do run_one/for r in 1 2 3; do run_one/' \
    run_virso_unified_sustained.sh > run_virso_unified_all.sh
chmod +x run_virso_unified_all.sh
bash run_virso_unified_all.sh >> "$LOG" 2>&1 && say "STAGE A done" || say "STAGE A FAILED"

# =================================================== B. FNO sustained
say "===== STAGE B: FNO sustained (FP32 + TF32) ====="
cd /home/jetson/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
OUT=results/jetson_fno_unified_sustained; mkdir -p $OUT
A=artifacts
fno () {  # tag ckpt ts bank precision
  for rep in $(seq 1 "$REPS"); do
    local t="$1_$5_rep${rep}"
    [ -f "$OUT/${t}.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    tegrastats --interval "$TEGRA_MS" --logfile "$OUT/${t}_tegrastats.log" & local tp=$!
    sleep 0.3
    "$PY_FD" -m src.eval.benchmark_energy_inference --mode torchscript \
      --checkpoint "$2" --torchscript "$3" --input-bank "$4" --sample-index 0 \
      --batch-size 1 --precision-mode "$5" --warmup-seconds "$WARM" \
      --measure-seconds "$DUR" --results-dir "$OUT" --result-tag "$t" >> "$LOG" 2>&1
    local rc=$?; kill "$tp" 2>/dev/null; wait "$tp" 2>/dev/null
    [ $rc -eq 0 ] && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 2
  done
}
while IFS=, read -r name seed ckpt ts bank; do
  [ -z "${name:-}" ] && continue
  for prec in fp32_strict tf32; do
    fno "$name" "$ckpt" "$ts" "$bank" "$prec"
  done
done < <("$PY_FD" - <<'PY'
import csv
W={"burgers_fno_base":"burgers_base_r2048","burgers_fno_base_r8192":"burgers_base_r8192",
   "darcy_fno_base_r85":"darcy_base_r85","darcy_fno_base_r141":"darcy_base_r141",
   "darcy_fno_base_r211":"darcy_base_r211","darcy_fno_base_r281":"darcy_base_r281",
   "darcy_fno_base_r421":"darcy_base_r421","darcy_fno_large":"darcy_large_r141"}
for r in csv.DictReader(open("manifests/fno_jetson_manifest.csv")):
    if r["experiment_name"] in W:
        print(f'{W[r["experiment_name"]]},{r["selected_seed"]},{r["checkpoint_path"]},{r["torchscript_path"]},{r["input_bank_path"]}')
PY
)
say "STAGE B done"

# =============================================== C. DeepONet sustained
say "===== STAGE C: DeepONet sustained (all 5 precisions) ====="
cd /home/jetson/jjyoo3/EDCNO_DeepONet || exit 1
export PYTHONPATH=$PWD
OUTD=results/jetson_deeponet_unified_sustained; mkdir -p $OUTD
don () {  # tag ckpt ts bank precision
  for rep in $(seq 1 "$REPS"); do
    local t="$1_$5_rep${rep}"
    [ -f "$OUTD/${t}.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    "$PY_FD" -m src.eval.benchmark_sustained_inference --mode torchscript \
      --checkpoint "$2" --torchscript "$3" --input-bank "$4" --precision "$5" \
      --batch-size 1 --unified-protocol 1 --warmup-sec "$WARM" --duration-sec "$DUR" \
      --tegrastats-interval-ms "$TEGRA_MS" --results-dir "$OUTD" --result-tag "$t" \
      >> "$LOG" 2>&1 && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 2
  done
}
while IFS=, read -r name ckpt ts bank; do
  [ -z "${name:-}" ] && continue
  for prec in fp32_strict tf32 bf16_autocast fp16_autocast fp16_native; do
    don "$name" "$ckpt" "$ts" "$bank" "$prec"
  done
done < <("$PY_FD" - <<'PY'
import csv
W={"burgers_deeponet_base":"burgers_base","burgers_deeponet_base_r4096":"burgers_r4096",
   "burgers_deeponet_base_r8192":"burgers_r8192","darcy_deeponet_base":"darcy_r141",
   "darcy_deeponet_base_r281":"darcy_r281","darcy_deeponet_base_r421":"darcy_r421",
   "darcy_deeponet_large":"darcy_large_r141"}
for r in csv.DictReader(open("manifests/deeponet_jetson_manifest.csv")):
    if r["experiment_name"] in W:
        print(f'{W[r["experiment_name"]]},{r["checkpoint_path"]},{r["torchscript_path"]},{r["input_bank_path"]}')
PY
)
say "STAGE C done"

# ==================================================== D. WNO sustained
say "===== STAGE D: WNO sustained (FP32/TF32/BF16/FP16auto) ====="
cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks || exit 1
BKW=/home/jetson/data/wno_inference_banks_exact
wno () {  # case ds ckpt bank precision
  for rep in $(seq 1 "$REPS"); do
    local t="$1_$5_rep${rep}"
    [ -f "results/jetson_wno_exact/wno_unified_sustained/${t}/$5/result.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    PYTHONNOUSERSITE=1 "$PY_EB" bench_wno_jetson_exact.py --case-id "$t" --dataset "$2" \
      --checkpoint "$3" --bank "$4" --precision-mode "$5" --sample-index 0 --batch-size 1 \
      --timing-class sustained --warmup-seconds "$WARM" --measure-seconds "$DUR" \
      --tegrastats-interval-ms "$TEGRA_MS" --device cuda \
      --results-root results/jetson_wno_exact --run-tag wno_unified_sustained \
      --compute-full-eval 0 --compute-perturbation 0 >> "$LOG" 2>&1 \
      && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 2
  done
}
for row in \
 "wno_burgers_base_r512|burgers|checkpoints/wno_burgers_base_r512.pth|$BKW/burgers_r512_bank.pt" \
 "wno_burgers_base_r1024|burgers|checkpoints/wno_burgers_base_r1024.pth|$BKW/burgers_r1024_bank.pt" \
 "wno_burgers_base_r2048|burgers|checkpoints/wno_burgers_base_r2048.pth|$BKW/burgers_r2048_bank.pt" \
 "wno_burgers_base_r4096|burgers|checkpoints/wno_burgers_base_r4096.pth|$BKW/burgers_r4096_bank.pt" \
 "wno_burgers_base_r8192|burgers|checkpoints/wno_burgers_base_r8192.pth|$BKW/burgers_r8192_bank.pt" \
 "wno_darcy_base_r85|darcy|checkpoints/wno_darcy_base_r85.pth|$BKW/darcy_r85_bank.pt" \
 "wno_darcy_base_r141|darcy|checkpoints/wno_darcy_base_r141.pth|$BKW/darcy_r141_bank.pt" \
 "wno_darcy_base_r211|darcy|checkpoints/wno_darcy_base_r211.pth|$BKW/darcy_r211_bank.pt" \
 "wno_darcy_base_r281|darcy|checkpoints/wno_darcy_base_r281.pth|$BKW/darcy_r281_bank.pt" \
 "wno_darcy_base_r421|darcy|checkpoints/wno_darcy_base_r421.pth|$BKW/darcy_r421_bank.pt" \
 "wno_darcy_large_r141|darcy|checkpoints/wno_darcy_large_r141.pth|$BKW/darcy_r141_bank.pt" ; do
  IFS='|' read -r c d k b <<< "$row"
  for prec in fp32_strict tf32 bf16_autocast fp16_autocast; do wno "$c" "$d" "$k" "$b" "$prec"; done
done
say "STAGE D done"

# ================================================= E. Sp2GNO sustained
say "===== STAGE E: Sp2GNO sustained (all 5 precisions) ====="
cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026 || exit 1
NEW=/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints
SUITE=inference_runs/sp2gno_unified_sustained
sp () {  # case dataset ckpt width sub res k precision
  for rep in $(seq 1 "$REPS"); do
    local rn="$1_$8_rep${rep}"
    [ -f "$SUITE/${rn}/reports/sp2gno_edge_summary_${rn}.csv" ] && { say "[SKIP] $rn"; continue; }
    local a=(--case_id "$1" --run_name "$rn" --suite_root "$SUITE" --dataset "$2"
             --data_dir /home/jetson/data --cache_dir cache --ckpt "$NEW/$3"
             --width "$4" --n_layers 6 --num_freq 64 --k "$7" --precision "$8"
             --unified_protocol 1 --warmup_seconds "$WARM" --min_duration_s "$DUR"
             --tegrastats_interval_ms "$TEGRA_MS" --validity_samples 8
             --sample_index 0 --rep "$rep")
    if [ "$2" = burgers ]; then a+=(--sub "$5" --burgers_split Jetson_data/burgers_split.json)
    else a+=(--res "$6" --ntrain 900 --nval 100 --ntest 200); fi
    say "[RUN] $rn"
    PYTHONNOUSERSITE=1 "$PY_EB" bench_sp2gno_jetson_exact.py "${a[@]}" >> "$LOG" 2>&1 \
      && say "[ OK ] $rn" || say "[FAIL] $rn"
    sleep 2
  done
}
for row in \
 "sp2gno_burgers_small_s2048|burgers|sp2gno_burgers_small_s2048.pth|13|4|0|8" \
 "sp2gno_burgers_base_s2048|burgers|sp2gno_burgers_base_s2048.pth|24|4|0|8" \
 "sp2gno_burgers_large_s2048|burgers|sp2gno_burgers_large_s2048.pth|45|4|0|8" \
 "sp2gno_burgers_base_r512|burgers|sp2gno_burgers_base_r512.pth|24|16|0|8" \
 "sp2gno_burgers_base_r1024|burgers|sp2gno_burgers_base_r1024.pth|24|8|0|8" \
 "sp2gno_burgers_base_s4096|burgers|sp2gno_burgers_base_s4096.pth|24|2|0|8" \
 "sp2gno_burgers_base_r8192|burgers|sp2gno_burgers_base_r8192.pth|24|1|0|8" \
 "sp2gno_darcy_small_r141|darcy|sp2gno_darcy_small_r141.pth|13|0|141|20" \
 "sp2gno_darcy_base_r141|darcy|sp2gno_darcy_base_r141.pth|24|0|141|20" \
 "sp2gno_darcy_large_r141|darcy|sp2gno_darcy_large_r141.pth|45|0|141|20" \
 "sp2gno_darcy_base_r85|darcy|sp2gno_darcy_base_r85.pth|24|0|85|20" \
 "sp2gno_darcy_base_r211|darcy|sp2gno_darcy_base_r211.pth|24|0|211|20" \
 "sp2gno_darcy_base_r281|darcy|sp2gno_darcy_base_r281.pth|24|0|281|20" \
 "sp2gno_darcy_base_r421|darcy|sp2gno_darcy_base_r421.pth|24|0|421|20" ; do
  IFS='|' read -r c d k w sub res kn <<< "$row"
  for prec in fp32_strict tf32 bf16_autocast fp16_autocast fp16_native; do
    sp "$c" "$d" "$k" "$w" "$sub" "$res" "$kn" "$prec"
  done
done
say "STAGE E done"

say "===== OPTION F FINISHED ====="
