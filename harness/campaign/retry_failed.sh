#!/usr/bin/env bash
set -u
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1
PY_EB=/home/jetson/miniforge3/envs/extra_bench/bin/python
LOG=/home/jetson/jjyoo3/retry_failed.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks
for rep in 1 2; do
  t="wno_darcy_base_r421_fp32_strict_rep${rep}"; say "[RUN] $t"
  PYTHONNOUSERSITE=1 "$PY_EB" bench_wno_jetson_exact.py --case-id "$t" --dataset darcy \
    --checkpoint checkpoints/wno_darcy_base_r421.pth \
    --bank /home/jetson/data/wno_inference_banks_exact/darcy_r421_bank.pt \
    --precision-mode fp32_strict --sample-index 0 --batch-size 1 --eval-batch-size 10 \
    --timing-class sustained --warmup-seconds 20 --measure-seconds 120 \
    --tegrastats-interval-ms 100 --device cuda \
    --results-root results/jetson_wno_exact --run-tag wno_unified_sustained \
    --compute-full-eval 0 --compute-perturbation 0 >> "$LOG" 2>&1 \
    && say "[ OK ] $t" || say "[FAIL] $t"
  sleep 5
done

cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026
rn="sp2gno_burgers_small_s2048_bf16_autocast_rep3"; say "[RUN] $rn"
PYTHONNOUSERSITE=1 "$PY_EB" bench_sp2gno_jetson_exact.py \
  --case_id sp2gno_burgers_small_s2048 --run_name "$rn" \
  --suite_root inference_runs/sp2gno_unified_sustained --dataset burgers \
  --data_dir /home/jetson/data --cache_dir cache \
  --ckpt /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints/sp2gno_burgers_small_s2048.pth \
  --width 13 --n_layers 6 --num_freq 64 --k 8 --precision bf16_autocast \
  --unified_protocol 1 --warmup_seconds 20 --min_duration_s 120 \
  --tegrastats_interval_ms 100 --validity_samples 8 --sample_index 0 --rep 3 \
  --sub 4 --burgers_split Jetson_data/burgers_split.json >> "$LOG" 2>&1 \
  && say "[ OK ] $rn" || say "[FAIL] $rn"
say "RETRY DONE"
