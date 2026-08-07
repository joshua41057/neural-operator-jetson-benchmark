#!/usr/bin/env bash
set -u
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1
PY_EB=/home/jetson/miniforge3/envs/extra_bench/bin/python
LOG=/home/jetson/jjyoo3/retry_wno421.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks
for rep in 1 2; do
  t="wno_darcy_base_r421_fp32_strict_rep${rep}"
  for attempt in 1 2 3; do
    [ -f "results/jetson_wno_exact/wno_unified_sustained/${t}/fp32_strict/result.json" ] && \
      grep -q '"status": "success"' "results/jetson_wno_exact/wno_unified_sustained/${t}/fp32_strict/result.json" && break
    say "[RUN] $t (attempt $attempt)"
    rm -rf "results/jetson_wno_exact/wno_unified_sustained/${t}"
    sync; sleep 20   # let the allocator and page cache settle before a 3.7 GB workload
    PYTHONNOUSERSITE=1 "$PY_EB" bench_wno_jetson_exact.py --case-id "$t" --dataset darcy \
      --checkpoint checkpoints/wno_darcy_base_r421.pth \
      --bank /home/jetson/data/wno_inference_banks_exact/darcy_r421_bank.pt \
      --precision-mode fp32_strict --sample-index 0 --batch-size 1 --eval-batch-size 10 \
      --timing-class sustained --warmup-seconds 20 --measure-seconds 120 \
      --tegrastats-interval-ms 100 --device cuda \
      --results-root results/jetson_wno_exact --run-tag wno_unified_sustained \
      --compute-full-eval 0 --compute-perturbation 0 >> "$LOG" 2>&1
    if grep -q '"status": "success"' "results/jetson_wno_exact/wno_unified_sustained/${t}/fp32_strict/result.json" 2>/dev/null; then
      say "[ OK ] $t"; break
    else
      say "[FAIL] $t (attempt $attempt)"
    fi
  done
done
say "WNO421 RETRY DONE"
