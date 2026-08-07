#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

OUTDIR="results/jetson_fno_profile_nsys"
mkdir -p "${OUTDIR}"

run_profile () {
  local tag="$1"
  shift

  local tegra_log="${OUTDIR}/${tag}_tegrastats.log"
  local rep_prefix="${OUTDIR}/${tag}"
  local rep_file="${OUTDIR}/${tag}.nsys-rep"
  local sqlite_file="${OUTDIR}/${tag}.sqlite"

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  rm -f "${rep_file}" "${sqlite_file}" "${OUTDIR}/${tag}_nsys_stats.txt"

  tegrastats --interval 100 --logfile "${tegra_log}" &
  TPID=$!

  set +e
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --cuda-memory-usage=false \
    --stats=true \
    --force-overwrite=true \
    -o "${rep_prefix}" \
    "$@"
  STATUS=$?
  set -e

  kill ${TPID} 2>/dev/null || true
  wait ${TPID} 2>/dev/null || true

  if [[ ${STATUS} -eq 0 ]]; then
    echo "[ OK ] ${tag}"

    # optional: regenerate compact text stats from rep
    nsys stats \
      --force-export=true \
      --report nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
      "${rep_file}" \
      > "${OUTDIR}/${tag}_nsys_stats.txt" 2>&1 || true
  else
    echo "[FAIL] ${tag} exit=${STATUS}"
  fi
}

# 1) very safe baseline
run_profile "burgers_base_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
    --torchscript artifacts/torchscript/burgers_fno_base_seed3.ts \
    --input-bank artifacts/benchmark_inputs/burgers_r2048_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 10 \
    --num-iters 30 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag burgers_base_ts_fp32 \
    --notes "nsys representative profile"

# 2) low-res Darcy
run_profile "darcy_base_r85_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_base_r85_seed2_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_base_r85_seed2.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r85_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 10 \
    --num-iters 30 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag darcy_base_r85_ts_fp32 \
    --notes "nsys representative profile"

# 3) high-res Darcy: much smaller count because profiler overhead is large
run_profile "darcy_base_r281_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r281_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 3 \
    --num-iters 8 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag darcy_base_r281_ts_fp32 \
    --notes "nsys representative profile reduced-overhead"

# 4) frontier large model: keep short
run_profile "darcy_large_on421_ts_fp32" \
  python -m src.eval.profile_inference \
    --mode torchscript \
    --checkpoint artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
    --torchscript artifacts/torchscript/darcy_fno_large_seed0.ts \
    --input-bank artifacts/benchmark_inputs/darcy_r421_bank.pt \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 2 \
    --num-iters 5 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag darcy_large_on421_ts_fp32 \
    --notes "nsys frontier profile reduced-overhead"