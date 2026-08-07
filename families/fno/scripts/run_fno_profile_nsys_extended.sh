#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
OUTDIR="results/jetson_fno_profile_nsys_extended"
mkdir -p "${OUTDIR}"

run_profile () {
  local tag="$1"
  shift

  local rep_prefix="${OUTDIR}/${tag}"
  local rep_file="${OUTDIR}/${tag}.nsys-rep"

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  rm -f "${rep_file}" "${OUTDIR}/${tag}.sqlite" "${OUTDIR}/${tag}_nsys_stats.txt"

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

  if [[ ${STATUS} -eq 0 ]]; then
    echo "[ OK ] ${tag}"
    nsys stats \
      --force-export=true \
      --report nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
      "${rep_file}" \
      > "${OUTDIR}/${tag}_nsys_stats.txt" 2>&1 || true
  else
    echo "[FAIL] ${tag} exit=${STATUS}"
  fi
}

run_case () {
  local tag="$1"
  local ckpt="$2"
  local ts="$3"
  local bank="$4"

  run_profile "${tag}" \
    "${PYBIN}" -m src.eval.profile_inference \
      --mode torchscript \
      --checkpoint "${ckpt}" \
      --torchscript "${ts}" \
      --input-bank "${bank}" \
      --precision fp32 \
      --batch-size 1 \
      --num-warmup 20 \
      --num-iters 30 \
      --device cuda \
      --results-dir "${OUTDIR}" \
      --result-tag "${tag}" \
      --notes "nsys resolution-coverage extension"
}

run_case "darcy_r141_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r141_seed0_best.pt \
  artifacts/torchscript/darcy_fno_base_r141_seed0.ts \
  artifacts/benchmark_inputs/darcy_r141_bank.pt

run_case "darcy_r211_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r211_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r211_seed1.ts \
  artifacts/benchmark_inputs/darcy_r211_bank.pt

run_case "darcy_r421_base_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r421_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r421_seed1.ts \
  artifacts/benchmark_inputs/darcy_r421_bank.pt

run_case "burgers_r2048_ts_fp32" \
  artifacts/checkpoints/burgers_fno_base_seed3_best.pt \
  artifacts/torchscript/burgers_fno_base_seed3.ts \
  artifacts/benchmark_inputs/burgers_r2048_bank.pt

run_case "burgers_r4096_ts_fp32" \
  artifacts/checkpoints/burgers_fno_base_r4096_seed0_best.pt \
  artifacts/torchscript/burgers_fno_base_r4096_seed0.ts \
  artifacts/benchmark_inputs/burgers_r4096_bank.pt

run_case "burgers_r8192_ts_fp32" \
  artifacts/checkpoints/burgers_fno_base_r8192_seed1_best.pt \
  artifacts/torchscript/burgers_fno_base_r8192_seed1.ts \
  artifacts/benchmark_inputs/burgers_r8192_bank.pt

echo "[DONE] FNO extended nsys profiling matrix"
