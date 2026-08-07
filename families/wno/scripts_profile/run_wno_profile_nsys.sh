#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/WNO_Sp2GNO_Benchmarks || exit 1

PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
BANKDIR="/home/jetson/data/wno_inference_banks_exact"
OUTDIR="results/profiles/wno_nsys"
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
  local dataset="$2"
  local ckpt="$3"
  local bank="$4"

  run_profile "${tag}" \
    "${PYBIN}" bench_wno_jetson_exact.py \
      --case-id "${tag}_profile" \
      --dataset "${dataset}" \
      --checkpoint "${ckpt}" \
      --bank "${bank}" \
      --precision-mode fp32_strict \
      --warmup-seconds 2 \
      --measure-seconds 3 \
      --run-tag profile_nsys \
      --compute-full-eval 0 \
      --compute-perturbation 0 \
      --results-root "${OUTDIR}/harness_out"
}

# Burgers
run_case "wno_burgers_small_r2048" burgers checkpoints/wno_burgers_small_r2048.pth "${BANKDIR}/burgers_r2048_bank.pt"
run_case "wno_burgers_base_r2048"  burgers checkpoints/wno_burgers_base_r2048.pth  "${BANKDIR}/burgers_r2048_bank.pt"
run_case "wno_burgers_large_r2048" burgers checkpoints/wno_burgers_large_r2048.pth "${BANKDIR}/burgers_r2048_bank.pt"
run_case "wno_burgers_base_r4096"  burgers checkpoints/wno_burgers_base_r4096.pth  "${BANKDIR}/burgers_r4096_bank.pt"
run_case "wno_burgers_base_r8192"  burgers checkpoints/wno_burgers_base_r8192.pth  "${BANKDIR}/burgers_r8192_bank.pt"

# Darcy
run_case "wno_darcy_small_r141" darcy checkpoints/wno_darcy_small_r141.pth "${BANKDIR}/darcy_r141_bank.pt"
run_case "wno_darcy_base_r141"  darcy checkpoints/wno_darcy_base_r141.pth  "${BANKDIR}/darcy_r141_bank.pt"
run_case "wno_darcy_large_r141" darcy checkpoints/wno_darcy_large_r141.pth "${BANKDIR}/darcy_r141_bank.pt"
run_case "wno_darcy_base_r281"  darcy checkpoints/wno_darcy_base_r281.pth  "${BANKDIR}/darcy_r281_bank.pt"
run_case "wno_darcy_base_r421"  darcy checkpoints/wno_darcy_base_r421.pth  "${BANKDIR}/darcy_r421_bank.pt"

echo "[DONE] WNO nsys profiling matrix (10 cases)"
