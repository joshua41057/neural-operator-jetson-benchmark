#!/usr/bin/env bash
set -euo pipefail

cd /home/jetson/jjyoo3/EDCNO_DeepONet || exit 1

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"

OUTDIR="results/profiles/deeponet_nsys_extended"
mkdir -p "$OUTDIR"

run_case () {
  local tag="$1"
  local ckpt="$2"
  local ts="$3"
  local bank="$4"

  local outbase="$OUTDIR/${tag}"
  local json_out="$OUTDIR/${tag}_forward.json"

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  rm -f "${outbase}.nsys-rep" "${outbase}.sqlite" "$json_out"

  set +e
  nsys profile \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --cuda-memory-usage=false \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    --stop-on-exit=true \
    --output="$outbase" \
    "${PYBIN}" -m src.eval.profile_forward_nvtx \
      --mode torchscript \
      --checkpoint "$ckpt" \
      --torchscript "$ts" \
      --input-bank "$bank" \
      --precision fp32_strict \
      --batch-size 1 \
      --warmup 20 \
      --profile-iters 80 \
      --device cuda \
      --result-json "$json_out"
  STATUS=$?
  set -e

  if [[ ${STATUS} -eq 0 ]]; then
    echo "[ OK ] ${tag}"
    nsys stats \
      --force-export=true \
      --report nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
      "${outbase}.nsys-rep" \
      > "${outbase}_nsys_stats.txt" 2>&1 || true
  else
    echo "[FAIL] ${tag} exit=${STATUS}"
  fi
}

run_case "burgers_base_r8192_ts_fp32" \
  artifacts/checkpoints/burgers_deeponet_base_r8192_seed0_best.pt \
  artifacts/torchscript/burgers_deeponet_base_r8192_seed0.ts \
  artifacts/benchmark_inputs/burgers_r8192_bank.pt

run_case "darcy_base_r421_ts_fp32" \
  artifacts/checkpoints/darcy_deeponet_base_r421_seed2_best.pt \
  artifacts/torchscript/darcy_deeponet_base_r421_seed2.ts \
  artifacts/benchmark_inputs/darcy_r421_bank.pt

echo "[DONE] DeepONet extended nsys profiling (2 cases)"
