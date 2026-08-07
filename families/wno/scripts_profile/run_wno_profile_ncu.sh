#!/usr/bin/env bash
# Run with sudo (ncu requires root on this Jetson):
#   sudo bash scripts_profile/run_wno_profile_ncu.sh
set -euo pipefail

export HOME=/home/jetson
cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks || exit 1

# NVMAP_IOC_GET_FD / "LaunchFailed" crashes on this Jetson are NOT about the CUDA
# caching allocator (PYTORCH_NO_CUDA_MEMORY_CACHING does nothing here -- verified:
# no code in this repo reads it, and stock PyTorch doesn't use it to gate the
# allocator either). Log evidence (this exact case crashed at kernel-launch index
# 480 out of ~372 kernels/forward-pass, i.e. partway into the *second* pass) points
# to an nvmap/driver resource ceiling tied to total kernel launches profiled in one
# ncu session, not kernel type. Fix: --warmup-seconds 0 keeps the session to exactly
# one forward pass, well under that ceiling.

PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
BANKDIR="/home/jetson/data/wno_inference_banks_exact"
OUTDIR="results/profiles/wno_ncu"
mkdir -p "${OUTDIR}"

run_ncu () {
  local tag="$1"
  shift

  if [[ -s "${OUTDIR}/${tag}.ncu-rep" ]]; then
    echo "[SKIP] ${tag} (already exists)"
    return 0
  fi

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  # Even with a single forward pass, WNO's per-layer conv/wavelet kernels alone
  # exceed the ~480-launch ceiling at large resolutions (r8192 crashed here).
  # Restrict to the dominant kernel classes identified from nsys (conv/wavelet,
  # movement, activation, dense) and hard-cap launch-count as a second safety net.
  "${NCU_BIN}" \
    --force-overwrite \
    --target-processes all \
    --replay-mode kernel \
    --cache-control all \
    --clock-control none \
    --set detailed \
    --kernel-name "regex:dgrad2d_grouped_direct_kernel|conv_depthwise2d_forward_kernel|dgrad_engine|scalePackedTensor_kernel|CatArrayBatchedCopy|GeluCUDAKernelImpl|mish_kernel|ampere_sgemm" \
    --launch-count 150 \
    --export "${OUTDIR}/${tag}" \
    --log-file "${OUTDIR}/${tag}.log" \
    "$@"

  if [[ ! -f "${OUTDIR}/${tag}.ncu-rep" ]]; then
    echo "[FAIL] ${tag} -- missing report, tail of log:"
    tail -60 "${OUTDIR}/${tag}.log" || true
  fi
}

run_case () {
  local tag="$1"
  local dataset="$2"
  local ckpt="$3"
  local bank="$4"

  run_ncu "${tag}" \
    "${PYBIN}" bench_wno_jetson_exact.py \
      --case-id "${tag}_profile" \
      --dataset "${dataset}" \
      --checkpoint "${ckpt}" \
      --bank "${bank}" \
      --precision-mode fp32_strict \
      --warmup-seconds 0 \
      --measure-seconds 0.001 \
      --run-tag profile_ncu \
      --compute-full-eval 0 \
      --compute-perturbation 0 \
      --results-root "${OUTDIR}/harness_out"
}

# Paper-selected 5: Burgers small/large scale contrast + Darcy small/large scale
# contrast + one model-capacity (large) point, mirroring the DeepONet evidence structure.
run_case "wno_burgers_base_r2048" burgers checkpoints/wno_burgers_base_r2048.pth "${BANKDIR}/burgers_r2048_bank.pt"
run_case "wno_burgers_base_r8192" burgers checkpoints/wno_burgers_base_r8192.pth "${BANKDIR}/burgers_r8192_bank.pt"
run_case "wno_darcy_base_r141"    darcy   checkpoints/wno_darcy_base_r141.pth    "${BANKDIR}/darcy_r141_bank.pt"
run_case "wno_darcy_base_r421"    darcy   checkpoints/wno_darcy_base_r421.pth    "${BANKDIR}/darcy_r421_bank.pt"
run_case "wno_darcy_large_r141"   darcy   checkpoints/wno_darcy_large_r141.pth   "${BANKDIR}/darcy_r141_bank.pt"

echo "[DONE] WNO ncu profiling (5 target cases)"
echo "Now run: sudo chown -R jetson:jetson results/profiles/wno_ncu"
