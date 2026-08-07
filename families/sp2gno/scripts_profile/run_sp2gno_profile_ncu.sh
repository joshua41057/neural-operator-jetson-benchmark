#!/usr/bin/env bash
# Run with sudo (ncu requires root on this Jetson):
#   sudo bash scripts_profile/run_sp2gno_profile_ncu.sh
set -euo pipefail

export HOME=/home/jetson
cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026 || exit 1

# NVMAP_IOC_GET_FD / "LaunchFailed" crashes on this Jetson are NOT about the CUDA
# caching allocator (PYTORCH_NO_CUDA_MEMORY_CACHING does nothing here -- verified:
# no code in this repo reads it, and stock PyTorch doesn't use it to gate the
# allocator either). Log evidence from FNO/WNO crashes both died at kernel-launch
# index ~480 within one ncu session -- an apparent nvmap/driver resource ceiling
# tied to total kernel launches profiled, not kernel type. Fix: --warmup 0 keeps
# the session to exactly one forward pass, well under that ceiling.

PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
CKPTDIR="/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints"
OUTDIR="results/profiles/sp2gno_ncu"
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

  # sp2gno's graph message-passing path launches enough distinct kernels per
  # forward pass to risk the same ~480-launch ceiling WNO hit. Restrict to the
  # dominant kernel classes identified from nsys (movement, dense, reduction)
  # and hard-cap launch-count as a second safety net.
  "${NCU_BIN}" \
    --force-overwrite \
    --target-processes all \
    --replay-mode kernel \
    --cache-control all \
    --clock-control none \
    --set detailed \
    --kernel-name "regex:CatArrayBatchedCopy|indexSelectLargeIndex|scatter_gather_elementwise_kernel|vectorized_layer_norm_kernel|ampere_sgemm|gemvx" \
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
  local width="$4"
  local sub="$5"
  local r="$6"
  local k="$7"

  local common_args=(
    --case_id "${tag}"
    --run_name "${tag}_profile"
    --suite_root "${OUTDIR}/harness_out"
    --dataset "${dataset}"
    --data_dir /home/jetson/data
    --cache_dir cache
    --ckpt "${ckpt}"
    --width "${width}"
    --n_layers 6
    --num_freq 64
    --k "${k}"
    --precision fp32_strict
    --warmup 0
    --min_duration_s 0.001
    --min_cycles 1
    --rep 1
  )

  if [[ "${dataset}" == "burgers" ]]; then
    common_args+=(--sub "${sub}" --burgers_split Jetson_data/burgers_split.json)
  else
    common_args+=(--r "${r}" --ntrain 900 --nval 100 --ntest 200)
  fi

  run_ncu "${tag}" "${PYBIN}" bench_sp2gno_jetson_exact.py "${common_args[@]}"
}

# Paper-selected 5: Burgers small/large scale contrast + Darcy small/large scale
# contrast + one model-capacity (large) point, mirroring the DeepONet evidence structure.
run_case sp2gno_burgers_base_s2048 burgers "${CKPTDIR}/sp2gno_burgers_base_s2048.pth" 24 4 0 8
run_case sp2gno_burgers_base_s4096 burgers "${CKPTDIR}/sp2gno_burgers_base_s4096.pth" 24 2 0 8
run_case sp2gno_darcy_base_r141    darcy   "${CKPTDIR}/sp2gno_darcy_base_r141.pth"    24 0 3 20
run_case sp2gno_darcy_base_r211    darcy   "${CKPTDIR}/sp2gno_darcy_base_r211.pth"    24 0 2 20
run_case sp2gno_darcy_large_r141   darcy   "${CKPTDIR}/sp2gno_darcy_large_r141.pth"   45 0 3 20

echo "[DONE] sp2gno ncu profiling (5 target cases)"
echo "Now run: sudo chown -R jetson:jetson results/profiles/sp2gno_ncu"
