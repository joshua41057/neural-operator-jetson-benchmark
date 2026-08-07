#!/usr/bin/env bash
set -euo pipefail

cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026 || exit 1

PYBIN="/home/jetson/miniforge3/envs/vs_wno/bin/python"
CKPTDIR="/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints"
OUTDIR="results/profiles/sp2gno_nsys"
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
    --warmup 3
    --min_duration_s 2
    --min_cycles 1
    --rep 1
  )

  if [[ "${dataset}" == "burgers" ]]; then
    common_args+=(--sub "${sub}" --burgers_split Jetson_data/burgers_split.json)
  else
    common_args+=(--r "${r}" --ntrain 900 --nval 100 --ntest 200)
  fi

  run_profile "${tag}" "${PYBIN}" bench_sp2gno_jetson_exact.py "${common_args[@]}"
}

# Burgers: 8192 / sub = resolution.
run_case sp2gno_burgers_small_s2048 burgers "${CKPTDIR}/sp2gno_burgers_small_s2048.pth" 13 4 0 8
run_case sp2gno_burgers_base_s2048  burgers "${CKPTDIR}/sp2gno_burgers_base_s2048.pth"  24 4 0 8
run_case sp2gno_burgers_large_s2048 burgers "${CKPTDIR}/sp2gno_burgers_large_s2048.pth" 45 4 0 8
run_case sp2gno_burgers_base_s4096  burgers "${CKPTDIR}/sp2gno_burgers_base_s4096.pth"  24 2 0 8

# Darcy: 421 / r = resolution.
run_case sp2gno_darcy_small_r141 darcy "${CKPTDIR}/sp2gno_darcy_small_r141.pth" 13 0 3 20
run_case sp2gno_darcy_base_r141  darcy "${CKPTDIR}/sp2gno_darcy_base_r141.pth"  24 0 3 20
run_case sp2gno_darcy_large_r141 darcy "${CKPTDIR}/sp2gno_darcy_large_r141.pth" 45 0 3 20
run_case sp2gno_darcy_base_r211  darcy "${CKPTDIR}/sp2gno_darcy_base_r211.pth"  24 0 2 20

echo "[DONE] sp2gno nsys profiling matrix (8 cases)"
