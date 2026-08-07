#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1

NCU_BIN="$(readlink -f "$(which ncu)")"
OUTDIR="results/jetson_fno_profile_ncu_sched"

export_one () {
  local tag="$1"
  local rep="${OUTDIR}/${tag}.ncu-rep"

  echo "=================================================="
  echo "[EXPORT] ${tag}"
  echo "=================================================="

  "${NCU_BIN}" \
    --import "${rep}" \
    --page details \
    > "${OUTDIR}/${tag}_details.txt"

  "${NCU_BIN}" \
    --import "${rep}" \
    --page details \
    --csv \
    > "${OUTDIR}/${tag}_details.csv"

  "${NCU_BIN}" \
    --import "${rep}" \
    --page raw \
    --csv \
    > "${OUTDIR}/${tag}_raw.csv"

  "${NCU_BIN}" \
    --import "${rep}" \
    --page details \
    --print-summary per-kernel \
    > "${OUTDIR}/${tag}_per_kernel.txt"

  "${NCU_BIN}" \
    --import "${rep}" \
    --page details \
    --print-summary per-kernel \
    --csv \
    > "${OUTDIR}/${tag}_per_kernel.csv"
}

export_one "darcy_r281_ts_fp32_ncu_sched"
export_one "darcy_large_on421_ts_fp32_ncu_sched"