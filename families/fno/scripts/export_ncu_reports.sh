#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1

NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
OUTDIR="results/jetson_fno_profile_ncu"

export_one () {
  local tag="$1"
  local rep="${OUTDIR}/${tag}.ncu-rep"

  echo "=================================================="
  echo "[EXPORT] ${tag}"
  echo "=================================================="

  # Human-readable details
  "${NCU_BIN}" --import "${rep}" \
    --page details \
    > "${OUTDIR}/${tag}_details.txt"

  # CSV details
  "${NCU_BIN}" --import "${rep}" \
    --page details \
    --csv \
    > "${OUTDIR}/${tag}_details.csv"

  # Raw metrics, CSV
  "${NCU_BIN}" --import "${rep}" \
    --page raw \
    --csv \
    > "${OUTDIR}/${tag}_raw.csv"

  # Per-kernel summary view
  "${NCU_BIN}" --import "${rep}" \
    --page details \
    --print-summary per-kernel \
    > "${OUTDIR}/${tag}_per_kernel.txt"

  "${NCU_BIN}" --import "${rep}" \
    --page details \
    --print-summary per-kernel \
    --csv \
    > "${OUTDIR}/${tag}_per_kernel.csv"
}

export_one "darcy_r85_ts_fp32_ncu"
export_one "darcy_r281_ts_fp32_ncu"