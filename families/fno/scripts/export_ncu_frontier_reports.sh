#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1

NCU_BIN="$(readlink -f "$(which ncu)")"
OUTDIR="results/jetson_fno_profile_ncu_frontier"

TAG="darcy_large_on421_ts_fp32_ncu"
REP="${OUTDIR}/${TAG}.ncu-rep"

echo "=================================================="
echo "[EXPORT] ${TAG}"
echo "=================================================="

"${NCU_BIN}" \
  --import "${REP}" \
  --page details \
  > "${OUTDIR}/${TAG}_details.txt"

"${NCU_BIN}" \
  --import "${REP}" \
  --page details \
  --csv \
  > "${OUTDIR}/${TAG}_details.csv"

"${NCU_BIN}" \
  --import "${REP}" \
  --page raw \
  --csv \
  > "${OUTDIR}/${TAG}_raw.csv"

"${NCU_BIN}" \
  --import "${REP}" \
  --page details \
  --print-summary per-kernel \
  > "${OUTDIR}/${TAG}_per_kernel.txt"

"${NCU_BIN}" \
  --import "${REP}" \
  --page details \
  --print-summary per-kernel \
  --csv \
  > "${OUTDIR}/${TAG}_per_kernel.csv"