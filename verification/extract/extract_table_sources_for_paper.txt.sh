#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/jjyoo3"
OUT="${ROOT}/table_sources_for_paper.txt"
: > "$OUT"

dump_file () {
  local f="$1"
  echo "" >> "$OUT"
  echo "######################################################################" >> "$OUT"
  echo "# FILE: $f" >> "$OUT"
  echo "######################################################################" >> "$OUT"
  if [ -f "$f" ]; then
    echo "[rows: $(wc -l < "$f")]" >> "$OUT"
    sed -e 's/\r$//' "$f" >> "$OUT"
  else
    echo "[MISSING]" >> "$OUT"
  fi
}

dump_head () {
  local f="$1"
  local n="${2:-120}"
  echo "" >> "$OUT"
  echo "######################################################################" >> "$OUT"
  echo "# FILE HEAD: $f" >> "$OUT"
  echo "######################################################################" >> "$OUT"
  if [ -f "$f" ]; then
    echo "[rows: $(wc -l < "$f")]" >> "$OUT"
    sed -e 's/\r$//' "$f" | head -n "$n" >> "$OUT"
  else
    echo "[MISSING]" >> "$OUT"
  fi
}

# FNO paper source tables
dump_file "${ROOT}/EDCNO/results/artifacts/paper_fno_precision_tf32_table.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/precision_tf32_vs_fp32.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_fno_precision_failure_summary_table.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_fno_precision_failure_examples_table.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_fno_long_energy_table.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/fno_energy_long_summary.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/tegrastats_summary.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_frontier_summary.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_kernel_attribution.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_fno_ncu_kernel_summary_table.csv"
dump_file "${ROOT}/EDCNO/results/artifacts/paper_fno_nsys_summary_table.csv"

# DeepONet paper source tables
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_fp32_deployment_summary.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_precision_summary.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_precision_numerics.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_long_energy_summary.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_tegrastats_board_summary.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_ncu_class_aggregate.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_ncu_paper_kernel_selection.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_nsys_forward_summary.csv"
dump_file "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_profile_plan.csv"

# Search raw JSONs for missing CUDA allocator values in rows that currently have dash.
echo "" >> "$OUT"
echo "######################################################################" >> "$OUT"
echo "# RAW JSON SEARCH: CUDA allocation candidates for dash rows" >> "$OUT"
echo "######################################################################" >> "$OUT"

grep -RInE "peak_cuda|peak.*alloc|max_memory|cuda.*mb|allocated_mb|Darcy.*281|fp16_native|fno.*281" \
  "${ROOT}/EDCNO/results" "${ROOT}/EDCNO_DeepONet/results" \
  --include="*.json" --include="*.csv" 2>/dev/null \
  | head -n 1000 >> "$OUT" || true

echo "" >> "$OUT"
echo "WROTE $OUT" >> "$OUT"
wc -l "$OUT"
ls -lh "$OUT"
