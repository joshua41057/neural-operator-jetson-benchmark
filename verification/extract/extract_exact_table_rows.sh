#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/jjyoo3"
OUT="${ROOT}/exact_table_rows_for_revision.txt"
: > "$OUT"

add_section () {
  echo "" >> "$OUT"
  echo "######################################################################" >> "$OUT"
  echo "# $1" >> "$OUT"
  echo "######################################################################" >> "$OUT"
}

dump_matches () {
  local title="$1"
  local file="$2"
  local pattern="$3"
  add_section "$title"
  echo "FILE: $file" >> "$OUT"
  if [ -f "$file" ]; then
    head -n 1 "$file" >> "$OUT"
    grep -Ei "$pattern" "$file" >> "$OUT" || true
  else
    echo "[MISSING]" >> "$OUT"
  fi
}

dump_file () {
  local title="$1"
  local file="$2"
  add_section "$title"
  echo "FILE: $file" >> "$OUT"
  if [ -f "$file" ]; then
    cat "$file" >> "$OUT"
  else
    echo "[MISSING]" >> "$OUT"
  fi
}

# -------------------------------------------------------------------
# FNO latency / scaling / backend / validity source rows
# -------------------------------------------------------------------
dump_matches "FNO model-scale latency rows" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_main_deployability_table.csv" \
  "burgers_fno_(small|base|large)|darcy_fno_(small|base|large)"

dump_matches "FNO resolution-scaling rows" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_resolution_scaling_table.csv" \
  "burgers_fno_base_r(512|1024|2048|4096)|darcy_fno_base_r(85|141|211|281)"

dump_matches "FNO backend sensitivity rows" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_backend_speedup_table.csv" \
  "burgers_fno_base|darcy_fno_base_r85|darcy_fno_base_r141|darcy_fno_base_r281|darcy_fno_large"

# -------------------------------------------------------------------
# DeepONet latency / scaling / backend / validity source rows
# -------------------------------------------------------------------
dump_file "DeepONet FP32 deployment summary" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_fp32_deployment_summary.csv"

dump_file "DeepONet validity table" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_validity_table.csv"

# -------------------------------------------------------------------
# Precision: FNO status/TF32/failures and DeepONet perturbation
# -------------------------------------------------------------------
dump_matches "FNO TF32 latency-effect rows for main cases" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_precision_tf32_table.csv" \
  "burgers_fno_base,|burgers_fno_base_r2048|darcy_fno_base,|darcy_fno_base_r141|darcy_fno_base_r281|darcy_fno_large,"

dump_file "FNO precision failure summary" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_precision_failure_summary_table.csv"

dump_file "FNO precision failure examples" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_precision_failure_examples_table.csv"

dump_file "DeepONet precision summary" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_precision_summary.csv"

dump_file "DeepONet precision numerics" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_precision_numerics.csv"

# Try to locate FNO numerical perturbation if it exists under any name.
add_section "SEARCH: FNO numerical perturbation candidates"
grep -RInE "perturb|relative.*L2|rel_l2|relative_l2|tf32.*fp32|fp32.*tf32|output.*diff|numerical" \
  "${ROOT}/EDCNO/results" \
  --include="*.csv" --include="*.json" --include="*.txt" 2>/dev/null \
  | head -n 500 >> "$OUT" || true

# -------------------------------------------------------------------
# Sustained energy / memory / thermal
# -------------------------------------------------------------------
dump_file "FNO sustained energy table" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_long_energy_table.csv"

dump_file "FNO energy long summary" \
  "${ROOT}/EDCNO/results/artifacts/fno_energy_long_summary.csv"

dump_file "DeepONet long energy summary" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_long_energy_summary.csv"

dump_file "DeepONet tegrastats board summary" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_tegrastats_board_summary.csv"

add_section "SEARCH: sustained CUDA allocation candidates for dash rows"
grep -RInE "cuda.*alloc|peak_cuda|max_memory|allocated_mb|Darcy.*281|fp16_native|darcy_base_r281" \
  "${ROOT}/EDCNO/results" "${ROOT}/EDCNO_DeepONet/results" \
  --include="*.csv" --include="*.json" 2>/dev/null \
  | head -n 800 >> "$OUT" || true

# -------------------------------------------------------------------
# Frontier
# -------------------------------------------------------------------
dump_file "FNO frontier summary" \
  "${ROOT}/EDCNO/results/artifacts/paper_frontier_summary.csv"

dump_file "FNO OOM frontier table" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_oom_frontier_table.csv"

dump_file "DeepONet query frontier summary if present" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_query_frontier_summary.csv"

add_section "SEARCH: DeepONet frontier candidates"
grep -RInE "frontier|query|chunk|q=|chunk_size|cuda.*alloc|energy" \
  "${ROOT}/EDCNO_DeepONet/results" \
  --include="*.csv" --include="*.json" 2>/dev/null \
  | head -n 800 >> "$OUT" || true

# -------------------------------------------------------------------
# Profiling
# -------------------------------------------------------------------
dump_file "FNO kernel attribution" \
  "${ROOT}/EDCNO/results/artifacts/paper_kernel_attribution.csv"

dump_file "FNO NCU kernel summary" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_ncu_kernel_summary_table.csv"

dump_file "FNO NSYS summary" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_nsys_summary_table.csv"

dump_file "DeepONet NCU class aggregate" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_ncu_class_aggregate.csv"

dump_file "DeepONet NCU paper kernel selection" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_ncu_paper_kernel_selection.csv"

dump_file "DeepONet NSYS forward summary" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_nsys_forward_summary.csv"

echo "WROTE $OUT"
wc -l "$OUT"
ls -lh "$OUT"
