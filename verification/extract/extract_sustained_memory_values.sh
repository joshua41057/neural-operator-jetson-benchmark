#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/jjyoo3"
OUT="${ROOT}/sustained_memory_values.txt"
: > "$OUT"

echo "### Sustained energy/memory source tables" >> "$OUT"
for f in \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_long_energy_table.csv" \
  "${ROOT}/EDCNO/results/artifacts/fno_energy_long_summary.csv" \
  "${ROOT}/EDCNO/results/artifacts/tegrastats_summary.csv" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_long_energy_summary.csv" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_tegrastats_board_summary.csv"
do
  echo -e "\n--- $f ---" >> "$OUT"
  if [ -f "$f" ]; then
    head -n 80 "$f" >> "$OUT"
  else
    echo "[MISSING]" >> "$OUT"
  fi
done

echo -e "\n### Raw JSON search for peak CUDA allocation in sustained runs" >> "$OUT"
grep -RInE "peak_cuda|max_memory|cuda.*alloc|allocated_mb|memory_allocated|Darcy.*281|darcy.*281|fp16_native|fno.*281" \
  "${ROOT}/EDCNO/results" "${ROOT}/EDCNO_DeepONet/results" \
  --include="*.json" --include="*.csv" 2>/dev/null \
  | head -n 1000 >> "$OUT" || true

echo "WROTE $OUT"
wc -l "$OUT"
ls -lh "$OUT"
