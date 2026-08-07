#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/jjyoo3"
OUT="${ROOT}/table4_model_scale_medians.txt"
: > "$OUT"

echo "### Candidate CSV/JSON files containing model-scale median latency" >> "$OUT"
find "${ROOT}/EDCNO" "${ROOT}/EDCNO_DeepONet" -type f \
  \( -name "*.csv" -o -name "*.json" \) \
  | grep -Ei "deploy|latency|fp32|summary|paper|inference" \
  | sort >> "$OUT" || true

echo -e "\n### FNO model-scale rows" >> "$OUT"
grep -RInE "burgers_fno_(small|base|large)|darcy_fno_(small|base|large)" \
  "${ROOT}/EDCNO/results" \
  --include="*.csv" --include="*.json" 2>/dev/null \
  | grep -Ei "median|p50|mean|p95|latency|params|rel|l2|torchscript|eager" \
  | head -n 500 >> "$OUT" || true

echo -e "\n### DeepONet model-scale rows" >> "$OUT"
grep -RInE "burgers_deeponet_(small|base|large)|darcy_deeponet_(small|base|large)" \
  "${ROOT}/EDCNO_DeepONet/results" \
  --include="*.csv" --include="*.json" 2>/dev/null \
  | grep -Ei "median|p50|mean|p95|latency|params|rel|l2|torchscript|eager" \
  | head -n 500 >> "$OUT" || true

echo -e "\n### Headers of likely CSVs" >> "$OUT"
for f in \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_main_deployability_table.csv" \
  "${ROOT}/EDCNO/results/artifacts/paper_fno_fp32_deployment_table.csv" \
  "${ROOT}/EDCNO/results/artifacts/fno_fp32_deployment_summary.csv" \
  "${ROOT}/EDCNO_DeepONet/results/artifacts/deeponet_fp32_deployment_summary.csv"
do
  echo -e "\n--- $f ---" >> "$OUT"
  if [ -f "$f" ]; then
    head -n 20 "$f" >> "$OUT"
  else
    echo "[MISSING]" >> "$OUT"
  fi
done

echo "WROTE $OUT"
wc -l "$OUT"
ls -lh "$OUT"
