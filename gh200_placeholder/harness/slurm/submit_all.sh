#!/bin/bash
# Submit the GH200 FP32 benchmark matrix — one single-GPU job per operator
# family (5 jobs), or a chosen subset. Run from a Delta AI login node.
#
#   bash submit_all.sh              # all 5 families
#   bash submit_all.sh fno wno      # only these
#
# Env: RUN_TAG REPS DURATION WARMUP INTERVAL_MS (see matrix.sh)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TAG="${RUN_TAG:-gh200_fp32_$(date +%Y%m%d)}"
REPS="${REPS:-3}"
DURATION="${DURATION:-120}"
WARMUP="${WARMUP:-20}"
INTERVAL_MS="${INTERVAL_MS:-200}"
SB=bench.sbatch

FAMILIES=("$@")
if [[ ${#FAMILIES[@]} -eq 0 ]]; then
  FAMILIES=(fno deeponet wno sp2gno hx)
fi

echo ">>> submitting to <PARTITION> / <ACCOUNT> | RUN_TAG=$RUN_TAG REPS=$REPS DURATION=$DURATION"
for fam in "${FAMILIES[@]}"; do
  jid=$(sbatch --parsable --job-name="gh200_${fam}" \
        --export=ALL,FAMILY="$fam",RUN_TAG="$RUN_TAG",REPS="$REPS",DURATION="$DURATION",WARMUP="$WARMUP",INTERVAL_MS="$INTERVAL_MS" \
        "$SB")
  printf '  submitted %-18s job %s\n' "$fam" "$jid"
done
echo ">>> track with: squeue --me ; logs in slurm/logs/gh200_<fam>-<jobid>.out"
