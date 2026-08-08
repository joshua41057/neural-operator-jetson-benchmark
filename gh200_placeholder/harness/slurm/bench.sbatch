#!/bin/bash
# Generic single-GPU GH200 benchmark job for Delta AI (NCSA, aarch64 / Hopper).
# Submitted by submit_all.sh, which sets FAMILY + RUN_TAG + knobs via --export.
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=<REPO>/slurm/logs/%x-%j.out
#SBATCH --error=<REPO>/slurm/logs/%x-%j.out
set -u

source <CONDA>/etc/profile.d/conda.sh
conda activate wno_sp2gno
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FAMILY="${FAMILY:?set FAMILY}"

echo "==================================================================="
echo "job=$SLURM_JOB_ID family=$FAMILY run_tag=${RUN_TAG:-?} node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "start=$(date '+%F %T')  reps=${REPS:-3} duration=${DURATION:-120}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "==================================================================="

bash <REPO>/slurm/matrix.sh "$FAMILY"
rc=$?

echo "==================================================================="
echo "end=$(date '+%F %T') rc=$rc"
exit $rc
