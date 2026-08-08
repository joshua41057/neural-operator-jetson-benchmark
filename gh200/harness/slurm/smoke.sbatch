#!/bin/bash
# Fast end-to-end validation: 1 rep of every family with a short measure window.
# Confirms each GH200 harness imports, loads its checkpoint/bank/graph, samples
# nvidia-smi power, and writes a result before committing the full matrix.
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION_INTERACTIVE>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --output=<REPO>/slurm/logs/%x-%j.out
#SBATCH --error=<REPO>/slurm/logs/%x-%j.out
set -u

source <CONDA>/etc/profile.d/conda.sh
conda activate wno_sp2gno
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RUN_TAG="${RUN_TAG:-gh200_smoke}"
export REPS=1
export DURATION="${DURATION:-8}"
export WARMUP="${WARMUP:-4}"
export INTERVAL_MS=200

echo "=== smoke node $(hostname) gpu=$CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

bash <REPO>/slurm/matrix.sh "${1:-all}"
echo "=== smoke done rc=$? ==="
