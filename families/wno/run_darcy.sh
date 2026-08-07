#!/bin/bash
# Darcy sweep (WNO + Sp2GNO): model-scale variants @ r141 + resolution scaling.
# WNO first (no eigendecomposition). Sp2GNO darcy builds a shared-graph cache per
# resolution (eigh of an NxN Laplacian) the first time it is used, then reuses it.
# Portable env activation: set CONDA_ENV to override the env name (default ~/GraphWNO).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$CONDA_EXE" ]; then source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"; fi
conda activate "${CONDA_ENV:-$HOME/GraphWNO}" 2>/dev/null || echo "warn: could not activate ${CONDA_ENV:-$HOME/GraphWNO}; assuming env already active"

run() { echo "### $(date '+%F %T') START $*"; python "$@"; echo "### $(date '+%F %T') END   $*"; }

# ---- WNO Darcy ----
run train_wno_darcy.py --variant small --res 141 --ckpt wno_darcy_small_r141.pth
run train_wno_darcy.py --variant base  --res 141 --ckpt wno_darcy_base_r141.pth
run train_wno_darcy.py --variant large --res 141 --ckpt wno_darcy_large_r141.pth
run train_wno_darcy.py --variant base  --res 281 --ckpt wno_darcy_base_r281.pth                # res-scaling
run train_wno_darcy.py --variant base  --res 421 --ckpt wno_darcy_base_r421.pth                # res-scaling

# ---- Sp2GNO Darcy (graph cache for s=141 built once on the first run) ----
run train_sp2gno_darcy.py --variant small --res 141 --batch_size 10 --ckpt sp2gno_darcy_small_r141.pth
run train_sp2gno_darcy.py --variant base  --res 141 --batch_size 10 --ckpt sp2gno_darcy_base_r141.pth
run train_sp2gno_darcy.py --variant large --res 141 --batch_size 8  --ckpt sp2gno_darcy_large_r141.pth
run train_sp2gno_darcy.py --variant base  --res 211 --batch_size 5  --ckpt sp2gno_darcy_base_r211.pth   # res-scaling

echo "ALL DARCY DONE $(date '+%F %T')"
