#!/bin/bash
# Burgers sweep (WNO + Sp2GNO): model-scale variants @ r2048 + resolution scaling.
# Runs sequentially; 1D so light-to-medium memory. Concurrent with run_darcy.sh.
# Portable env activation: set CONDA_ENV to override the env name (default ~/GraphWNO).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$CONDA_EXE" ]; then source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"; fi
conda activate "${CONDA_ENV:-$HOME/GraphWNO}" 2>/dev/null || echo "warn: could not activate ${CONDA_ENV:-$HOME/GraphWNO}; assuming env already active"

run() { echo "### $(date '+%F %T') START $*"; python "$@"; echo "### $(date '+%F %T') END   $*"; }

# ---- Sp2GNO Burgers (fast: small N) ----
run train_sp2gno_burgers.py --variant small --res 2048 --ckpt sp2gno_burgers_small_s2048.pth
run train_sp2gno_burgers.py --variant base  --res 2048 --ckpt sp2gno_burgers_base_s2048.pth
run train_sp2gno_burgers.py --variant large --res 2048 --ckpt sp2gno_burgers_large_s2048.pth
run train_sp2gno_burgers.py --variant base  --res 4096 --ckpt sp2gno_burgers_base_s4096.pth   # res-scaling

# ---- WNO Burgers ----
run train_wno_burgers.py --variant small --res 2048 --ckpt wno_burgers_small_r2048.pth
run train_wno_burgers.py --variant base  --res 2048 --ckpt wno_burgers_base_r2048.pth
run train_wno_burgers.py --variant large --res 2048 --ckpt wno_burgers_large_r2048.pth
run train_wno_burgers.py --variant base  --res 4096 --ckpt wno_burgers_base_r4096.pth          # res-scaling
run train_wno_burgers.py --variant base  --res 8192 --ckpt wno_burgers_base_r8192.pth          # res-scaling

echo "ALL BURGERS DONE $(date '+%F %T')"
