#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.export_torchscript --checkpoint checkpoints/darcy_fno_small/seed0/best.pt --output results/darcy_fno_small_seed0.ts
python -m src.eval.export_torchscript --checkpoint checkpoints/darcy_fno_base/seed0/best.pt --output results/darcy_fno_base_seed0.ts
python -m src.eval.export_torchscript --checkpoint checkpoints/darcy_fno_large/seed0/best.pt --output results/darcy_fno_large_seed0.ts
python -m src.eval.export_torchscript --checkpoint checkpoints/burgers_fno_small/seed0/best.pt --output results/burgers_fno_small_seed0.ts
python -m src.eval.export_torchscript --checkpoint checkpoints/burgers_fno_base/seed0/best.pt --output results/burgers_fno_base_seed0.ts
python -m src.eval.export_torchscript --checkpoint checkpoints/burgers_fno_large/seed0/best.pt --output results/burgers_fno_large_seed0.ts
