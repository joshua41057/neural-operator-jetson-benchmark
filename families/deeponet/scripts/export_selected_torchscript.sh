#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

MANIFEST="jetson_fno_package/manifests/fno_jetson_manifest.csv"

python - <<'PY'
import csv
import subprocess
from pathlib import Path

manifest = Path("jetson_fno_package/manifests/fno_jetson_manifest.csv")

with open(manifest, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ckpt = row["checkpoint_path"]
        out = row["torchscript_path"]

        cmd = [
            "python", "-m", "src.eval.export_torchscript",
            "--checkpoint", ckpt,
            "--output", out,
            "--device", "cpu",
        ]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)
PY