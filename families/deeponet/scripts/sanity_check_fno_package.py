from __future__ import annotations

import csv
from pathlib import Path
import torch

manifest = Path("jetson_fno_package/manifests/fno_jetson_manifest.csv")

with open(manifest, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ckpt_path = Path(row["checkpoint_path"])
        ts_path = Path(row["torchscript_path"])

        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state" in ckpt
        assert "config" in ckpt
        assert "x_normalizer" in ckpt
        assert "y_normalizer" in ckpt

        assert ts_path.exists(), f"Missing TorchScript: {ts_path}"
        model = torch.jit.load(str(ts_path), map_location="cpu")
        model.eval()

        print(f"[OK] {row['experiment_name']} seed{row['selected_seed']}")