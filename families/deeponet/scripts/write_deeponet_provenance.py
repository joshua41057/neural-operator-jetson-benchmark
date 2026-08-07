#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"


def main():
    out = Path("results/artifacts/deeponet_jetson_provenance.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "nvidia_smi": sh(["bash", "-lc", "nvidia-smi 2>/dev/null || true"]),
        "tegrastats_path": sh(["bash", "-lc", "command -v tegrastats || true"]),
        "manifest_sha256": sh(["sha256sum", "manifests/deeponet_jetson_manifest.csv"]),
        "checkpoint_count": sh(["bash", "-lc", "find artifacts/checkpoints -name '*deeponet*best.pt' | wc -l"]),
        "torchscript_count": sh(["bash", "-lc", "find artifacts/torchscript -name '*deeponet*.ts' | wc -l"]),
        "summary_count": sh(["bash", "-lc", "find artifacts/summaries -name '*deeponet*summary.json' | wc -l"]),
        "input_banks": sh(["bash", "-lc", "find artifacts/benchmark_inputs -name '*bank.pt' | sort"]),
    }

    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
