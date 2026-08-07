from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.eval.common import load_model_and_normalizers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt, cfg, model, x_norm, y_norm, wrapper = load_model_and_normalizers(
        args.checkpoint,
        map_location="cpu",
    )
    wrapper.eval().to(args.device)

    scripted = torch.jit.script(wrapper)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"Saved scripted TorchScript model to {output}")


if __name__ == "__main__":
    main()
