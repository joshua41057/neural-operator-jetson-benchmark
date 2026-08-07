from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def resize_2d_no_channel(x: torch.Tensor, target_hw: int) -> torch.Tensor:
    # [N, H, W] -> [N, 1, H, W] -> interpolate -> [N, H, W]
    x_nchw = x.unsqueeze(1)
    y = F.interpolate(
        x_nchw,
        size=(target_hw, target_hw),
        mode="bilinear",
        align_corners=False,
    )
    return y.squeeze(1).contiguous()


def resize_2d_channels_last(x: torch.Tensor, target_hw: int) -> torch.Tensor:
    # [N, H, W, C] -> [N, C, H, W] -> interpolate -> [N, H, W, C]
    x_nchw = x.permute(0, 3, 1, 2)
    y = F.interpolate(
        x_nchw,
        size=(target_hw, target_hw),
        mode="bilinear",
        align_corners=False,
    )
    return y.permute(0, 2, 3, 1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-bank", type=str, required=True)
    ap.add_argument("--dst-bank", type=str, required=True)
    ap.add_argument("--target-res", type=int, required=True)
    args = ap.parse_args()

    src = Path(args.src_bank)
    dst = Path(args.dst_bank)

    payload = torch.load(src, map_location="cpu", weights_only=False)
    x = payload["x"]

    if x.ndim == 3:
        # Darcy bank here is [N, H, W]
        y = resize_2d_no_channel(x, args.target_res)
    elif x.ndim == 4:
        # Support [N, H, W, C] as well
        y = resize_2d_channels_last(x, args.target_res)
    else:
        raise ValueError(f"Unsupported tensor shape for frontier synthesis: {tuple(x.shape)}")

    out = dict(payload)
    out["x"] = y
    out["synthetic_frontier"] = True
    out["synthetic_from"] = str(src)
    out["resolution"] = [args.target_res, args.target_res]

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"Saved {dst} with x.shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()