from __future__ import annotations

from pathlib import Path
import json
import torch

from src.eval.common import load_model_and_normalizers


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.norm((a - b).reshape(a.shape[0], -1), dim=1)
    den = torch.norm(b.reshape(b.shape[0], -1), dim=1).clamp_min(1e-12)
    return float((num / den).mean().item())


def run_case(name: str, ckpt_path: str, ts_path: str, bank_path: str, out_dir: Path):
    payload = torch.load(bank_path, map_location="cpu", weights_only=False)
    x = payload["x"].float()

    _, _, _, _, _, eager = load_model_and_normalizers(ckpt_path, map_location="cpu")
    eager.eval()

    ts = torch.jit.load(ts_path, map_location="cpu").eval()

    with torch.no_grad():
        y_eager = eager(x)
        y_ts = ts(x)

    out = {
        "name": name,
        "checkpoint": ckpt_path,
        "torchscript": ts_path,
        "input_bank": bank_path,
        "num_samples": int(x.shape[0]),
        "mean_abs_diff": float((y_eager - y_ts).abs().mean().item()),
        "max_abs_diff": float((y_eager - y_ts).abs().max().item()),
        "relative_l2_diff": relative_l2(y_eager, y_ts),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def main():
    out_dir = Path("results/jetson_fno_consistency")

    run_case(
        name="burgers_base_consistency",
        ckpt_path="artifacts/checkpoints/burgers_fno_base_seed3_best.pt",
        ts_path="artifacts/torchscript/burgers_fno_base_seed3.ts",
        bank_path="artifacts/benchmark_inputs/burgers_r2048_bank.pt",
        out_dir=out_dir,
    )

    run_case(
        name="darcy_base_consistency",
        ckpt_path="artifacts/checkpoints/darcy_fno_base_seed0_best.pt",
        ts_path="artifacts/torchscript/darcy_fno_base_seed0.ts",
        bank_path="artifacts/benchmark_inputs/darcy_r141_bank.pt",
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()