from __future__ import annotations

from pathlib import Path
import torch

from src.data.mat_reader import read_mat_key
from src.data.preprocessing import resize_1d, resize_2d, to_float_tensor


def save_bank(path: Path, dataset: str, resolution, input_key: str, source_path: str, x: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "resolution": list(resolution) if isinstance(resolution, (list, tuple)) else [int(resolution)],
        "input_key": input_key,
        "source_path": source_path,
        "num_samples": int(x.shape[0]),
        "x": x.cpu(),
    }
    torch.save(payload, path)
    print(f"Saved {path} shape={tuple(x.shape)}")


def build_burgers_bank(src_path: str, resolution: int, sample_indices: list[int], out_path: Path):
    x_np = read_mat_key(src_path, "a")
    x = to_float_tensor(x_np)
    x = resize_1d(x, resolution)
    x = x[sample_indices]
    save_bank(out_path, "burgers", [resolution], "a", src_path, x)


def build_darcy_bank(src_path: str, resolution: int, sample_indices: list[int], out_path: Path):
    x_np = read_mat_key(src_path, "Kcoeff")
    x = to_float_tensor(x_np)
    x = resize_2d(x, [resolution, resolution])
    x = x[sample_indices]
    save_bank(out_path, "darcy", [resolution, resolution], "Kcoeff", src_path, x)


def main():
    root = Path("artifacts/benchmark_inputs")
    root.mkdir(parents=True, exist_ok=True)

    sample_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    burgers_src = "/home/jetson/data/burgers_data_R10.mat"
    darcy_src = "/home/jetson/data/piececonst_r421_N1024_smooth2.mat"

    build_burgers_bank(
        src_path=burgers_src,
        resolution=8192,
        sample_indices=sample_indices,
        out_path=root / "burgers_r8192_bank.pt",
    )

    build_darcy_bank(
        src_path=darcy_src,
        resolution=421,
        sample_indices=sample_indices,
        out_path=root / "darcy_r421_bank.pt",
    )


if __name__ == "__main__":
    main()