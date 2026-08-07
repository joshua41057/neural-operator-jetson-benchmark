import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from sample_codes.utils import UnitGaussianNormalizer


ROOT = Path("/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks")
OUT = Path("/home/jetson/data/wno_inference_banks_exact")
OUT.mkdir(parents=True, exist_ok=True)

BURGERS_MAT = Path("/home/jetson/data/burgers_data_R10.mat")
BURGERS_SPLIT = ROOT / "Jetson_data" / "burgers_split.json"

DARCY_TRAIN_MAT = Path("/home/jetson/data/piececonst_r421_N1024_smooth1.mat")
DARCY_TEST_MAT = Path("/home/jetson/data/piececonst_r421_N1024_smooth2.mat")


def darcy_indices(res):
    return np.round(np.linspace(0, 420, res)).astype(int)


def make_burgers_bank(res):
    sub = 8192 // res
    mat = sio.loadmat(BURGERS_MAT, variable_names=["a", "u"])

    a_all = mat["a"][:, ::sub].astype(np.float32)
    u_all = mat["u"][:, ::sub].astype(np.float32)

    with open(BURGERS_SPLIT) as f:
        split = json.load(f)

    te_i = np.asarray(split["test"], dtype=np.int64)

    x = torch.from_numpy(a_all[te_i])[:, :, None]
    y = torch.from_numpy(u_all[te_i])

    out = OUT / f"burgers_r{res}_bank.pt"
    torch.save(
        {
            "dataset": "burgers",
            "resolution": res,
            "sub": sub,
            "source": str(BURGERS_MAT),
            "split": str(BURGERS_SPLIT),
            "indices": torch.from_numpy(te_i),
            "x": x,
            "y": y,
            "normalization": "none",
        },
        out,
    )
    print("[OK]", out, "x", tuple(x.shape), "y", tuple(y.shape))


def load_darcy(path, n, idx):
    d = sio.loadmat(path, variable_names=["coeff", "sol"])
    coeff = d["coeff"][:n][:, idx][:, :, idx].astype(np.float32)
    sol = d["sol"][:n][:, idx][:, :, idx].astype(np.float32)
    return coeff, sol


def make_darcy_bank(res):
    idx = darcy_indices(res)
    s = len(idx)

    # exact training protocol:
    # smooth1 first 1000 loaded, then first 900 used for train statistics
    x_train_all, y_train_all = load_darcy(DARCY_TRAIN_MAT, 1000, idx)
    x_test, y_test = load_darcy(DARCY_TEST_MAT, 200, idx)

    x_train = torch.from_numpy(x_train_all[:900])
    y_train = torch.from_numpy(y_train_all[:900])

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    x_norm = UnitGaussianNormalizer(x_train)
    y_norm = UnitGaussianNormalizer(y_train)

    x_test_encoded = x_norm.encode(x_test).reshape(x_test.shape[0], s, s, 1)

    out = OUT / f"darcy_r{res}_bank.pt"
    torch.save(
        {
            "dataset": "darcy",
            "resolution": res,
            "source_train": str(DARCY_TRAIN_MAT),
            "source_test": str(DARCY_TEST_MAT),
            "x_field": "coeff",
            "y_field": "sol",
            "indices": torch.arange(x_test.shape[0]),
            "darcy_idx": torch.from_numpy(idx),
            "x": x_test_encoded,
            "y": y_test,
            "y_mean": y_norm.mean,
            "y_std": y_norm.std,
            "eps": y_norm.eps,
            "normalization": "x encoded using UnitGaussianNormalizer on smooth1[:900]; y raw; output decoded with y_norm from smooth1[:900]",
        },
        out,
    )
    print("[OK]", out, "x", tuple(x_test_encoded.shape), "y", tuple(y_test.shape))


for res in [512, 1024, 2048, 4096, 8192]:
    make_burgers_bank(res)

for res in [85, 141, 211, 281, 421]:
    make_darcy_bank(res)

print("[DONE] exact WNO inference banks written to", OUT)
