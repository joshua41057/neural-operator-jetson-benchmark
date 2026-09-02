import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch


def resolve_path(p, root):
    if p is None:
        return None
    p = str(p)
    if os.path.isabs(p):
        return p
    cand = os.path.join(root, p)
    if os.path.exists(cand):
        return cand
    return p


def load_input_bank(path):
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".npy":
        return torch.from_numpy(np.load(path))

    if suffix == ".npz":
        z = np.load(path)
        for k in ["x", "input", "inputs", "a", "coeff", "u0", "input_bank"]:
            if k in z:
                return torch.from_numpy(z[k])
        return torch.from_numpy(z[list(z.keys())[0]])

    if suffix in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")

        if torch.is_tensor(obj):
            return obj

        if isinstance(obj, dict):
            for k in ["x", "input", "inputs", "a", "coeff", "u0", "input_bank"]:
                if k in obj:
                    v = obj[k]
                    return v if torch.is_tensor(v) else torch.as_tensor(v)

            print("Input bank dict keys:", sorted(obj.keys()))
            raise RuntimeError("Could not identify input tensor key in input bank dict.")

        if isinstance(obj, (list, tuple)):
            for v in obj:
                if torch.is_tensor(v):
                    return v
            return torch.as_tensor(obj)

    raise RuntimeError(f"Unsupported input bank format: {path}")


def load_torchscript(path):
    model = torch.jit.load(path, map_location="cuda")
    model.eval()
    return model


@torch.inference_mode()
def run_model(model, x, tf32):
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)

    x = x.cuda(non_blocking=True).float()

    # Preserve benchmark bank semantics. Usually x is already [1, ...].
    if x.ndim >= 1 and x.shape[0] > 1:
        outs = []
        for i in range(x.shape[0]):
            y = model(x[i:i+1])
            if isinstance(y, (tuple, list)):
                y = y[0]
            outs.append(y.detach().float().cpu())
        return torch.cat(outs, dim=0)

    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y.detach().float().cpu()


def rel_l2(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return (torch.linalg.norm(a - b) / torch.linalg.norm(a)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    d = json.load(open(args.summary_json))

    ts_path = resolve_path(d.get("torchscript"), args.root)
    bank_path = resolve_path(d.get("input_bank"), args.root)

    print("torchscript:", ts_path)
    print("input_bank :", bank_path)

    if ts_path is None or not os.path.exists(ts_path):
        raise RuntimeError(f"TorchScript path not found: {ts_path}")
    if bank_path is None or not os.path.exists(bank_path):
        raise RuntimeError(f"Input bank path not found: {bank_path}")

    x = load_input_bank(bank_path)
    model = load_torchscript(ts_path)

    torch.cuda.empty_cache()
    y_fp32 = run_model(model, x, tf32=False)

    torch.cuda.empty_cache()
    y_tf32 = run_model(model, x, tf32=True)

    value = rel_l2(y_fp32, y_tf32)
    max_abs = (y_fp32 - y_tf32).abs().max().item()

    out = {
        "summary_json": args.summary_json,
        "torchscript": ts_path,
        "input_bank": bank_path,
        "input_shape": list(x.shape),
        "output_shape": list(y_fp32.shape),
        "tf32_vs_fp32_relative_l2": value,
        "tf32_vs_fp32_max_abs": max_abs,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
