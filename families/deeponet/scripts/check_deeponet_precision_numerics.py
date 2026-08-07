#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.eval.common import load_model_and_normalizers


PRECISIONS = ["tf32", "bf16_autocast", "fp16_autocast", "fp16_native"]
BACKENDS = ["eager", "torchscript"]


def load_bank(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ["x", "inputs", "input", "data"]:
            if k in obj:
                return obj[k].float()
    raise RuntimeError(f"Unsupported input bank format: {path}")


def set_precision_flags(precision: str) -> None:
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"


def run_forward(model, x: torch.Tensor, precision: str) -> torch.Tensor:
    set_precision_flags(precision)

    if precision == "bf16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x).float()

    if precision == "fp16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x).float()

    if precision == "fp16_native":
        return model.half()(x.half()).float()

    return model(x).float()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(a.shape[0], -1).float()
    b = b.reshape(b.shape[0], -1).float()
    num = (a - b).norm(dim=1)
    den = b.norm(dim=1) + 1e-12
    return float((num / den).mean().item())


def main() -> None:
    manifest = Path("manifests/deeponet_jetson_manifest.csv")
    out = Path("results/artifacts/deeponet_precision_numerics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(manifest)

    rows_out = []

    with manifest.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        exp = r["experiment_name"]
        bank_path = r["input_bank_path"]

        print(f"=== {exp} ===", flush=True)

        x_cpu = load_bank(bank_path)
        x = x_cpu.cuda(non_blocking=False)

        for backend in BACKENDS:
            print(f"  backend={backend}", flush=True)

            if backend == "eager":
                _, _, _, _, _, model = load_model_and_normalizers(
                    r["checkpoint_path"],
                    map_location="cpu",
                )
                model.eval().cuda()
            else:
                model = torch.jit.load(r["torchscript_path"], map_location="cuda")
                model.eval()

            with torch.no_grad():
                try:
                    y_ref = run_forward(model, x, "fp32_strict")
                    torch.cuda.synchronize()
                except Exception as e:
                    for prec in PRECISIONS:
                        rows_out.append({
                            "experiment_name": exp,
                            "backend": backend,
                            "precision": prec,
                            "status": "failure",
                            "failure_stage": "fp32_reference",
                            "rel_l2_vs_fp32": "",
                            "max_abs_diff_vs_fp32": "",
                            "mean_abs_diff_vs_fp32": "",
                            "error": repr(e),
                        })
                    del model
                    torch.cuda.empty_cache()
                    continue

                for prec in PRECISIONS:
                    try:
                        y = run_forward(model, x, prec)
                        torch.cuda.synchronize()

                        rows_out.append({
                            "experiment_name": exp,
                            "backend": backend,
                            "precision": prec,
                            "status": "success",
                            "failure_stage": "",
                            "rel_l2_vs_fp32": rel_l2(y, y_ref),
                            "max_abs_diff_vs_fp32": float((y - y_ref).abs().max().item()),
                            "mean_abs_diff_vs_fp32": float((y - y_ref).abs().mean().item()),
                            "error": "",
                        })

                    except Exception as e:
                        rows_out.append({
                            "experiment_name": exp,
                            "backend": backend,
                            "precision": prec,
                            "status": "failure",
                            "failure_stage": "reduced_precision",
                            "rel_l2_vs_fp32": "",
                            "max_abs_diff_vs_fp32": "",
                            "mean_abs_diff_vs_fp32": "",
                            "error": repr(e),
                        })

            del model
            torch.cuda.empty_cache()

    fieldnames = [
        "experiment_name",
        "backend",
        "precision",
        "status",
        "failure_stage",
        "rel_l2_vs_fp32",
        "max_abs_diff_vs_fp32",
        "mean_abs_diff_vs_fp32",
        "error",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {out} rows={len(rows_out)}")


if __name__ == "__main__":
    main()
