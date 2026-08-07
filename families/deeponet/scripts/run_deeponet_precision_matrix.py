#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path


PRECISIONS = ["fp32_strict", "tf32", "bf16_autocast", "fp16_autocast", "fp16_native"]
BACKENDS = ["eager", "torchscript"]


def classify_failure(text: str) -> str:
    t = text.lower()
    if "out of memory" in t or "cuda oom" in t:
        return "cuda_oom"
    if "expected all tensors to be on the same device" in t:
        return "device_mismatch"
    if "not implemented" in t or "not supported" in t:
        return "unsupported_op"
    if "dtype" in t:
        return "dtype_error"
    if "half" in t or "bfloat16" in t or "bf16" in t:
        return "reduced_precision_error"
    return "runtime_error"


def main():
    manifest = Path("manifests/deeponet_jetson_manifest.csv")
    results_dir = Path("results/jetson_deeponet_precision")
    log_dir = Path("logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_deeponet_precision_matrix_{stamp}.log"

    rows = list(csv.DictReader(manifest.open()))
    with log_path.open("w", encoding="utf-8") as log:
        for r in rows:
            exp = r["experiment_name"]
            ckpt = r["checkpoint_path"]
            ts = r["torchscript_path"]
            bank = r["input_bank_path"]

            for backend in BACKENDS:
                for prec in PRECISIONS:
                    tag = f"{exp}_{backend}_{prec}"
                    out_json = results_dir / f"{tag}.json"

                    cmd = [
                        "python", "-m", "src.eval.benchmark_inference",
                        "--mode", backend,
                        "--checkpoint", ckpt,
                        "--input-bank", bank,
                        "--precision", prec,
                        "--batch-size", "1",
                        "--num-warmup", "20",
                        "--num-iters", "50",
                        "--device", "cuda",
                        "--results-dir", str(results_dir),
                        "--result-tag", tag,
                    ]
                    if backend == "torchscript":
                        cmd += ["--torchscript", ts]

                    header = f"\n=== RUN {tag} ===\n"
                    print(header, end="")
                    log.write(header)
                    log.flush()

                    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    print(proc.stdout)
                    log.write(proc.stdout)
                    log.flush()

                    if proc.returncode != 0:
                        fail = {
                            "result_tag": tag,
                            "status": "failure",
                            "failure_class": classify_failure(proc.stdout),
                            "returncode": proc.returncode,
                            "experiment_name": exp,
                            "backend": backend,
                            "precision": prec,
                            "checkpoint": ckpt,
                            "torchscript": ts if backend == "torchscript" else None,
                            "input_bank": bank,
                            "stdout_tail": proc.stdout[-4000:],
                        }
                        out_json.write_text(json.dumps(fail, indent=2, sort_keys=True))
                        print(f"Wrote failure JSON {out_json}")

    print(f"Log: {log_path}")
    print("Precision JSON count:", len(list(results_dir.glob("*.json"))))


if __name__ == "__main__":
    main()
