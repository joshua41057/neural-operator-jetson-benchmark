#!/usr/bin/env bash
set -u
set -o pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

RESULTS_DIR="results/jetson_fno_precision"
mkdir -p "${RESULTS_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${RESULTS_DIR}/run_fno_precision_matrix_${RUN_TS}.log"
SUCCESS_LOG="${RESULTS_DIR}/run_fno_precision_matrix_${RUN_TS}_success.txt"
FAIL_LOG="${RESULTS_DIR}/run_fno_precision_matrix_${RUN_TS}_fail.txt"

touch "${MASTER_LOG}" "${SUCCESS_LOG}" "${FAIL_LOG}"

# Timing parameters for precision feasibility/latency.
# Energy will be measured separately with longer runs.
NUM_WARMUP="${NUM_WARMUP:-30}"
NUM_ITERS="${NUM_ITERS:-100}"
REPEATS="${REPEATS:-3}"

# Set OVERWRITE=1 to rerun existing jsons.
OVERWRITE="${OVERWRITE:-0}"

python - <<PY 2>&1 | tee -a "${MASTER_LOG}"
import csv
import subprocess
import time
from pathlib import Path

manifest = Path("manifests/fno_jetson_manifest.csv")
results_dir = Path("${RESULTS_DIR}")
success_log = Path("${SUCCESS_LOG}")
fail_log = Path("${FAIL_LOG}")

num_warmup = int("${NUM_WARMUP}")
num_iters = int("${NUM_ITERS}")
repeats = int("${REPEATS}")
overwrite = int("${OVERWRITE}")

precision_modes = [
    "fp32_strict",
    "tf32",
    "bf16_autocast",
    "fp16_autocast",
    "fp16_native",
]

with open(manifest, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} manifest rows")
print(f"Precision modes: {precision_modes}")
print(f"num_warmup={num_warmup}, num_iters={num_iters}, repeats={repeats}, overwrite={overwrite}")

for row in rows:
    exp = row["experiment_name"]
    seed = row["selected_seed"]
    ckpt = row["checkpoint_path"]
    ts = row["torchscript_path"]
    bank = row["input_bank_path"]

    for mode in ["eager", "torchscript"]:
        for precision_mode in precision_modes:
            tag = f"{exp}_seed{seed}_{mode}_{precision_mode}"
            out_json = results_dir / f"{tag}.json"

            if out_json.exists() and not overwrite:
                print(f"[SKIP] {tag}")
                continue

            cmd = [
                "python", "-m", "src.eval.benchmark_precision_inference",
                "--mode", mode,
                "--checkpoint", ckpt,
                "--input-bank", bank,
                "--sample-index", "0",
                "--batch-size", "1",
                "--precision-mode", precision_mode,
                "--num-warmup", str(num_warmup),
                "--num-iters", str(num_iters),
                "--repeats", str(repeats),
                "--device", "cuda",
                "--results-dir", str(results_dir),
                "--result-tag", tag,
            ]

            if mode == "torchscript":
                cmd += ["--torchscript", ts]

            print("=" * 100)
            print("[RUN]", tag)
            print("CMD:", " ".join(cmd))
            print("=" * 100)

            proc = subprocess.run(cmd)
            if proc.returncode == 0:
                print(f"[ OK ] {tag}")
                with open(success_log, "a", encoding="utf-8") as f:
                    f.write(tag + "\\n")
            else:
                print(f"[FAIL] {tag} returncode={proc.returncode}")
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(tag + "\\n")

            time.sleep(1)

print("Finished precision matrix.")
print(f"Success log: {success_log}")
print(f"Fail log   : {fail_log}")
PY
