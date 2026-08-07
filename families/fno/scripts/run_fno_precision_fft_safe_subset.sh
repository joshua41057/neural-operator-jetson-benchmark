#!/usr/bin/env bash
set -u
set -o pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

RESULTS_DIR="results/jetson_fno_precision_fft_safe"
mkdir -p "${RESULTS_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${RESULTS_DIR}/run_fno_precision_fft_safe_subset_${RUN_TS}.log"
SUCCESS_LOG="${RESULTS_DIR}/run_fno_precision_fft_safe_subset_${RUN_TS}_success.txt"
FAIL_LOG="${RESULTS_DIR}/run_fno_precision_fft_safe_subset_${RUN_TS}_fail.txt"

touch "${MASTER_LOG}" "${SUCCESS_LOG}" "${FAIL_LOG}"

NUM_WARMUP="${NUM_WARMUP:-30}"
NUM_ITERS="${NUM_ITERS:-100}"
REPEATS="${REPEATS:-3}"
OVERWRITE="${OVERWRITE:-0}"

python - <<PY 2>&1 | tee -a "${MASTER_LOG}"
import subprocess
import time
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")
success_log = Path("${SUCCESS_LOG}")
fail_log = Path("${FAIL_LOG}")

num_warmup = int("${NUM_WARMUP}")
num_iters = int("${NUM_ITERS}")
repeats = int("${REPEATS}")
overwrite = int("${OVERWRITE}")

cases = [
    {
        "name": "burgers_base_r2048",
        "ckpt": "artifacts/checkpoints/burgers_fno_base_seed3_best.pt",
        "bank": "artifacts/benchmark_inputs/burgers_r2048_bank.pt",
    },
    {
        "name": "burgers_base_r4096",
        "ckpt": "artifacts/checkpoints/burgers_fno_base_r4096_seed0_best.pt",
        "bank": "artifacts/benchmark_inputs/burgers_r4096_bank.pt",
    },
    {
        "name": "darcy_base_r141",
        "ckpt": "artifacts/checkpoints/darcy_fno_base_seed0_best.pt",
        "bank": "artifacts/benchmark_inputs/darcy_r141_bank.pt",
    },
    {
        "name": "darcy_base_r281",
        "ckpt": "artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt",
        "bank": "artifacts/benchmark_inputs/darcy_r281_bank.pt",
    },
    {
        "name": "darcy_large_r141",
        "ckpt": "artifacts/checkpoints/darcy_fno_large_seed0_best.pt",
        "bank": "artifacts/benchmark_inputs/darcy_r141_bank.pt",
    },
    {
        "name": "darcy_base_r281_on421",
        "ckpt": "artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt",
        "bank": "artifacts/benchmark_inputs/darcy_r421_bank.pt",
    },
    {
        "name": "darcy_large_r141_on421",
        "ckpt": "artifacts/checkpoints/darcy_fno_large_seed0_best.pt",
        "bank": "artifacts/benchmark_inputs/darcy_r421_bank.pt",
    },
]

precision_modes = [
    "fp32_strict",
    "tf32",
    "fft_safe_bf16_autocast",
    "fft_safe_fp16_autocast",
]

print(f"FFT-safe subset cases: {len(cases)}")
print(f"Precision modes: {precision_modes}")
print(f"num_warmup={num_warmup}, num_iters={num_iters}, repeats={repeats}, overwrite={overwrite}")

for case in cases:
    for precision_mode in precision_modes:
        tag = f"{case['name']}_eager_{precision_mode}"
        out_json = results_dir / f"{tag}.json"

        if out_json.exists() and not overwrite:
            print(f"[SKIP] {tag}")
            continue

        cmd = [
            "python", "-m", "src.eval.benchmark_precision_fft_safe",
            "--checkpoint", case["ckpt"],
            "--input-bank", case["bank"],
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

print("Finished FFT-safe precision subset.")
print(f"Success log: {success_log}")
print(f"Fail log   : {fail_log}")
PY