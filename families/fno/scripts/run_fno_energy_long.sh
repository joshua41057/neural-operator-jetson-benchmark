#!/usr/bin/env bash
set -u
set -o pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD

RESULTS_DIR="results/jetson_fno_energy_long"
mkdir -p "${RESULTS_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${RESULTS_DIR}/run_fno_energy_long_${RUN_TS}.log"
SUCCESS_LOG="${RESULTS_DIR}/run_fno_energy_long_${RUN_TS}_success.txt"
FAIL_LOG="${RESULTS_DIR}/run_fno_energy_long_${RUN_TS}_fail.txt"

touch "${MASTER_LOG}" "${SUCCESS_LOG}" "${FAIL_LOG}"

# For paper-quality energy, keep runs long enough relative to tegrastats sampling.
WARMUP_SECONDS="${WARMUP_SECONDS:-20}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
TEGRATS_INTERVAL_MS="${TEGRATS_INTERVAL_MS:-100}"
OVERWRITE="${OVERWRITE:-0}"

python - <<PY 2>&1 | tee -a "${MASTER_LOG}"
import subprocess
import time
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")
success_log = Path("${SUCCESS_LOG}")
fail_log = Path("${FAIL_LOG}")

warmup_seconds = float("${WARMUP_SECONDS}")
measure_seconds = float("${MEASURE_SECONDS}")
tegrastats_interval_ms = str("${TEGRATS_INTERVAL_MS}")
overwrite = int("${OVERWRITE}")

cases = [
    # Lightweight 1D reference
    dict(
        tag="burgers_base_r2048_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/burgers_fno_base_seed3_best.pt",
        ts="artifacts/torchscript/burgers_fno_base_seed3.ts",
        bank="artifacts/benchmark_inputs/burgers_r2048_bank.pt",
    ),
    dict(
        tag="burgers_base_r2048_ts_tf32_energy",
        mode="torchscript",
        precision="tf32",
        ckpt="artifacts/checkpoints/burgers_fno_base_seed3_best.pt",
        ts="artifacts/torchscript/burgers_fno_base_seed3.ts",
        bank="artifacts/benchmark_inputs/burgers_r2048_bank.pt",
    ),

    # Burgers resolution scaling, FP32 strict
    dict(
        tag="burgers_base_r8192_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/burgers_fno_base_r8192_seed1_best.pt",
        ts="artifacts/torchscript/burgers_fno_base_r8192_seed1.ts",
        bank="artifacts/benchmark_inputs/burgers_r8192_bank.pt",
    ),

    # Darcy baseline
    dict(
        tag="darcy_base_r141_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_base_seed0_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_seed0.ts",
        bank="artifacts/benchmark_inputs/darcy_r141_bank.pt",
    ),
    dict(
        tag="darcy_base_r141_ts_tf32_energy",
        mode="torchscript",
        precision="tf32",
        ckpt="artifacts/checkpoints/darcy_fno_base_seed0_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_seed0.ts",
        bank="artifacts/benchmark_inputs/darcy_r141_bank.pt",
    ),

    # Darcy resolution scaling, FP32 strict
    dict(
        tag="darcy_base_r85_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_base_r85_seed2_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_r85_seed2.ts",
        bank="artifacts/benchmark_inputs/darcy_r85_bank.pt",
    ),
    dict(
        tag="darcy_base_r211_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_base_r211_seed1_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_r211_seed1.ts",
        bank="artifacts/benchmark_inputs/darcy_r211_bank.pt",
    ),
    dict(
        tag="darcy_base_r281_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_r281_seed1.ts",
        bank="artifacts/benchmark_inputs/darcy_r281_bank.pt",
    ),
    dict(
        tag="darcy_base_r421_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_base_r421_seed1_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_r421_seed1.ts",
        bank="artifacts/benchmark_inputs/darcy_r421_bank.pt",
    ),

    # Darcy model scaling, FP32 strict
    dict(
        tag="darcy_small_r141_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_small_seed4_best.pt",
        ts="artifacts/torchscript/darcy_fno_small_seed4.ts",
        bank="artifacts/benchmark_inputs/darcy_r141_bank.pt",
    ),
    dict(
        tag="darcy_large_r141_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_large_seed0_best.pt",
        ts="artifacts/torchscript/darcy_fno_large_seed0.ts",
        bank="artifacts/benchmark_inputs/darcy_r141_bank.pt",
    ),

    # Native/frontier resolution
    dict(
        tag="darcy_base_r281_on421_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_r281_seed1.ts",
        bank="artifacts/benchmark_inputs/darcy_r421_bank.pt",
    ),
    dict(
        tag="darcy_base_r281_on421_ts_tf32_energy",
        mode="torchscript",
        precision="tf32",
        ckpt="artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt",
        ts="artifacts/torchscript/darcy_fno_base_r281_seed1.ts",
        bank="artifacts/benchmark_inputs/darcy_r421_bank.pt",
    ),
    dict(
        tag="darcy_large_r141_on421_ts_fp32_strict_energy",
        mode="torchscript",
        precision="fp32_strict",
        ckpt="artifacts/checkpoints/darcy_fno_large_seed0_best.pt",
        ts="artifacts/torchscript/darcy_fno_large_seed0.ts",
        bank="artifacts/benchmark_inputs/darcy_r421_bank.pt",
    ),
    dict(
        tag="darcy_large_r141_on421_ts_tf32_energy",
        mode="torchscript",
        precision="tf32",
        ckpt="artifacts/checkpoints/darcy_fno_large_seed0_best.pt",
        ts="artifacts/torchscript/darcy_fno_large_seed0.ts",
        bank="artifacts/benchmark_inputs/darcy_r421_bank.pt",
    ),
]

print(f"Energy cases: {len(cases)}")
print(f"warmup_seconds={warmup_seconds}, measure_seconds={measure_seconds}, tegrastats_interval_ms={tegrastats_interval_ms}, overwrite={overwrite}")

for c in cases:
    tag = c["tag"]
    out_json = results_dir / f"{tag}.json"
    tegra_log = results_dir / f"{tag}_tegrastats.log"

    if out_json.exists() and not overwrite:
        print(f"[SKIP] {tag}")
        continue

    print("=" * 100)
    print("[RUN]", tag)
    print("=" * 100)

    tegra_proc = subprocess.Popen([
        "tegrastats",
        "--interval", tegrastats_interval_ms,
        "--logfile", str(tegra_log),
    ])

    status = 0
    try:
        cmd = [
            "python", "-m", "src.eval.benchmark_energy_inference",
            "--mode", c["mode"],
            "--checkpoint", c["ckpt"],
            "--torchscript", c["ts"],
            "--input-bank", c["bank"],
            "--sample-index", "0",
            "--batch-size", "1",
            "--precision-mode", c["precision"],
            "--warmup-seconds", str(warmup_seconds),
            "--measure-seconds", str(measure_seconds),
            "--device", "cuda",
            "--results-dir", str(results_dir),
            "--result-tag", tag,
        ]

        print("CMD:", " ".join(cmd))
        proc = subprocess.run(cmd)
        status = proc.returncode

    except Exception as e:
        print(f"[EXC] {tag}: {e}")
        status = 1

    finally:
        tegra_proc.terminate()
        try:
            tegra_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegra_proc.kill()
            tegra_proc.wait()

    if status == 0 and out_json.exists():
        print(f"[ OK ] {tag}")
        with open(success_log, "a", encoding="utf-8") as f:
            f.write(tag + "\\n")
    else:
        print(f"[FAIL] {tag} returncode={status}")
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(tag + "\\n")

    # Cooldown to avoid immediate thermal/run-to-run interference.
    time.sleep(10)

print("Finished long-run energy matrix.")
print(f"Success log: {success_log}")
print(f"Fail log   : {fail_log}")
PY
