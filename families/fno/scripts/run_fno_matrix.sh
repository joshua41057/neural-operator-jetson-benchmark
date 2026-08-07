#!/usr/bin/env bash
set -u
set -o pipefail

cd ~/jjyoo3/EDCNO || exit 1
mkdir -p results/jetson_fno
export PYTHONPATH=$PWD

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="results/jetson_fno/run_fno_matrix_${RUN_TS}.log"
SUCCESS_LOG="results/jetson_fno/run_fno_matrix_${RUN_TS}_success.txt"
FAIL_LOG="results/jetson_fno/run_fno_matrix_${RUN_TS}_fail.txt"

touch "${MASTER_LOG}" "${SUCCESS_LOG}" "${FAIL_LOG}"

python - <<'PY' 2>&1 | tee -a "${MASTER_LOG}"
import csv
import subprocess
import time
from pathlib import Path

manifest = Path("manifests/fno_jetson_manifest.csv")
results_dir = Path("results/jetson_fno")
results_dir.mkdir(parents=True, exist_ok=True)

success_log = Path("results/jetson_fno").glob("run_fno_matrix_*_success.txt")
fail_log = Path("results/jetson_fno").glob("run_fno_matrix_*_fail.txt")

# resolve latest-created logs from shell-created files
success_log = sorted(Path("results/jetson_fno").glob("run_fno_matrix_*_success.txt"))[-1]
fail_log = sorted(Path("results/jetson_fno").glob("run_fno_matrix_*_fail.txt"))[-1]

with open(manifest, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded manifest with {len(rows)} experiment rows")

for row in rows:
    exp = row["experiment_name"]
    seed = row["selected_seed"]
    ckpt = row["checkpoint_path"]
    ts = row["torchscript_path"]
    bank = row["input_bank_path"]

    runs = [
        ("eager", "fp32"),
        ("eager", "fp16"),
        ("torchscript", "fp32"),
        ("torchscript", "fp16"),
    ]

    for mode, precision in runs:
        tag = f"{exp}_seed{seed}_{mode}_{precision}"
        out_json = results_dir / f"{tag}.json"
        tegra_log = results_dir / f"{tag}_tegrastats.log"

        if out_json.exists():
            print(f"[SKIP] {tag} already exists")
            continue

        print("=" * 80)
        print(f"[RUN ] {tag}")
        print(f"       checkpoint : {ckpt}")
        print(f"       torchscript: {ts if mode == 'torchscript' else 'N/A'}")
        print(f"       input_bank : {bank}")
        print("=" * 80)

        tegra_proc = subprocess.Popen([
            "tegrastats",
            "--interval", "100",
            "--logfile", str(tegra_log),
        ])

        status = 0
        try:
            cmd = [
                "python", "-m", "src.eval.benchmark_inference",
                "--mode", mode,
                "--checkpoint", ckpt,
                "--input-bank", bank,
                "--precision", precision,
                "--batch-size", "1",
                "--num-warmup", "30",
                "--num-iters", "100",
                "--device", "cuda",
                "--results-dir", str(results_dir),
                "--result-tag", tag,
            ]

            if mode == "torchscript":
                cmd += ["--torchscript", ts]

            print("CMD :", " ".join(cmd))
            proc = subprocess.run(cmd)
            status = proc.returncode

        except Exception as e:
            print(f"[EXC ] {tag}: {e}")
            status = 1

        finally:
            tegra_proc.terminate()
            tegra_proc.wait()

        if status == 0 and out_json.exists():
            print(f"[ OK ] {tag}")
            with open(success_log, "a", encoding="utf-8") as f:
                f.write(f"{tag}\n")
        else:
            print(f"[FAIL] {tag}")
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{tag}\n")

        # short cooldown to reduce immediate run-to-run interference
        time.sleep(2)

print("Finished full pass over manifest.")
print(f"Success log: {success_log}")
print(f"Fail log   : {fail_log}")
PY