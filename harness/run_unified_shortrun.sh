#!/usr/bin/env bash
# Unified cross-family short-run sweep (FP32, TorchScript for FNO/DeepONet).
#
# Brings FNO and DeepONet to R=3 repetitions so that all four families share the
# same repetition convention, and re-measures Sp2GNO under the unified protocol
# (request staged outside the timed window). WNO already conforms and its 3-rep
# short-run sweep from run_wno_shortrun_fp32.sh is reused unchanged.
#
# Scope: the 14 configurations per family that feed Fig. 2, Table 5 and Table 11.
set -u
PY_FD="${PY_FD:-/home/jetson/miniforge3/envs/vs_wno/bin/python}"   # FNO / DeepONet env (torch+CUDA)
export PY_FD
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1

REPS="${REPS:-3}"
STAMP="${STAMP:-unified_shortrun}"
LOG="/home/jetson/jjyoo3/${STAMP}.log"
: > "${LOG}"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

FNO_CFGS="burgers_fno_small burgers_fno_base burgers_fno_large burgers_fno_base_r512 burgers_fno_base_r1024 burgers_fno_base_r4096 burgers_fno_base_r8192 darcy_fno_small darcy_fno_base darcy_fno_large darcy_fno_base_r85 darcy_fno_base_r211 darcy_fno_base_r281 darcy_fno_base_r421"
DON_CFGS="burgers_deeponet_small burgers_deeponet_base burgers_deeponet_large burgers_deeponet_base_r512 burgers_deeponet_base_r1024 burgers_deeponet_base_r4096 burgers_deeponet_base_r8192 darcy_deeponet_small darcy_deeponet_base darcy_deeponet_large darcy_deeponet_base_r85 darcy_deeponet_base_r211 darcy_deeponet_base_r281 darcy_deeponet_base_r421"

# ---------------------------------------------------------------- FNO (R=3)
say "=== FNO short-run, R=${REPS} ==="
cd /home/jetson/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
for rep in $(seq 1 "${REPS}"); do
  "${PY_FD}" - "$rep" "$FNO_CFGS" <<'PY' >> "${LOG}" 2>&1
import csv, os, subprocess, sys
from pathlib import Path
rep, wanted = sys.argv[1], set(sys.argv[2].split())
out = Path("results/jetson_fno_unified"); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open("manifests/fno_jetson_manifest.csv")))
for r in rows:
    if r["experiment_name"] not in wanted:
        continue
    tag = f'{r["experiment_name"]}_seed{r["selected_seed"]}_torchscript_fp32_rep{rep}'
    if (out / f"{tag}.json").exists():
        print(f"[SKIP] {tag}"); continue
    cmd = [os.environ.get("PY_FD", "python3"), "-m", "src.eval.benchmark_inference",
           "--mode", "torchscript", "--checkpoint", r["checkpoint_path"],
           "--torchscript", r["torchscript_path"], "--input-bank", r["input_bank_path"],
           "--precision", "fp32", "--batch-size", "1",
           "--num-warmup", "30", "--num-iters", "100",
           "--results-dir", str(out), "--result-tag", tag]
    print(f"[RUN] {tag}", flush=True)
    subprocess.run(cmd, check=False)
PY
done

# ----------------------------------------------------------- DeepONet (R=3)
say "=== DeepONet short-run, R=${REPS} ==="
cd /home/jetson/jjyoo3/EDCNO_DeepONet || exit 1
export PYTHONPATH=$PWD
for rep in $(seq 1 "${REPS}"); do
  "${PY_FD}" - "$rep" "$DON_CFGS" <<'PY' >> "${LOG}" 2>&1
import csv, os, subprocess, sys
from pathlib import Path
rep, wanted = sys.argv[1], set(sys.argv[2].split())
out = Path("results/jetson_deeponet_unified"); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open("manifests/deeponet_jetson_manifest.csv")))
for r in rows:
    if r["experiment_name"] not in wanted:
        continue
    tag = f'{r["experiment_name"]}_torchscript_fp32_rep{rep}'
    if (out / f"{tag}.json").exists():
        print(f"[SKIP] {tag}"); continue
    cmd = [os.environ.get("PY_FD", "python3"), "-m", "src.eval.benchmark_inference",
           "--mode", "torchscript", "--checkpoint", r["checkpoint_path"],
           "--torchscript", r["torchscript_path"], "--input-bank", r["input_bank_path"],
           "--precision", "fp32", "--batch-size", "1",
           "--num-warmup", "30", "--num-iters", "100",
           "--results-dir", str(out), "--result-tag", tag]
    print(f"[RUN] {tag}", flush=True)
    subprocess.run(cmd, check=False)
PY
done

# ------------------------------------------- Sp2GNO (unified prestaged, R=3)
say "=== Sp2GNO short-run, unified prestaged protocol, R=${REPS} ==="
cd /home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026 || exit 1
PYBIN=/home/jetson/miniforge3/envs/extra_bench/bin/python
NEW=/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/checkpoints
SUITE=inference_runs/sp2gno_unified_shortrun

run_sp () {  # case dataset ckpt width sub res k
  for rep in $(seq 1 "${REPS}"); do
    local rn="$1_fp32_strict_rep${rep}"
    [ -f "${SUITE}/${rn}/reports/sp2gno_edge_summary_${rn}.csv" ] && { say "[SKIP] ${rn}"; continue; }
    local a=(--case_id "$1" --run_name "${rn}" --suite_root "${SUITE}" --dataset "$2"
             --data_dir /home/jetson/data --cache_dir cache --ckpt "${NEW}/$3"
             --width "$4" --n_layers 6 --num_freq 64 --k "$7" --precision fp32_strict
             --unified_protocol 1 --timing_class short_run --num_iters 100
             --warmup_seconds 20 --validity_samples 8 --sample_index 0 --rep "${rep}")
    if [ "$2" = "burgers" ]; then a+=(--sub "$5" --burgers_split Jetson_data/burgers_split.json)
    else a+=(--res "$6" --ntrain 900 --nval 100 --ntest 200); fi
    say "[RUN] ${rn}"
    PYTHONNOUSERSITE=1 "${PYBIN}" bench_sp2gno_jetson_exact.py "${a[@]}" >> "${LOG}" 2>&1 \
      && say "[ OK ] ${rn}" || say "[FAIL] ${rn}"
  done
}
run_sp sp2gno_burgers_small_s2048 burgers sp2gno_burgers_small_s2048.pth 13 4  0 8
run_sp sp2gno_burgers_base_s2048  burgers sp2gno_burgers_base_s2048.pth  24 4  0 8
run_sp sp2gno_burgers_large_s2048 burgers sp2gno_burgers_large_s2048.pth 45 4  0 8
run_sp sp2gno_burgers_base_r512   burgers sp2gno_burgers_base_r512.pth   24 16 0 8
run_sp sp2gno_burgers_base_r1024  burgers sp2gno_burgers_base_r1024.pth  24 8  0 8
run_sp sp2gno_burgers_base_s4096  burgers sp2gno_burgers_base_s4096.pth  24 2  0 8
run_sp sp2gno_burgers_base_r8192  burgers sp2gno_burgers_base_r8192.pth  24 1  0 8
run_sp sp2gno_darcy_small_r141 darcy sp2gno_darcy_small_r141.pth 13 0 141 20
run_sp sp2gno_darcy_base_r141  darcy sp2gno_darcy_base_r141.pth  24 0 141 20
run_sp sp2gno_darcy_large_r141 darcy sp2gno_darcy_large_r141.pth 45 0 141 20
run_sp sp2gno_darcy_base_r85   darcy sp2gno_darcy_base_r85.pth   24 0  85 20
run_sp sp2gno_darcy_base_r211  darcy sp2gno_darcy_base_r211.pth  24 0 211 20
run_sp sp2gno_darcy_base_r281  darcy sp2gno_darcy_base_r281.pth  24 0 281 20
run_sp sp2gno_darcy_base_r421  darcy sp2gno_darcy_base_r421.pth  24 0 421 20

# ------------------------------------------------ WNO (unified, R=3)
say "=== WNO short-run, R=${REPS} ==="
cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks || exit 1
BKW=/home/jetson/data/wno_inference_banks_exact
run_wno_uni () {  # case ds ckpt bank
  for rep in $(seq 1 "${REPS}"); do
    local t="$1_rep${rep}"
    [ -f "results/jetson_wno_exact/wno_unified_shortrun/${t}/fp32_strict/result.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    PYTHONNOUSERSITE=1 "${PYBIN}" bench_wno_jetson_exact.py \
      --case-id "${t}" --dataset "$2" --checkpoint "$3" --bank "$4" \
      --precision-mode fp32_strict --sample-index 0 --batch-size 1 \
      --timing-class short_run --num-warmup 30 --num-iters 100 --device cuda \
      --results-root results/jetson_wno_exact --run-tag wno_unified_shortrun \
      --compute-full-eval 0 --compute-perturbation 0 >> "${LOG}" 2>&1 \
      && say "[ OK ] $t" || say "[FAIL] $t"
  done
}
run_wno_uni wno_burgers_small_r2048 burgers checkpoints/wno_burgers_small_r2048.pth $BKW/burgers_r2048_bank.pt
run_wno_uni wno_burgers_base_r2048  burgers checkpoints/wno_burgers_base_r2048.pth  $BKW/burgers_r2048_bank.pt
run_wno_uni wno_burgers_large_r2048 burgers checkpoints/wno_burgers_large_r2048.pth $BKW/burgers_r2048_bank.pt
run_wno_uni wno_burgers_base_r512   burgers checkpoints/wno_burgers_base_r512.pth   $BKW/burgers_r512_bank.pt
run_wno_uni wno_burgers_base_r1024  burgers checkpoints/wno_burgers_base_r1024.pth  $BKW/burgers_r1024_bank.pt
run_wno_uni wno_burgers_base_r4096  burgers checkpoints/wno_burgers_base_r4096.pth  $BKW/burgers_r4096_bank.pt
run_wno_uni wno_burgers_base_r8192  burgers checkpoints/wno_burgers_base_r8192.pth  $BKW/burgers_r8192_bank.pt
run_wno_uni wno_darcy_small_r141 darcy checkpoints/wno_darcy_small_r141.pth $BKW/darcy_r141_bank.pt
run_wno_uni wno_darcy_base_r141  darcy checkpoints/wno_darcy_base_r141.pth  $BKW/darcy_r141_bank.pt
run_wno_uni wno_darcy_large_r141 darcy checkpoints/wno_darcy_large_r141.pth $BKW/darcy_r141_bank.pt
run_wno_uni wno_darcy_base_r85   darcy checkpoints/wno_darcy_base_r85.pth   $BKW/darcy_r85_bank.pt
run_wno_uni wno_darcy_base_r211  darcy checkpoints/wno_darcy_base_r211.pth  $BKW/darcy_r211_bank.pt
run_wno_uni wno_darcy_base_r281  darcy checkpoints/wno_darcy_base_r281.pth  $BKW/darcy_r281_bank.pt
run_wno_uni wno_darcy_base_r421  darcy checkpoints/wno_darcy_base_r421.pth  $BKW/darcy_r421_bank.pt

say "=== UNIFIED SHORT-RUN DONE ==="
