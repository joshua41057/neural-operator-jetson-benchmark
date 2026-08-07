#!/usr/bin/env bash
# Queued after Option F: the remaining short-run sweeps that Tables 17, 18, 23 and 26
# depend on, all under the unified protocol of Section 4.3.
#
#   Q1  FNO eager        14 configs x R=3   -> Table 17 eager rows,  §B.2 backend text
#   Q2  DeepONet eager   14 configs x R=3   -> Table 26 eager rows,  §C.2 backend text
#   Q3  FNO ablations    11 configs x R=3   -> Table 18,             §B.2 ablation claim
#   Q4  FNO frontier     10 cells  x R=3    -> Table 23,             Table 11 FNO row
#
# The FNO short-run harness now records peak CUDA allocation over the timed window,
# so the memory columns of Tables 17/18/23 are populated by these runs.
set -u
PY_FD="${PY_FD:-/home/jetson/miniforge3/envs/vs_wno/bin/python}"
export PY_FD
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1

REPS="${REPS:-3}"
LOG=/home/jetson/jjyoo3/queue_after_F.log; : > "$LOG"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "queue start | reps=${REPS}"

CFG14_FNO="burgers_fno_small burgers_fno_base burgers_fno_large burgers_fno_base_r512 burgers_fno_base_r1024 burgers_fno_base_r4096 burgers_fno_base_r8192 darcy_fno_small darcy_fno_base darcy_fno_large darcy_fno_base_r85 darcy_fno_base_r211 darcy_fno_base_r281 darcy_fno_base_r421"
CFG14_DON="burgers_deeponet_small burgers_deeponet_base burgers_deeponet_large burgers_deeponet_base_r512 burgers_deeponet_base_r1024 burgers_deeponet_base_r4096 burgers_deeponet_base_r8192 darcy_deeponet_small darcy_deeponet_base darcy_deeponet_large darcy_deeponet_base_r85 darcy_deeponet_base_r211 darcy_deeponet_base_r281 darcy_deeponet_base_r421"
CFG11_ABL="burgers_fno_base_modes12 burgers_fno_base_modes16 burgers_fno_base_modes32 burgers_fno_base_nocoords burgers_fno_base_pad0 burgers_fno_base_pad40 darcy_fno_base_modes12 darcy_fno_base_modes24 darcy_fno_base_nocoords darcy_fno_base_pad0 darcy_fno_base_pad15"

# ------------------------------------------------- Q1: FNO eager
say "===== Q1: FNO eager short-run ====="
cd /home/jetson/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
for rep in $(seq 1 "$REPS"); do
  "$PY_FD" - "$rep" "$CFG14_FNO" eager results/jetson_fno_unified <<'PY' >> "$LOG" 2>&1
import csv, os, subprocess, sys
rep, wanted, mode, out = sys.argv[1], set(sys.argv[2].split()), sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
for r in csv.DictReader(open("manifests/fno_jetson_manifest.csv")):
    if r["experiment_name"] not in wanted: continue
    tag = f'{r["experiment_name"]}_seed{r["selected_seed"]}_{mode}_fp32_rep{rep}'
    if os.path.exists(f"{out}/{tag}.json"): print(f"[SKIP] {tag}"); continue
    cmd = [os.environ["PY_FD"], "-m", "src.eval.benchmark_inference", "--mode", mode,
           "--checkpoint", r["checkpoint_path"], "--input-bank", r["input_bank_path"],
           "--precision", "fp32", "--batch-size", "1", "--num-warmup", "30",
           "--num-iters", "100", "--results-dir", out, "--result-tag", tag]
    if mode == "torchscript": cmd += ["--torchscript", r["torchscript_path"]]
    print(f"[RUN] {tag}", flush=True); subprocess.run(cmd, check=False)
PY
done
say "Q1 done"

# ------------------------------------------------- Q2: DeepONet eager
say "===== Q2: DeepONet eager short-run ====="
cd /home/jetson/jjyoo3/EDCNO_DeepONet || exit 1
export PYTHONPATH=$PWD
for rep in $(seq 1 "$REPS"); do
  "$PY_FD" - "$rep" "$CFG14_DON" eager results/jetson_deeponet_unified <<'PY' >> "$LOG" 2>&1
import csv, os, subprocess, sys
rep, wanted, mode, out = sys.argv[1], set(sys.argv[2].split()), sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
for r in csv.DictReader(open("manifests/deeponet_jetson_manifest.csv")):
    if r["experiment_name"] not in wanted: continue
    tag = f'{r["experiment_name"]}_{mode}_fp32_rep{rep}'
    if os.path.exists(f"{out}/{tag}.json"): print(f"[SKIP] {tag}"); continue
    cmd = [os.environ["PY_FD"], "-m", "src.eval.benchmark_inference", "--mode", mode,
           "--checkpoint", r["checkpoint_path"], "--input-bank", r["input_bank_path"],
           "--precision", "fp32", "--batch-size", "1", "--num-warmup", "30",
           "--num-iters", "100", "--results-dir", out, "--result-tag", tag]
    if mode == "torchscript": cmd += ["--torchscript", r["torchscript_path"]]
    print(f"[RUN] {tag}", flush=True); subprocess.run(cmd, check=False)
PY
done
say "Q2 done"

# ------------------------------------------------- Q3: FNO ablations (TorchScript)
say "===== Q3: FNO ablations, TorchScript ====="
cd /home/jetson/jjyoo3/EDCNO || exit 1
export PYTHONPATH=$PWD
for rep in $(seq 1 "$REPS"); do
  "$PY_FD" - "$rep" "$CFG11_ABL" torchscript results/jetson_fno_unified_ablation <<'PY' >> "$LOG" 2>&1
import csv, os, subprocess, sys
rep, wanted, mode, out = sys.argv[1], set(sys.argv[2].split()), sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
for r in csv.DictReader(open("manifests/fno_jetson_manifest.csv")):
    if r["experiment_name"] not in wanted: continue
    tag = f'{r["experiment_name"]}_seed{r["selected_seed"]}_{mode}_fp32_rep{rep}'
    if os.path.exists(f"{out}/{tag}.json"): print(f"[SKIP] {tag}"); continue
    cmd = [os.environ["PY_FD"], "-m", "src.eval.benchmark_inference", "--mode", mode,
           "--checkpoint", r["checkpoint_path"], "--torchscript", r["torchscript_path"],
           "--input-bank", r["input_bank_path"], "--precision", "fp32", "--batch-size", "1",
           "--num-warmup", "30", "--num-iters", "100", "--results-dir", out, "--result-tag", tag]
    print(f"[RUN] {tag}", flush=True); subprocess.run(cmd, check=False)
PY
done
say "Q3 done"

# ------------------------------------------------- Q4: FNO spatial-resolution frontier
say "===== Q4: FNO frontier (synthetic banks beyond trained resolution) ====="
OUTF=results/jetson_fno_unified_frontier; mkdir -p "$OUTF"
BANK=artifacts/benchmark_inputs/frontier_synth
CK_BASE=artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt
TS_BASE=artifacts/torchscript/darcy_fno_base_r281_seed1.ts
front () {  # label ckpt ts res bank
  for rep in $(seq 1 "$REPS"); do
    local t="$1_on_$4_rep${rep}"
    [ -f "$OUTF/${t}.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    "$PY_FD" -m src.eval.benchmark_inference --mode torchscript \
      --checkpoint "$2" --torchscript "$3" --input-bank "$5" --precision fp32 \
      --batch-size 1 --num-warmup 30 --num-iters 100 \
      --results-dir "$OUTF" --result-tag "$t" >> "$LOG" 2>&1 \
      && say "[ OK ] $t" || say "[FAIL] $t"
  done
}
CK_LARGE=$("$PY_FD" -c "
import csv
for r in csv.DictReader(open('manifests/fno_jetson_manifest.csv')):
    if r['experiment_name']=='darcy_fno_large': print(r['checkpoint_path'])")
TS_LARGE=$("$PY_FD" -c "
import csv
for r in csv.DictReader(open('manifests/fno_jetson_manifest.csv')):
    if r['experiment_name']=='darcy_fno_large': print(r['torchscript_path'])")
for res in 421 561 701 841 981; do
  b="$BANK/darcy_r${res}_bank.pt"
  [ "$res" = 421 ] && b="artifacts/benchmark_inputs/darcy_r421_bank.pt"
  [ -f "$b" ] || { say "[MISS] bank for r${res}"; continue; }
  front darcy_base_trained281 "$CK_BASE" "$TS_BASE" "$res" "$b"
  front darcy_large_trained141 "$CK_LARGE" "$TS_LARGE" "$res" "$b"
done
say "Q4 done"

say "===== QUEUE FINISHED ====="
