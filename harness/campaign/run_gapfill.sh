#!/usr/bin/env bash
# Gap-fill: rows present in Tables 17/22/26 that the unified sweeps did not cover.
#   G1  FNO short-run       burgers_fno_base_r2048, darcy_fno_base_r141  x {eager,TS} x R=3
#   G2  DeepONet short-run  burgers_deeponet_base_r2048, darcy_deeponet_base_r141 x {eager,TS} x R=3
#   G3  FNO frontier sustained  base@281 and large@141 checkpoints evaluated at 421x421 x {FP32,TF32} x R=3
set -u
PY_FD="${PY_FD:-/home/jetson/miniforge3/envs/vs_wno/bin/python}"; export PY_FD
export PYTHONNOUSERSITE=1
. /home/jetson/jjyoo3/preflight_clocks.sh
preflight_clocks || exit 1
LOG=/home/jetson/jjyoo3/gapfill.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

short () {  # manifest cfgnames outdir projdir tagfmt
  cd "$4"; export PYTHONPATH=$PWD
  for rep in 1 2 3; do for mode in eager torchscript; do
    "$PY_FD" - "$rep" "$2" "$mode" "$3" "$1" "$5" <<'PY' >> "$LOG" 2>&1
import csv, os, subprocess, sys
rep,wanted,mode,out,man,fmt = sys.argv[1],set(sys.argv[2].split()),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]
os.makedirs(out, exist_ok=True)
for r in csv.DictReader(open(man)):
    if r["experiment_name"] not in wanted: continue
    tag = fmt.format(n=r["experiment_name"], s=r.get("selected_seed",""), m=mode, rep=rep)
    if os.path.exists(f"{out}/{tag}.json"): print(f"[SKIP] {tag}"); continue
    cmd=[os.environ["PY_FD"],"-m","src.eval.benchmark_inference","--mode",mode,
         "--checkpoint",r["checkpoint_path"],"--input-bank",r["input_bank_path"],
         "--precision","fp32","--batch-size","1","--num-warmup","30","--num-iters","100",
         "--results-dir",out,"--result-tag",tag]
    if mode=="torchscript": cmd+=["--torchscript",r["torchscript_path"]]
    print(f"[RUN] {tag}",flush=True); subprocess.run(cmd,check=False)
PY
  done; done
}

say "===== G1: FNO short-run gap ====="
short manifests/fno_jetson_manifest.csv "burgers_fno_base_r2048 darcy_fno_base_r141" \
      results/jetson_fno_unified /home/jetson/jjyoo3/EDCNO "{n}_seed{s}_{m}_fp32_rep{rep}"
say "G1 done"

say "===== G2: DeepONet short-run gap ====="
short manifests/deeponet_jetson_manifest.csv "burgers_deeponet_base_r2048 darcy_deeponet_base_r141" \
      results/jetson_deeponet_unified /home/jetson/jjyoo3/EDCNO_DeepONet "{n}_{m}_fp32_rep{rep}"
say "G2 done"

say "===== G3: FNO frontier sustained ====="
cd /home/jetson/jjyoo3/EDCNO; export PYTHONPATH=$PWD
OUT=results/jetson_fno_unified_sustained
BANK=artifacts/benchmark_inputs/darcy_r421_bank.pt
fr () {  # label ckpt ts precision
  for rep in 1 2 3; do
    t="$1_frontier_$4_rep${rep}"
    [ -f "$OUT/${t}.json" ] && { say "[SKIP] $t"; continue; }
    say "[RUN] $t"
    tegrastats --interval 100 --logfile "$OUT/${t}_tegrastats.log" & local tp=$!
    sleep 0.3
    "$PY_FD" -m src.eval.benchmark_energy_inference --mode torchscript \
      --checkpoint "$2" --torchscript "$3" --input-bank "$BANK" --sample-index 0 \
      --batch-size 1 --precision-mode "$4" --warmup-seconds 20 --measure-seconds 120 \
      --results-dir "$OUT" --result-tag "$t" >> "$LOG" 2>&1
    local rc=$?; kill "$tp" 2>/dev/null; wait "$tp" 2>/dev/null
    [ $rc -eq 0 ] && say "[ OK ] $t" || say "[FAIL] $t"
    sleep 3
  done
}
read -r CKB TSB < <("$PY_FD" -c "
import csv
for r in csv.DictReader(open('manifests/fno_jetson_manifest.csv')):
    if r['experiment_name']=='darcy_fno_base_r281': print(r['checkpoint_path'],r['torchscript_path'])")
read -r CKL TSL < <("$PY_FD" -c "
import csv
for r in csv.DictReader(open('manifests/fno_jetson_manifest.csv')):
    if r['experiment_name']=='darcy_fno_large': print(r['checkpoint_path'],r['torchscript_path'])")
for p in fp32_strict tf32; do
  fr darcy_base_on421 "$CKB" "$TSB" "$p"
  fr darcy_large_on421 "$CKL" "$TSL" "$p"
done
say "G3 done"
say "===== GAPFILL FINISHED ====="
