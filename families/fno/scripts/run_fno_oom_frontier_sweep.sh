#!/usr/bin/env bash
set -euo pipefail

cd ~/jjyoo3/EDCNO || exit 1
export PYTHONPATH="$PWD"

PYTHON_BIN="$(which python)"
OUTDIR="results/jetson_fno_oom_frontier"
BANKDIR="artifacts/benchmark_inputs/frontier_synth"
mkdir -p "${OUTDIR}" "${BANKDIR}"

STATUS_CSV="${OUTDIR}/frontier_status.csv"
if [[ ! -f "${STATUS_CSV}" ]]; then
  echo "tag,model,input_bank,status,exit_code,json_exists,stderr_path" > "${STATUS_CSV}"
fi

build_bank () {
  local src="$1"
  local dst="$2"
  local target="$3"
  "${PYTHON_BIN}" scripts/build_synthetic_frontier_banks.py \
    --src-bank "${src}" \
    --dst-bank "${dst}" \
    --target-res "${target}"
}

run_one () {
  local tag="$1"
  local checkpoint="$2"
  local torchscript="$3"
  local input_bank="$4"

  local tegra_log="${OUTDIR}/${tag}_tegrastats.log"
  local stdout_log="${OUTDIR}/${tag}.stdout"
  local stderr_log="${OUTDIR}/${tag}.stderr"
  local out_json="${OUTDIR}/${tag}.json"

  echo "=================================================="
  echo "[RUN ] ${tag}"
  echo "=================================================="

  tegrastats --interval 100 --logfile "${tegra_log}" &
  TPID=$!

  set +e
  "${PYTHON_BIN}" -m src.eval.benchmark_inference \
    --mode torchscript \
    --checkpoint "${checkpoint}" \
    --torchscript "${torchscript}" \
    --input-bank "${input_bank}" \
    --precision fp32 \
    --batch-size 1 \
    --num-warmup 5 \
    --num-iters 20 \
    --device cuda \
    --results-dir "${OUTDIR}" \
    --result-tag "${tag}" \
    > "${stdout_log}" 2> "${stderr_log}"
  STATUS=$?
  set -e

  kill ${TPID} 2>/dev/null || true
  wait ${TPID} 2>/dev/null || true

  JSON_EXISTS=0
  [[ -f "${out_json}" ]] && JSON_EXISTS=1

  RUN_STATUS="fail"
  if [[ ${STATUS} -eq 0 && ${JSON_EXISTS} -eq 1 ]]; then
    RUN_STATUS="success"
  elif grep -Ei "out of memory|cuda out of memory|allocation failed|CUDNN_STATUS_ALLOC_FAILED|INTERNAL ASSERT FAILED" "${stderr_log}" >/dev/null 2>&1; then
    RUN_STATUS="oom_or_alloc_fail"
  fi

  echo "${tag},${checkpoint},${input_bank},${RUN_STATUS},${STATUS},${JSON_EXISTS},${stderr_log}" >> "${STATUS_CSV}"
  echo "[DONE] ${tag} => ${RUN_STATUS}"
}

# Build larger synthetic banks from real r421 bank
build_bank artifacts/benchmark_inputs/darcy_r421_bank.pt "${BANKDIR}/darcy_r561_bank.pt" 561
build_bank artifacts/benchmark_inputs/darcy_r421_bank.pt "${BANKDIR}/darcy_r701_bank.pt" 701
build_bank artifacts/benchmark_inputs/darcy_r421_bank.pt "${BANKDIR}/darcy_r841_bank.pt" 841
build_bank artifacts/benchmark_inputs/darcy_r421_bank.pt "${BANKDIR}/darcy_r981_bank.pt" 981

# Base frontier
run_one "darcy_base_r281_on_561_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
  "${BANKDIR}/darcy_r561_bank.pt"

run_one "darcy_base_r281_on_701_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
  "${BANKDIR}/darcy_r701_bank.pt"

run_one "darcy_base_r281_on_841_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
  "${BANKDIR}/darcy_r841_bank.pt"

run_one "darcy_base_r281_on_981_ts_fp32" \
  artifacts/checkpoints/darcy_fno_base_r281_seed1_best.pt \
  artifacts/torchscript/darcy_fno_base_r281_seed1.ts \
  "${BANKDIR}/darcy_r981_bank.pt"

# Large frontier
run_one "darcy_large_r141_on_561_ts_fp32" \
  artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  artifacts/torchscript/darcy_fno_large_seed0.ts \
  "${BANKDIR}/darcy_r561_bank.pt"

run_one "darcy_large_r141_on_701_ts_fp32" \
  artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  artifacts/torchscript/darcy_fno_large_seed0.ts \
  "${BANKDIR}/darcy_r701_bank.pt"

run_one "darcy_large_r141_on_841_ts_fp32" \
  artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  artifacts/torchscript/darcy_fno_large_seed0.ts \
  "${BANKDIR}/darcy_r841_bank.pt"

run_one "darcy_large_r141_on_981_ts_fp32" \
  artifacts/checkpoints/darcy_fno_large_seed0_best.pt \
  artifacts/torchscript/darcy_fno_large_seed0.ts \
  "${BANKDIR}/darcy_r981_bank.pt"