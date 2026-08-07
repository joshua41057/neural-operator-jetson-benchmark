#!/usr/bin/env bash
set -euo pipefail

OUT="fno_code_audit_main.txt"

FILES=(
  "requirements.txt"
  "setup.txt"
  "manifests/fno_jetson_manifest.csv"

  "src/__init__.py"

  "src/models/__init__.py"
  "src/models/fno.py"
  "src/models/layers.py"

  "src/eval/__init__.py"
  "src/eval/common.py"
  "src/eval/benchmark_inference.py"
  "src/eval/benchmark_energy_inference.py"
  "src/eval/benchmark_precision_inference.py"
  "src/eval/benchmark_precision_fft_safe.py"
  "src/eval/evaluate_fno.py"
  "src/eval/export_torchscript.py"
  "src/eval/module_profile_eager.py"
  "src/eval/profile_inference.py"

  "src/data/__init__.py"
  "src/data/datasets.py"
  "src/data/mat_reader.py"
  "src/data/preprocessing.py"

  "src/train/__init__.py"
  "src/train/engine.py"
  "src/train/train_fno.py"

  "src/utils/__init__.py"
  "src/utils/config.py"
  "src/utils/coords.py"
  "src/utils/device.py"
  "src/utils/io.py"
  "src/utils/metrics.py"
  "src/utils/normalizer.py"
  "src/utils/seed.py"

  "scripts/aggregate_summaries.py"
  "scripts/build_fno_final_artifacts.py"
  "scripts/build_fno_latex_tables.py"
  "scripts/build_fno_validity_table.py"
  "scripts/build_paper_artifacts.py"
  "scripts/build_synthetic_frontier_banks.py"
  "scripts/check_eager_torchscript_consistency.py"
  "scripts/export_ncu_frontier_reports.sh"
  "scripts/export_ncu_reports.sh"
  "scripts/export_ncu_sched_reports.sh"
  "scripts/inspect_mat.py"
  "scripts/plot_fno_paper_figures.py"
  "scripts/rebuild_fno_ablation_validity_table.py"
  "scripts/rebuild_fno_validity_table_recursive.py"
  "scripts/run_fno_energy_long.sh"
  "scripts/run_fno_frontier_stress.sh"
  "scripts/run_fno_matrix.sh"
  "scripts/run_fno_oom_frontier_sweep.sh"
  "scripts/run_fno_precision_fft_safe_subset.sh"
  "scripts/run_fno_precision_frontier_tf32.sh"
  "scripts/run_fno_precision_matrix.sh"
  "scripts/run_fno_profile_module.sh"
  "scripts/run_fno_profile_ncu_frontier.sh"
  "scripts/run_fno_profile_ncu_reduced.sh"
  "scripts/run_fno_profile_ncu_sched.sh"
  "scripts/run_fno_profile_ncu.sh"
  "scripts/run_fno_profile_nsys.sh"
  "scripts/run_fno_profile_perf.sh"
  "scripts/run_fno_sample_variance.sh"
  "scripts/run_fno_sustained.sh"
  "scripts/sanity_check_fno_package.py"
  "scripts/summarize_extra_fno_results.py"
  "scripts/summarize_fno_energy_long.py"
  "scripts/summarize_fno_precision_results.py"
  "scripts/summarize_jetson_fno_results.py"
  "scripts/summarize_nsys_profiles.py"
  "scripts/summarize_precision_fft_safe.py"
)

{
  echo "# FNO Jetson Code Audit Dump"
  echo "# Generated at: $(date -Iseconds)"
  echo "# Repo: $(pwd)"
  echo

  for f in "${FILES[@]}"; do
    echo
    echo "================================================================================"
    echo "FILE: ${f}"
    echo "================================================================================"
    echo

    if [[ -f "$f" ]]; then
      cat "$f"
    else
      echo "[MISSING FILE]"
    fi

    echo
  done
} > "$OUT"

echo "Wrote $OUT"
wc -l "$OUT"
du -h "$OUT"
