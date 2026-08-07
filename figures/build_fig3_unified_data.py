#!/usr/bin/env python3
"""Regenerate fig3_precision_data.json from the unified-protocol sustained sweeps.

Replaces the hard-coded BLOCKS table that fig_precision_sweep.py used to carry
inline. Every latency/energy pair here is the same aggregate that the appendix
sustained tables (22, 29, 34, 38) and the main-text telemetry table (9) are built
from, so the figure cannot drift away from the tables.

Latency is the mean over R = 3 repetitions of the per-repetition median of a
120-second sustained window; energy is average board power divided by throughput,
with power the mean instantaneous VDD_IN over the same window.

Admission status is *not* re-derived here. A precision path appears as a failure
only where the family's validity summary recorded a non-Success paper_status:
WNO FP16 native fails at both workloads on a dtype mismatch, WNO FP16 autocast
fails numerical admission on Darcy, and FNO's cuFFT kernels reject the reduced
precisions outright so those cells were never attempted.
"""
import json
import os
import pickle
import subprocess
import sys

AGG = "/tmp/claude-1000/-home-jetson/2fbc43e9-f9ac-49f3-b258-2bd622821d3a/scratchpad/agg_all.pkl"
OUT = "/home/jetson/jjyoo3/edge_figures/fig3_precision_data.json"

PRECISIONS = ["FP32", "TF32", "BF16 auto", "FP16 auto", "FP16 native"]
SUFFIX = {"FP32": "fp32_strict", "TF32": "tf32", "BF16 auto": "bf16_autocast",
          "FP16 auto": "fp16_autocast", "FP16 native": "fp16_native"}

# (family, workload, aggregate table, key prefix)
BLOCKS = [
    ("FNO",      "Burgers 2048", "fno_s", "burgers_base_r2048"),
    ("FNO",      "Darcy 141²", "fno_s", "darcy_base_r141"),
    ("DeepONet", "Burgers 2048", "don_s", "burgers_base"),
    ("DeepONet", "Darcy 141²", "don_s", "darcy_r141"),
    ("WNO",      "Burgers 2048", "wno_s", "wno_burgers_base_r2048"),
    ("WNO",      "Darcy 141²", "wno_s", "wno_darcy_base_r141"),
    ("Sp2GNO",   "Burgers 2048", "sp_s", "sp2gno_burgers_base_s2048"),
    ("Sp2GNO",   "Darcy 141²", "sp_s", "sp2gno_darcy_base_r141"),
]

# Paths that were attempted and did not earn admission, with the recorded reason.
# Sources: WNO wno_final_validity_summary.csv, Sp2GNO sp2gno_final_validity_summary.csv
# (both retained under _PREVIOUS_RUNS/_RETAINED_SOURCES/), and the FNO cuFFT
# dtype/size restriction documented in Section 6.2.
FAIL = {
    ("FNO", "Burgers 2048", "BF16 auto"): "cuFFT unsupported",
    ("FNO", "Burgers 2048", "FP16 auto"): "cuFFT unsupported",
    ("FNO", "Burgers 2048", "FP16 native"): "cuFFT unsupported",
    ("FNO", "Darcy 141²", "BF16 auto"): "cuFFT unsupported",
    ("FNO", "Darcy 141²", "FP16 auto"): "cuFFT unsupported",
    ("FNO", "Darcy 141²", "FP16 native"): "cuFFT unsupported",
    ("WNO", "Burgers 2048", "FP16 native"): "dtype mismatch",
    ("WNO", "Darcy 141²", "FP16 auto"): "NaN (unstable)",
    ("WNO", "Darcy 141²", "FP16 native"): "dtype mismatch",
}

if not os.path.exists(AGG):
    sys.exit(f"aggregate not found: {AGG}\nrun agg_all.py first")
D = pickle.load(open(AGG, "rb"))

out, missing = [], []
for fam, workload, table, prefix in BLOCKS:
    cells = {}
    for prec in PRECISIONS:
        if (fam, workload, prec) in FAIL:
            cells[prec] = [None, None, FAIL[(fam, workload, prec)]]
            continue
        key = f"{prefix}_{SUFFIX[prec]}"
        v = D[table].get(key)
        if v is None:
            missing.append(f"{fam}/{workload}/{prec} -> {table}[{key}]")
            continue
        cells[prec] = [round(v["med"], 3), round(v["J"], 4), None]
    out.append({"family": fam, "workload": workload, "cells": cells})

if missing:
    raise SystemExit("missing configurations:\n  " + "\n  ".join(missing))

json.dump(out, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
for b in out:
    ok = [p for p in PRECISIONS if b["cells"][p][0] is not None]
    lo = min(b["cells"][p][0] for p in ok)
    hi = max(b["cells"][p][0] for p in ok)
    print(f"  {b['family']:9s} {b['workload']:12s} {len(ok)}/5 admitted  "
          f"{lo:8.3f}-{hi:8.3f} ms")
