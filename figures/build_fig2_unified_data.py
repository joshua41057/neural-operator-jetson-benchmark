#!/usr/bin/env python3
"""Regenerate fig2_full_data.json from the unified-protocol short-run sweeps.

Replaces build_fig2_full_data.py, which read the family-native sweeps. Every
value here comes from the cross-family unified protocol described in Section 4.3:
a preloaded device-resident batch-size-one request, a 20 s time-based warmup,
100 timed passes, R = 3 repetitions summarised as the mean of the per-repetition
statistics, linear-interpolation percentiles, and a pinned-clock board state
asserted by preflight before every sweep.

Panel composition is identical to the published figure: the model-scale panels
(a)/(b) use the scale-group checkpoints (burgers_fno_base, darcy_fno_base) while
the ladders (c)/(d) use the resolution-group checkpoints (burgers_fno_base_r2048,
darcy_fno_base_r141) at their shared rung. These are separately trained
checkpoints with their own held-out error, tabulated separately in the appendix,
so they are not interchangeable.
"""
import csv, glob, json, os, statistics as st
from collections import defaultdict

FNO_D = "/home/jetson/jjyoo3/EDCNO/results/jetson_fno_unified"
DON_D = "/home/jetson/jjyoo3/EDCNO_DeepONet/results/jetson_deeponet_unified"
WNO_D = "/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks/results/jetson_wno_exact/wno_unified_shortrun"
SP_D = "/home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026/inference_runs/sp2gno_unified_shortrun"
OUT = "/home/jetson/jjyoo3/edge_figures/fig2_full_data.json"


def _mean_pair(recs):
    """(mean median, mean P95) over repetitions, rounded to the reported precision."""
    return [round(st.mean(m for m, _ in recs), 4), round(st.mean(p for _, p in recs), 4)]


def load_fno():
    """TorchScript rows only: Fig. 2 plots the TorchScript backend for FNO."""
    g = defaultdict(list)
    for p in glob.glob(os.path.join(FNO_D, "*_torchscript_fp32_rep*.json")):
        d = json.load(open(p))
        g[os.path.basename(p).rsplit("_seed", 1)[0]].append((d["median_ms"], d["p95_ms"]))
    return g


def load_don():
    """TorchScript rows only: Fig. 2 plots the TorchScript backend for DeepONet."""
    g = defaultdict(list)
    for p in glob.glob(os.path.join(DON_D, "*_torchscript_fp32_rep*.json")):
        d = json.load(open(p))
        g[os.path.basename(p).split("_torchscript")[0]].append((d["median_ms"], d["p95_ms"]))
    return g


def load_wno():
    g = defaultdict(list)
    for p in glob.glob(os.path.join(WNO_D, "*", "fp32_strict", "result.json")):
        d = json.load(open(p))
        if d.get("status") != "success":
            continue
        c = os.path.basename(os.path.dirname(os.path.dirname(p))).rsplit("_rep", 1)[0]
        g[c].append((d["p50_latency_ms"], d["p95_latency_ms"]))
    return g


def load_sp():
    g = defaultdict(list)
    for p in glob.glob(os.path.join(SP_D, "*", "reports", "sp2gno_edge_summary_*.csv")):
        for d in csv.DictReader(open(p)):
            if d.get("status") == "success":
                g[d["case_id"]].append((float(d["p50_latency_ms"]), float(d["p95_latency_ms"])))
    return g


# (family -> panel -> [(config key, parameter count)])
# "b"/"c" are the model-scale panels (a)/(b); "d"/"e" are the ladders (c)/(d).
SPEC = {
    "FNO": dict(src=load_fno,
        b=[("burgers_fno_small", 72033), ("burgers_fno_base", 235537), ("burgers_fno_large", 820033)],
        c=[("darcy_fno_small", 667713), ("darcy_fno_base", 3287553), ("darcy_fno_large", 28345217)],
        d=["burgers_fno_base_r512", "burgers_fno_base_r1024", "burgers_fno_base_r2048",
           "burgers_fno_base_r4096", "burgers_fno_base_r8192"],
        e=["darcy_fno_base_r85", "darcy_fno_base_r141", "darcy_fno_base_r211",
           "darcy_fno_base_r281", "darcy_fno_base_r421"]),
    "DeepONet": dict(src=load_don,
        b=[("burgers_deeponet_small", 82945), ("burgers_deeponet_base", 461313), ("burgers_deeponet_large", 2365441)],
        c=[("darcy_deeponet_small", 514433), ("darcy_deeponet_base", 2639361), ("darcy_deeponet_large", 7603713)],
        d=["burgers_deeponet_base_r512", "burgers_deeponet_base_r1024", "burgers_deeponet_base_r2048",
           "burgers_deeponet_base_r4096", "burgers_deeponet_base_r8192"],
        e=["darcy_deeponet_base_r85", "darcy_deeponet_base_r141", "darcy_deeponet_base_r211",
           "darcy_deeponet_base_r281", "darcy_deeponet_base_r421"]),
    "WNO": dict(src=load_wno,
        b=[("wno_burgers_small_r2048", 74859), ("wno_burgers_base_r2048", 242457), ("wno_burgers_large_r2048", 820695)],
        c=[("wno_darcy_small_r141", 91037), ("wno_darcy_base_r141", 232001), ("wno_darcy_large_r141", 813197)],
        d=["wno_burgers_base_r512", "wno_burgers_base_r1024", "wno_burgers_base_r2048",
           "wno_burgers_base_r4096", "wno_burgers_base_r8192"],
        e=["wno_darcy_base_r85", "wno_darcy_base_r141", "wno_darcy_base_r211",
           "wno_darcy_base_r281", "wno_darcy_base_r421"]),
    "Sp2GNO": dict(src=load_sp,
        b=[("sp2gno_burgers_small_s2048", 70645), ("sp2gno_burgers_base_s2048", 234347), ("sp2gno_burgers_large_s2048", 814325)],
        c=[("sp2gno_darcy_small_r141", 70658), ("sp2gno_darcy_base_r141", 234371), ("sp2gno_darcy_large_r141", 814370)],
        d=["sp2gno_burgers_base_r512", "sp2gno_burgers_base_r1024", "sp2gno_burgers_base_s2048",
           "sp2gno_burgers_base_s4096", "sp2gno_burgers_base_r8192"],
        e=["sp2gno_darcy_base_r85", "sp2gno_darcy_base_r141", "sp2gno_darcy_base_r211",
           "sp2gno_darcy_base_r281", "sp2gno_darcy_base_r421"]),
}

out, missing = {}, []
for fam, spec in SPEC.items():
    g = spec["src"]()
    fam_out = {}
    for panel in ("b", "c"):
        rows = []
        for key, params in spec[panel]:
            if key not in g:
                missing.append(f"{fam}/{panel}/{key}"); continue
            m, p = _mean_pair(g[key])
            rows.append([m, p, params])
        fam_out[panel] = rows
    for panel in ("d", "e"):
        rows = []
        for key in spec[panel]:
            if key not in g:
                missing.append(f"{fam}/{panel}/{key}"); continue
            rows.append(_mean_pair(g[key]))
        fam_out[panel] = rows
    # "a" is the reference (Base) point of each problem, reused by the plot header
    fam_out["a"] = {"Burgers": fam_out["b"][1][:2], "Darcy": fam_out["c"][1][:2]}
    out[fam] = fam_out

if missing:
    raise SystemExit("missing configurations:\n  " + "\n  ".join(missing))

json.dump(out, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
for fam in out:
    print(f"  {fam:9s} (a)Burgers={out[fam]['a']['Burgers'][0]:8.3f}  (b)Darcy={out[fam]['a']['Darcy'][0]:8.3f}")
