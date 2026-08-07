"""Regenerates the PANEL_A / PANEL_B / PANEL_C literals in
plot_figure2_deployment_overview.py from the canonical per-family sources,
so the figure's numbers never have to be hand-copied again.

Run it, diff the printed dicts against what's in the figure script, and
paste over if anything moved (e.g. after a checkpoint retrain).

    python build_fig2_data.py

All four families are pulled as MEDIAN (p50) latency, FP32, batch size 1 --
TorchScript backend for FNO/DeepONet, eager for WNO/Sp2GNO (no TorchScript
path exists for their runtime in this harness). See the data-source
docstring at the top of plot_figure2_deployment_overview.py for exactly
which CSV/JSON tree backs each number.
"""
import csv
import glob
import json
import statistics
from collections import defaultdict

EDCNO = "/home/jetson/jjyoo3/EDCNO"
DEEPONET = "/home/jetson/jjyoo3/EDCNO_DeepONet"
WNO_ROOT = "/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks"
SP2GNO_ROOT = "/home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026"


# ---------------------------------------------------------------------------
# FNO -- median_ms column, torchscript + fp32 rows only.
# ---------------------------------------------------------------------------
def load_fno():
    rows = {}
    with open(f"{EDCNO}/results/artifacts/paper_fno_main_deployability_table.csv") as f:
        for r in csv.DictReader(f):
            if r["mode"] == "torchscript" and r["precision"] == "fp32":
                rows[r["tag"]] = float(r["median_ms"])
    with open(f"{EDCNO}/results/artifacts/paper_fno_resolution_scaling_table.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["tag"], float(r["median_ms"]))

    def g(tag):
        return round(rows[tag], 3)

    panel_a = {"Burgers": g("burgers_fno_base_seed3_torchscript_fp32"),
               "Darcy": g("darcy_fno_base_seed0_torchscript_fp32")}
    panel_b = {
        "Burgers": [g("burgers_fno_small_seed2_torchscript_fp32"),
                    g("burgers_fno_base_seed3_torchscript_fp32"),
                    g("burgers_fno_large_seed0_torchscript_fp32")],
        "Darcy": [g("darcy_fno_small_seed4_torchscript_fp32"),
                  g("darcy_fno_base_seed0_torchscript_fp32"),
                  g("darcy_fno_large_seed0_torchscript_fp32")],
    }
    panel_c = {
        "Burgers": [g("burgers_fno_base_r512_seed2_torchscript_fp32"),
                    g("burgers_fno_base_r1024_seed0_torchscript_fp32"),
                    g("burgers_fno_base_r2048_seed2_torchscript_fp32"),
                    g("burgers_fno_base_r4096_seed0_torchscript_fp32"),
                    g("burgers_fno_base_r8192_seed1_torchscript_fp32")],
        "Darcy": [g("darcy_fno_base_r85_seed2_torchscript_fp32"),
                  g("darcy_fno_base_r141_seed0_torchscript_fp32"),
                  g("darcy_fno_base_r211_seed1_torchscript_fp32"),
                  g("darcy_fno_base_r281_seed1_torchscript_fp32"),
                  g("darcy_fno_base_r421_seed1_torchscript_fp32")],
    }
    return panel_a, panel_b, panel_c


# ---------------------------------------------------------------------------
# DeepONet -- median_ms column, backend=torchscript + precision=fp32 rows.
# ---------------------------------------------------------------------------
def load_deeponet():
    rows = {}
    with open(f"{DEEPONET}/results/artifacts/deeponet_fp32_deployment_summary.csv") as f:
        for r in csv.DictReader(f):
            if r["backend"] == "torchscript" and r["precision"] == "fp32":
                rows[r["experiment_name"]] = float(r["median_ms"])

    def g(name):
        return round(rows[name], 3)

    panel_a = {"Burgers": g("burgers_deeponet_base"), "Darcy": g("darcy_deeponet_base")}
    panel_b = {
        "Burgers": [g("burgers_deeponet_small"), g("burgers_deeponet_base"), g("burgers_deeponet_large")],
        "Darcy": [g("darcy_deeponet_small"), g("darcy_deeponet_base"), g("darcy_deeponet_large")],
    }
    panel_c = {
        "Burgers": [g("burgers_deeponet_base_r512"), g("burgers_deeponet_base_r1024"),
                    g("burgers_deeponet_base_r2048"), g("burgers_deeponet_base_r4096"),
                    g("burgers_deeponet_base_r8192")],
        "Darcy": [g("darcy_deeponet_base_r85"), g("darcy_deeponet_base_r141"),
                  g("darcy_deeponet_base_r211"), g("darcy_deeponet_base_r281"),
                  g("darcy_deeponet_base_r421")],
    }
    return panel_a, panel_b, panel_c


# ---------------------------------------------------------------------------
# WNO -- p50_latency_ms from bench_wno_jetson_exact.py result.json, fp32_strict,
# averaged over the 3 reps in each run tag.
# ---------------------------------------------------------------------------
WNO_RUN_TAGS = [
    "wno_exact_20260704_140032",
    "wno_exact_new_r1024_r85_r211_20260712_130009",
    "wno_exact_new_r512_20260713_040509",
]


def load_wno():
    p50s = defaultdict(list)
    for tag in WNO_RUN_TAGS:
        for f in glob.glob(f"{WNO_ROOT}/results/jetson_wno_exact/{tag}/*/fp32_strict/result.json"):
            with open(f) as fh:
                d = json.load(fh)
            if d.get("status") != "success":
                continue
            case_id = f.split("/")[-3].rsplit("_rep", 1)[0]
            p50s[case_id].append(d["p50_latency_ms"])

    def g(case):
        return round(statistics.mean(p50s[case]), 3)

    panel_a = {"Burgers": g("wno_burgers_base_r2048"), "Darcy": g("wno_darcy_base_r141")}
    panel_b = {
        "Burgers": [g("wno_burgers_small_r2048"), g("wno_burgers_base_r2048"), g("wno_burgers_large_r2048")],
        "Darcy": [g("wno_darcy_small_r141"), g("wno_darcy_base_r141"), g("wno_darcy_large_r141")],
    }
    panel_c = {
        "Burgers": [g("wno_burgers_base_r512"), g("wno_burgers_base_r1024"), g("wno_burgers_base_r2048"),
                    g("wno_burgers_base_r4096"), g("wno_burgers_base_r8192")],
        "Darcy": [g("wno_darcy_base_r85"), g("wno_darcy_base_r141"), g("wno_darcy_base_r211"),
                  g("wno_darcy_base_r281"), g("wno_darcy_base_r421")],
    }
    return panel_a, panel_b, panel_c


# ---------------------------------------------------------------------------
# Sp2GNO -- p50_latency_ms from bench_sp2gno_jetson_exact.py's edge-summary
# CSV, fp32_strict, averaged over reps. The 20260705 suite's darcy r141/r211
# rows are excluded (cache/eigenbasis mismatch bug, latency-only fields are
# fine but excluded anyway for a single, unambiguous source per case).
# ---------------------------------------------------------------------------
SP2GNO_SUITE_ROOTS = [
    "sp2gno_exact_precision_all_20260705_025941",           # burgers only used
    "sp2gno_exact_precision_burgers_r512_r1024_r8192_20260712_080742",
    "sp2gno_exact_precision_darcy_r85_20260712_121120",
    "sp2gno_exact_precision_darcy_r141x3_r211_r281_r421_20260712_223106",
]
SP2GNO_EXCLUDE_FROM_OLD_ROOT = {
    "sp2gno_darcy_small_r141", "sp2gno_darcy_base_r141",
    "sp2gno_darcy_large_r141", "sp2gno_darcy_base_r211",
}


def load_sp2gno():
    p50s = defaultdict(list)
    for root in SP2GNO_SUITE_ROOTS:
        for f in glob.glob(f"{SP2GNO_ROOT}/inference_runs/{root}/*/reports/sp2gno_edge_summary_*.csv"):
            if f.endswith("_FAILED.csv"):
                continue
            with open(f) as fh:
                r = next(csv.DictReader(fh))
            if r["precision"] != "fp32_strict":
                continue
            if root == "sp2gno_exact_precision_all_20260705_025941" and r["case_id"] in SP2GNO_EXCLUDE_FROM_OLD_ROOT:
                continue
            p50s[r["case_id"]].append(float(r["p50_latency_ms"]))

    def g(case):
        return round(statistics.mean(p50s[case]), 3)

    panel_a = {"Burgers": g("sp2gno_burgers_base_s2048"), "Darcy": g("sp2gno_darcy_base_r141")}
    panel_b = {
        "Burgers": [g("sp2gno_burgers_small_s2048"), g("sp2gno_burgers_base_s2048"), g("sp2gno_burgers_large_s2048")],
        "Darcy": [g("sp2gno_darcy_small_r141"), g("sp2gno_darcy_base_r141"), g("sp2gno_darcy_large_r141")],
    }
    panel_c = {
        "Burgers": [g("sp2gno_burgers_base_r512"), g("sp2gno_burgers_base_r1024"), g("sp2gno_burgers_base_s2048"),
                    g("sp2gno_burgers_base_s4096"), g("sp2gno_burgers_base_r8192")],
        "Darcy": [g("sp2gno_darcy_base_r85"), g("sp2gno_darcy_base_r141"), g("sp2gno_darcy_base_r211"),
                  g("sp2gno_darcy_base_r281"), g("sp2gno_darcy_base_r421")],
    }
    return panel_a, panel_b, panel_c


def main():
    loaders = {"FNO": load_fno, "DeepONet": load_deeponet, "WNO": load_wno, "Sp2GNO": load_sp2gno}
    results = {fam: fn() for fam, fn in loaders.items()}

    for panel_idx, panel_name in enumerate(["PANEL_A", "PANEL_B", "PANEL_C"]):
        print(f"\n{panel_name} = {{")
        for problem in ["Burgers", "Darcy"]:
            print(f'    "{problem}": {{')
            for fam in ["FNO", "DeepONet", "WNO", "Sp2GNO"]:
                val = results[fam][panel_idx][problem]
                print(f'        "{fam}": {val!r},')
            print("    },")
        print("}")


if __name__ == "__main__":
    main()
