"""Pulls median + P95 FP32 latency for all 4 families across:
  Panel A: cross-family @ base scale, Burgers r2048 / Darcy r141
  Panel B/C: model scale (small/base/large) @ Burgers r2048 / Darcy r141
             -- also carries each point's parameter count, so the scale
             panels can be plotted against actual param count (log x) rather
             than an evenly-spaced Small/Base/Large category.
  Panel D/E: resolution sweep, Burgers r512-r8192 / Darcy r85-r421 @ base scale

Sources (median_ms / p95_ms columns throughout):
  FNO      -> EDCNO/results/artifacts/{paper_fno_main_deployability_table,paper_fno_resolution_scaling_table}.csv
              (resolution_scaling_table has no p95 for burgers r512-r4096; use json_runs_raw.csv instead, which has both)
              Params from appendix_fno_median_revised.tex (Params column, verified against json_runs_raw.csv's own metadata).
  DeepONet -> EDCNO_DeepONet/results/artifacts/deeponet_fp32_deployment_summary.csv (backend=torchscript)
              Params from appendix_deeponet_median_revised.tex.
  WNO      -> WNO_Sp2GNO_Benchmarks/results/jetson_wno_exact/{3 run tags}/*/fp32_strict/result.json (p50/p95, avg of 3 reps)
              Params from appendix_wno_sp2gno_median_revised.tex.
  Sp2GNO   -> VirSO/sp2gno/sp2gno_new_benchmarks_june_2026/inference_runs/{4 suite roots}/*/reports/*.csv (p50/p95, avg of 3 reps)
              Params from appendix_wno_sp2gno_median_revised.tex.
"""
import csv, glob, json, statistics
from collections import defaultdict

EDCNO = "/home/jetson/jjyoo3/EDCNO"
DEEPONET = "/home/jetson/jjyoo3/EDCNO_DeepONet"
WNO_ROOT = "/home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks"
SP2GNO_ROOT = "/home/jetson/VirSO/sp2gno/sp2gno_new_benchmarks_june_2026"

# ---------------------------------------------------------------------------
FNO_PARAMS = {"Burgers": [72033, 235537, 820033], "Darcy": [667713, 3287553, 28345217]}


def load_fno():
    raw = {}
    for r in csv.DictReader(open(f"{EDCNO}/results/artifacts/json_runs_raw.csv")):
        raw.setdefault(r["experiment_name"], r)

    def g(tag):
        r = raw[tag]
        return round(float(r["median_ms"]), 4), round(float(r["p95_ms"]), 4)

    def g3(tag, params):
        return (*g(tag), params)

    return {
        "a": {"Burgers": g("burgers_fno_base_seed3_torchscript_fp32"),
              "Darcy": g("darcy_fno_base_seed0_torchscript_fp32")},
        "b": [g3("burgers_fno_small_seed2_torchscript_fp32", FNO_PARAMS["Burgers"][0]),
              g3("burgers_fno_base_seed3_torchscript_fp32", FNO_PARAMS["Burgers"][1]),
              g3("burgers_fno_large_seed0_torchscript_fp32", FNO_PARAMS["Burgers"][2])],
        "c": [g3("darcy_fno_small_seed4_torchscript_fp32", FNO_PARAMS["Darcy"][0]),
              g3("darcy_fno_base_seed0_torchscript_fp32", FNO_PARAMS["Darcy"][1]),
              g3("darcy_fno_large_seed0_torchscript_fp32", FNO_PARAMS["Darcy"][2])],
        "d": [g("burgers_fno_base_r512_seed2_torchscript_fp32"),
              g("burgers_fno_base_r1024_seed0_torchscript_fp32"),
              g("burgers_fno_base_r2048_seed2_torchscript_fp32"),
              g("burgers_fno_base_r4096_seed0_torchscript_fp32"),
              g("burgers_fno_base_r8192_seed1_torchscript_fp32")],
        "e": [g("darcy_fno_base_r85_seed2_torchscript_fp32"),
              g("darcy_fno_base_r141_seed0_torchscript_fp32"),
              g("darcy_fno_base_r211_seed1_torchscript_fp32"),
              g("darcy_fno_base_r281_seed1_torchscript_fp32"),
              g("darcy_fno_base_r421_seed1_torchscript_fp32")],
    }


DEEPONET_PARAMS = {"Burgers": [82945, 461313, 2365441], "Darcy": [514433, 2639361, 7603713]}


def load_deeponet():
    rows = {}
    for r in csv.DictReader(open(f"{DEEPONET}/results/artifacts/deeponet_fp32_deployment_summary.csv")):
        if r["backend"] == "torchscript" and r["precision"] == "fp32":
            rows[r["experiment_name"]] = r

    def g(name):
        r = rows[name]
        return round(float(r["median_ms"]), 4), round(float(r["p95_ms"]), 4)

    def g3(name, params):
        return (*g(name), params)

    return {
        "a": {"Burgers": g("burgers_deeponet_base"), "Darcy": g("darcy_deeponet_base")},
        "b": [g3("burgers_deeponet_small", DEEPONET_PARAMS["Burgers"][0]),
              g3("burgers_deeponet_base", DEEPONET_PARAMS["Burgers"][1]),
              g3("burgers_deeponet_large", DEEPONET_PARAMS["Burgers"][2])],
        "c": [g3("darcy_deeponet_small", DEEPONET_PARAMS["Darcy"][0]),
              g3("darcy_deeponet_base", DEEPONET_PARAMS["Darcy"][1]),
              g3("darcy_deeponet_large", DEEPONET_PARAMS["Darcy"][2])],
        "d": [g("burgers_deeponet_base_r512"), g("burgers_deeponet_base_r1024"), g("burgers_deeponet_base_r2048"),
              g("burgers_deeponet_base_r4096"), g("burgers_deeponet_base_r8192")],
        "e": [g("darcy_deeponet_base_r85"), g("darcy_deeponet_base_r141"), g("darcy_deeponet_base_r211"),
              g("darcy_deeponet_base_r281"), g("darcy_deeponet_base_r421")],
    }


WNO_RUN_TAGS = [
    "wno_exact_20260704_140032",
    "wno_exact_new_r1024_r85_r211_20260712_130009",
    "wno_exact_new_r512_20260713_040509",
]
WNO_PARAMS = {"Burgers": [74859, 242457, 820695], "Darcy": [91037, 232001, 813197]}


def load_wno():
    p50s, p95s = defaultdict(list), defaultdict(list)
    for tag in WNO_RUN_TAGS:
        for f in glob.glob(f"{WNO_ROOT}/results/jetson_wno_exact/{tag}/*/fp32_strict/result.json"):
            d = json.load(open(f))
            if d.get("status") != "success":
                continue
            case = f.split("/")[-3].rsplit("_rep", 1)[0]
            p50s[case].append(d["p50_latency_ms"])
            p95s[case].append(d["p95_latency_ms"])

    def g(case):
        return round(statistics.mean(p50s[case]), 4), round(statistics.mean(p95s[case]), 4)

    def g3(case, params):
        return (*g(case), params)

    return {
        "a": {"Burgers": g("wno_burgers_base_r2048"), "Darcy": g("wno_darcy_base_r141")},
        "b": [g3("wno_burgers_small_r2048", WNO_PARAMS["Burgers"][0]),
              g3("wno_burgers_base_r2048", WNO_PARAMS["Burgers"][1]),
              g3("wno_burgers_large_r2048", WNO_PARAMS["Burgers"][2])],
        "c": [g3("wno_darcy_small_r141", WNO_PARAMS["Darcy"][0]),
              g3("wno_darcy_base_r141", WNO_PARAMS["Darcy"][1]),
              g3("wno_darcy_large_r141", WNO_PARAMS["Darcy"][2])],
        "d": [g("wno_burgers_base_r512"), g("wno_burgers_base_r1024"), g("wno_burgers_base_r2048"),
              g("wno_burgers_base_r4096"), g("wno_burgers_base_r8192")],
        "e": [g("wno_darcy_base_r85"), g("wno_darcy_base_r141"), g("wno_darcy_base_r211"),
              g("wno_darcy_base_r281"), g("wno_darcy_base_r421")],
    }


SP2GNO_SUITE_ROOTS = [
    "sp2gno_exact_precision_all_20260705_025941",
    "sp2gno_exact_precision_burgers_r512_r1024_r8192_20260712_080742",
    "sp2gno_exact_precision_darcy_r85_20260712_121120",
    "sp2gno_exact_precision_darcy_r141x3_r211_r281_r421_20260712_223106",
]
SP2GNO_EXCLUDE_FROM_OLD_ROOT = {
    "sp2gno_darcy_small_r141", "sp2gno_darcy_base_r141",
    "sp2gno_darcy_large_r141", "sp2gno_darcy_base_r211",
}
SP2GNO_PARAMS = {"Burgers": [70645, 234347, 814325], "Darcy": [70658, 234371, 814370]}


def load_sp2gno():
    p50s, p95s = defaultdict(list), defaultdict(list)
    for root in SP2GNO_SUITE_ROOTS:
        for f in glob.glob(f"{SP2GNO_ROOT}/inference_runs/{root}/*/reports/sp2gno_edge_summary_*.csv"):
            if f.endswith("_FAILED.csv"):
                continue
            r = next(csv.DictReader(open(f)))
            if r["precision"] != "fp32_strict":
                continue
            if root == "sp2gno_exact_precision_all_20260705_025941" and r["case_id"] in SP2GNO_EXCLUDE_FROM_OLD_ROOT:
                continue
            p50s[r["case_id"]].append(float(r["p50_latency_ms"]))
            p95s[r["case_id"]].append(float(r["p95_latency_ms"]))

    def g(case):
        return round(statistics.mean(p50s[case]), 4), round(statistics.mean(p95s[case]), 4)

    def g3(case, params):
        return (*g(case), params)

    return {
        "a": {"Burgers": g("sp2gno_burgers_base_s2048"), "Darcy": g("sp2gno_darcy_base_r141")},
        "b": [g3("sp2gno_burgers_small_s2048", SP2GNO_PARAMS["Burgers"][0]),
              g3("sp2gno_burgers_base_s2048", SP2GNO_PARAMS["Burgers"][1]),
              g3("sp2gno_burgers_large_s2048", SP2GNO_PARAMS["Burgers"][2])],
        "c": [g3("sp2gno_darcy_small_r141", SP2GNO_PARAMS["Darcy"][0]),
              g3("sp2gno_darcy_base_r141", SP2GNO_PARAMS["Darcy"][1]),
              g3("sp2gno_darcy_large_r141", SP2GNO_PARAMS["Darcy"][2])],
        "d": [g("sp2gno_burgers_base_r512"), g("sp2gno_burgers_base_r1024"), g("sp2gno_burgers_base_s2048"),
              g("sp2gno_burgers_base_s4096"), g("sp2gno_burgers_base_r8192")],
        "e": [g("sp2gno_darcy_base_r85"), g("sp2gno_darcy_base_r141"), g("sp2gno_darcy_base_r211"),
              g("sp2gno_darcy_base_r281"), g("sp2gno_darcy_base_r421")],
    }


if __name__ == "__main__":
    data = {"FNO": load_fno(), "DeepONet": load_deeponet(), "WNO": load_wno(), "Sp2GNO": load_sp2gno()}
    json.dump(data, open("/home/jetson/jjyoo3/edge_figures/fig2_full_data.json", "w"), indent=2)
    print("wrote fig2_full_data.json")
    for fam, panels in data.items():
        print(fam, "b:", panels["b"], "c:", panels["c"])
