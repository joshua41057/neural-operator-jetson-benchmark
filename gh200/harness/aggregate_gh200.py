#!/usr/bin/env python3
"""Aggregate GH200 FP32 benchmark results from all five families into one table.

Reads the per-family result files under <results_root>/<run_tag>/ (each family
writes a slightly different schema — see load_* below), extracts the four paper
metrics, averages across reps, and writes:
  <run_tag>/gh200_fp32_summary.csv   (per case: mean/std over reps)
  <run_tag>/gh200_fp32_raw.csv       (every rep)
and prints a Med ms / Avg W / J/inf / GPU MB table.

Metric definitions (uniform across families):
  Med ms  = median / p50 latency (ms)
  Avg W   = mean GPU power (nvidia-smi power.draw) over the timed window
  J/inf   = energy per inference (avg_power / throughput, i.e. avg_power * mean latency)
  GPU MB  = torch.cuda.max_memory_allocated (primary)
  GPU MB (residency) = nvidia-smi memory.used peak (secondary; includes CUDA context)

Usage: python aggregate_gh200.py --run-tag gh200_fp32_YYYYMMDD [--results-root ...]
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as stats


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def _first(d, *keys):
    for k in keys:
        if k in d and d[k] not in (None, "", "nan"):
            v = _f(d[k])
            if v is not None:
                return v
    return None


def _rec(family, case, rep, med, avgw, jinf, gpu_torch, gpu_res, extra=None):
    r = {
        "family": family, "case": case, "rep": rep,
        "med_ms": med, "avg_w": avgw, "j_per_inf": jinf,
        "gpu_mb_torch": gpu_torch, "gpu_mb_residency": gpu_res,
    }
    if extra:
        r.update(extra)
    return r


def _repnum(tag):
    import re
    m = re.search(r"rep(\d+)", tag)
    return int(m.group(1)) if m else 1


def load_json_family(root, family, subdir):
    """FNO / DeepONet / HX-style JSON results."""
    recs = []
    base = os.path.join(root, subdir)
    for path in glob.glob(os.path.join(base, "**", "*.json"), recursive=True):
        name = os.path.basename(path)
        if name.startswith("_") or "nvsmi" in name:
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("status") not in (None, "success"):
            continue
        med = _first(d, "median_ms", "p50_latency_ms")
        if med is None:
            continue
        avgw = _first(d, "avg_power_w", "vdd_in_mean_w", "gpu_power_mean_w")
        jinf = _first(d, "energy_per_inference_j", "energy_j_per_inference")
        gpu_torch = _first(d, "peak_cuda_allocated_mb", "peak_cuda_alloc_mb", "cuda_peak_allocated_mb", "cuda_peak_alloc_mb")
        gpu_res = _first(d, "gpu_mem_used_peak_mb", "board_ram_peak_mb")
        # case name: strip _rep<n> / trailing tags
        case = d.get("variant") or d.get("case_id") or name.replace(".json", "")
        import re
        case = re.sub(r"_rep\d+$", "", str(case))
        if family == "hx":
            case = f"hx_{d.get('variant', case)}"
        recs.append(_rec(family, case, _repnum(path), med, avgw, jinf, gpu_torch, gpu_res))
    return recs


def load_wno(root):
    recs = []
    base = os.path.join(root, "wno")
    for path in glob.glob(os.path.join(base, "**", "result.json"), recursive=True):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if d.get("status") != "success":
            continue
        cid = str(d.get("case_id", ""))
        import re
        case = re.sub(r"_rep\d+$", "", cid)
        recs.append(_rec("wno", case, _repnum(cid),
                         _first(d, "p50_latency_ms"),
                         _first(d, "vdd_in_mean_w", "gpu_power_mean_w"),
                         _first(d, "energy_j_per_inference"),
                         _first(d, "cuda_peak_allocated_mb"),
                         _first(d, "board_ram_peak_mb", "gpu_mem_used_peak_mb")))
    return recs


def load_sp2gno(root):
    recs = []
    base = os.path.join(root, "sp2gno")
    for path in glob.glob(os.path.join(base, "**", "reports", "sp2gno_edge_summary_*.csv"), recursive=True):
        if path.endswith("_FAILED.csv"):
            continue
        try:
            d = next(csv.DictReader(open(path, newline="")))
        except Exception:
            continue
        if d.get("status") != "success":
            continue
        case = d.get("case_id") or d.get("run_name", "")
        import re
        case = re.sub(r"_(fp32_strict|tf32).*$", "", str(case))
        rep = _f(d.get("rep")) or _repnum(d.get("run_name", ""))
        recs.append(_rec("sp2gno", case, int(rep),
                         _first(d, "p50_latency_ms"),
                         _first(d, "vdd_in_mean_w", "gpu_power_mean_w"),
                         _first(d, "energy_j_per_inference"),
                         _first(d, "cuda_peak_allocated_mb"),
                         _first(d, "board_ram_peak_mb", "gpu_mem_used_peak_mb")))
    return recs


CASE_ORDER = ["burgers_small", "burgers_base", "burgers_large",
              "darcy_small", "darcy_base", "darcy_large",
              "full", "spectral", "layer2"]


def sort_key(case):
    for i, tok in enumerate(CASE_ORDER):
        if tok in case:
            return i
    return 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="<REPO>/results")
    ap.add_argument("--run-tag", required=True)
    args = ap.parse_args()

    root = os.path.join(args.results_root, args.run_tag)
    recs = []
    recs += load_json_family(root, "fno", "fno")
    recs += load_json_family(root, "deeponet", "deeponet")
    recs += load_json_family(root, "hx", "hx")
    recs += load_wno(root)
    recs += load_sp2gno(root)

    if not recs:
        print(f"[!] no results under {root}")
        return

    raw_csv = os.path.join(root, "gh200_fp32_raw.csv")
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    # group by (family, case)
    groups = {}
    for r in recs:
        groups.setdefault((r["family"], r["case"]), []).append(r)

    METR = ["med_ms", "avg_w", "j_per_inf", "gpu_mb_torch", "gpu_mb_residency"]
    summ = []
    for (fam, case), rs in groups.items():
        row = {"family": fam, "case": case, "n_reps": len(rs)}
        for m in METR:
            vals = [r[m] for r in rs if r[m] is not None]
            row[m + "_mean"] = stats.mean(vals) if vals else None
            row[m + "_std"] = (stats.pstdev(vals) if len(vals) > 1 else 0.0) if vals else None
        summ.append(row)

    summ.sort(key=lambda r: (["fno", "deeponet", "wno", "sp2gno", "hx"].index(r["family"])
                             if r["family"] in ["fno", "deeponet", "wno", "sp2gno", "hx"] else 9,
                             sort_key(r["case"])))

    sum_csv = os.path.join(root, "gh200_fp32_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    def fmt(x, p=3):
        return f"{x:.{p}f}" if isinstance(x, float) else ("-" if x is None else str(x))

    print(f"\n{'Family':9s} {'Case':28s} {'n':>2s} {'Med.ms':>9s} {'Avg.W':>8s} {'J/inf':>9s} {'GPU MB':>9s} {'(resid)':>9s}")
    print("-" * 92)
    for r in summ:
        print(f"{r['family']:9s} {r['case'][:28]:28s} {r['n_reps']:>2d} "
              f"{fmt(r['med_ms_mean']):>9s} {fmt(r['avg_w_mean'],1):>8s} "
              f"{fmt(r['j_per_inf_mean'],4):>9s} {fmt(r['gpu_mb_torch_mean'],2):>9s} "
              f"{fmt(r['gpu_mb_residency_mean'],1):>9s}")
    print(f"\n[raw] {raw_csv}\n[summary] {sum_csv}")


if __name__ == "__main__":
    main()
