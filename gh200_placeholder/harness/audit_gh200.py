#!/usr/bin/env python3
"""Independently re-derive every GH200 table value from the raw NVML logs.

Unlike aggregate_gh200.py (which trusts the power summaries stored in each record),
this walks the per-rep native files, re-parses the 200 ms NVML log that backs each
one, and checks every link in the chain: log -> record -> raw.csv -> summary.csv.
Exit status is non-zero if any ERROR-level check fails.

    python audit_gh200.py --results-root results --run-tag gh200_fp32_20260715

NVML logs are located by filesystem layout (not by the path fields inside the
records, which are normalized to placeholders in the public deposit).
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

issues = []
def note(sev, msg): issues.append((sev, msg))


def parse_nvml(path):
    if not path or not os.path.exists(path):
        return None
    ts, pw = [], []
    with open(path, errors="ignore") as f:
        for ln in f:
            p = ln.split(",")
            if len(p) < 5:
                continue
            try:
                pw.append(float(p[1]))
            except ValueError:
                continue
            m = re.match(r"\s*\d{4}/\d\d/\d\d (\d\d):(\d\d):(\d\d)\.(\d+)", p[0])
            if m:
                ts.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000)
    if not pw:
        return None
    iv = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    return dict(n=len(pw), mean=st.mean(pw),
                span=(ts[-1] - ts[0]) if len(ts) > 1 else None,
                med_int_ms=(st.median(iv) * 1000 if iv else None))


def collect(root):
    recs = []

    def add(**k): recs.append(k)

    # FNO / DeepONet: <base>.json + sibling <base>{_nvsmi,_tegrastats_raw}.log
    for fam, suf in [("fno", "_nvsmi.log"), ("deeponet", "_tegrastats_raw.log")]:
        for jp in sorted(glob.glob(f"{root}/{fam}/{fam}_*_rep*.json")):
            d = json.load(open(jp))
            if d.get("status") not in (None, "success"):
                note("ERR", f"{jp} status={d.get('status')}")
            case = re.sub(r"_rep\d+$", "", os.path.basename(jp)[:-5])
            add(family=fam, case=case, rep=int(re.search(r"rep(\d+)", jp)[1]),
                med=d.get("median_ms", d.get("p50_latency_ms")),
                avgw=d.get("avg_power_w", d.get("gpu_power_mean_w")),
                jinf=d.get("energy_per_inference_j", d.get("energy_j_per_inference")),
                cuda=d.get("peak_cuda_allocated_mb", d.get("cuda_peak_allocated_mb")),
                thr=d.get("throughput_inf_s"),
                tf32=d.get("precision_info", {}).get("allow_tf32_matmul"),
                nvml=jp[:-5] + suf)

    # WNO: <dir>/result.json + sibling tegrastats.log
    for jp in sorted(glob.glob(f"{root}/wno/**/result.json", recursive=True)):
        d = json.load(open(jp))
        if d.get("status") != "success":
            note("ERR", f"{jp} status={d.get('status')}")
        cid = d["case_id"]
        add(family="wno", case=re.sub(r"_rep\d+$", "", cid), rep=int(re.search(r"rep(\d+)", cid)[1]),
            med=d.get("p50_latency_ms"), avgw=d.get("gpu_power_mean_w", d.get("vdd_in_mean_w")),
            jinf=d.get("energy_j_per_inference"), cuda=d.get("cuda_peak_allocated_mb"),
            thr=d.get("throughput_inf_s"),
            tf32=d.get("precision_info", {}).get("allow_tf32_matmul"),
            nvml=os.path.join(os.path.dirname(jp), "tegrastats.log"))

    # Sp2GNO: <dir>/reports/sp2gno_edge_summary_*.csv + <dir>/logs/tegrastats_*.log
    for cp in sorted(glob.glob(f"{root}/sp2gno/**/reports/sp2gno_edge_summary_*.csv", recursive=True)):
        d = next(csv.DictReader(open(cp)))
        if d.get("status") != "success":
            note("ERR", f"{cp} status={d.get('status')}")
        logs = glob.glob(f"{os.path.dirname(os.path.dirname(cp))}/logs/tegrastats_*.log")
        add(family="sp2gno", case=d["case_id"], rep=int(d["rep"]),
            med=float(d["p50_latency_ms"]), avgw=float(d["gpu_power_mean_w"]),
            jinf=float(d["energy_j_per_inference"]), cuda=float(d["cuda_peak_allocated_mb"]),
            thr=float(d["throughput_inf_s"]),
            tf32=(d.get("allow_tf32_matmul", "").lower() == "true"),
            nvml=logs[0] if logs else None)

    # HX: <dir>/virso_gh200_*.json + <dir>/logs/tegrastats_*.log
    for jp in sorted(glob.glob(f"{root}/hx/**/virso_gh200_*.json", recursive=True)):
        d = json.load(open(jp))
        if d.get("status") != "success":
            note("ERR", f"{jp} status={d.get('status')}")
        logs = glob.glob(f"{os.path.dirname(jp)}/logs/tegrastats_*.log")
        add(family="hx", case=f"hx_{d['variant']}", rep=int(re.search(r"_rep(\d+)", jp)[1]),
            med=d["p50_latency_ms"], avgw=d.get("avg_power_w", d.get("gpu_power_mean_w")),
            jinf=d["energy_j_per_inference"],
            cuda=d.get("cuda_peak_allocated_mb", d.get("cuda_peak_alloc_mb")),
            thr=None, tf32=False, nvml=logs[0] if logs else None)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--run-tag", default="gh200_fp32_20260715")
    args = ap.parse_args()
    root = os.path.join(args.results_root, args.run_tag)

    recs = collect(root)
    print(f"[collected] {len(recs)} rep-records from {root}")

    by = defaultdict(list)
    for r in recs:
        by[(r["family"], r["case"])].append(r)
    for k, v in sorted(by.items()):
        if len(v) != 3:
            note("WARN", f"{k} has {len(v)} reps: {sorted(x['rep'] for x in v)}")

    nvml_bad = 0
    for r in recs:
        pm = parse_nvml(r["nvml"])
        if pm is None:
            note("ERR", f"NVML missing: {r['family']}/{r['case']} rep{r['rep']}")
            continue
        if abs(pm["mean"] - r["avgw"]) > 0.05:
            nvml_bad += 1
            note("ERR", f"avgW {r['family']}/{r['case']} rep{r['rep']}: log={pm['mean']:.3f} stored={r['avgw']:.3f}")
        if pm["med_int_ms"] and not (170 <= pm["med_int_ms"] <= 230):
            note("WARN", f"interval {r['family']}/{r['case']} rep{r['rep']}: {pm['med_int_ms']:.0f}ms")
        if pm["span"] and not (110 <= pm["span"] <= 135):
            note("WARN", f"window {r['family']}/{r['case']} rep{r['rep']}: {pm['span']:.1f}s")
        if r["thr"] and abs(r["avgw"] / r["thr"] - r["jinf"]) / r["jinf"] > 0.01:
            note("ERR", f"energy!=P/thr {r['family']}/{r['case']} rep{r['rep']}")
        if r.get("tf32") is True:
            note("ERR", f"TF32 enabled {r['family']}/{r['case']} rep{r['rep']}")
    print(f"[nvml] re-parsed {len(recs)} logs, avgW mismatches={nvml_bad}")

    def close(a, b, tol=1e-6):
        return a is not None and b not in (None, "") and abs(float(a) - float(b)) <= tol

    raw = list(csv.DictReader(open(f"{root}/gh200_fp32_raw.csv")))
    rawmiss = sum(
        0 if any(close(x["med_ms"], r["med"]) and close(x["avg_w"], r["avgw"])
                 and close(x["j_per_inf"], r["jinf"]) and close(x["gpu_mb_torch"], r["cuda"])
                 for x in raw if x["family"] == r["family"] and x["case"] == r["case"])
        else note("ERR", f"raw.csv no match {r['family']}/{r['case']} rep{r['rep']}") or 1
        for r in recs)
    print(f"[raw.csv] {len(recs)} records, unmatched={rawmiss}")

    summ = {(s["family"], s["case"]): s for s in csv.DictReader(open(f"{root}/gh200_fp32_summary.csv"))}
    sumbad = 0
    for k, v in by.items():
        for metric, key in [("med_ms_mean", "med"), ("avg_w_mean", "avgw"),
                            ("j_per_inf_mean", "jinf"), ("gpu_mb_torch_mean", "cuda")]:
            if abs(st.mean([x[key] for x in v]) - float(summ[k][metric])) > 1e-6:
                sumbad += 1
                note("ERR", f"summary {k} {metric}")
    print(f"[summary.csv] recomputed means, mismatches={sumbad}")

    errs = [m for s, m in issues if s == "ERR"]
    warns = [m for s, m in issues if s == "WARN"]
    print("\n=== ISSUES ===")
    if not issues:
        print("NONE — all checks passed")
    else:
        for s, m in issues:
            print(f"[{s}] {m}")
    print(f"\n{len(errs)} ERR, {len(warns)} WARN")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
