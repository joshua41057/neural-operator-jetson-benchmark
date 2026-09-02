#!/usr/bin/env python3
"""Fit and evaluate the kernel-class relation reported in the manuscript.

    log s = alpha + beta * phi + eps

phi is the share of profiled CUDA kernel time in bandwidth-sensitive classes
(movement/materialization plus dense GEMM), taken from the Jetson Nsight Systems
trace of the configuration. s is the ratio of Jetson to GH200 sustained median
latency for the same checkpoint and request path, both sustained class.

The acceptance criterion was fixed before fitting: R^2 >= 0.6 and held-out
predictions within a factor of two. The relation fails both, and the manuscript
reports it as a negative result. This script reproduces the reported numbers.

Run from the repository root:  python3 verification/kernel_class_fit.py
"""
import csv, math, os, pickle, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERN = re.compile(r'\s*(\d+\.\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s+[\d.]+\s+(.*)$')


def klass(n):
    """Group a kernel name into the taxonomy of Fig. 5. Order matters."""
    if 'elementwise_kernel_with_index' in n:                       return 'Other'
    if any(k in n for k in ('cudnn', 'convolve', 'conv_depthwise',
                            'dgrad', 'wgrad')):                    return 'Conv/Wavelet'
    if 'fft' in n.lower() or 'spRadix' in n:                       return 'FFT/Spectral'
    if any(k in n for k in ('sgemm', 'gemm', 'gemv', 'cutlass',
                            'splitKreduce', 'dot_kernel', 'trsm')): return 'Dense/GEMM'
    if any(k in n for k in ('CatArrayBatchedCopy', 'indexSelect', 'gpu_index_kernel',
                            'scatter_gather', 'index_elementwise_kernel', 'Gather',
                            'copy_', 'CopyKernel', 'permute', 'transpose')):
        return 'Movement/Materialization'
    if any(k in n for k in ('layer_norm', 'reduce_kernel', 'batch_norm',
                            'softmax', 'norm_kernel')):            return 'Reduction/Norm'
    return 'Activation/Elementwise'


def phi(stats_path):
    """Bandwidth-sensitive share of profiled CUDA kernel time."""
    txt = open(os.path.join(ROOT, stats_path), errors='replace').read()
    i = txt.find('cuda_gpu_kern_sum):')
    seg = txt[i:]
    j = seg.find('\nProcessing [')
    if j > 0:
        seg = seg[:j]
    tot = bw = 0
    for line in seg.splitlines():
        m = KERN.match(line)
        if not m:
            continue
        ns, name = int(m.group(2)), m.group(4).strip()
        tot += ns
        if klass(name) in ('Movement/Materialization', 'Dense/GEMM'):
            bw += ns
    return bw / tot


# (label, gh200 case, agg_all key group, agg_all key, nsys stats path)
FIT = [
 ("FNO Burgers r2048",        "fno_burgers_base_r2048",  "fno_s", "burgers_base_r2048_fp32_strict",
  "families/fno/results/jetson_fno_profile_nsys_extended/burgers_r2048_ts_fp32_nsys_stats.txt"),
 ("FNO Darcy 141",            "fno_darcy_base_r141",     "fno_s", "darcy_base_r141_fp32_strict",
  "families/fno/results/jetson_fno_profile_nsys_extended/darcy_r141_ts_fp32_nsys_stats.txt"),
 ("DeepONet Burgers r2048",   "deeponet_burgers_base_r2048", "don_s", "burgers_base_fp32_strict",
  "families/deeponet/results/profiles/deeponet_nsys/burgers_base_r2048_ts_fp32_nsys_stats.txt"),
 ("DeepONet Darcy 141",       "deeponet_darcy_base_r141", "don_s", "darcy_r141_fp32_strict",
  "families/deeponet/results/profiles/deeponet_nsys/darcy_base_r141_ts_fp32_nsys_stats.txt"),
 ("DeepONet Darcy 141 large", "deeponet_darcy_large_r141", "don_s", "darcy_large_r141_fp32_strict",
  "families/deeponet/results/profiles/deeponet_nsys/darcy_large_r141_ts_fp32_nsys_stats.txt"),
 ("WNO Burgers r2048",        "wno_burgers_base_r2048",  "wno_s", "wno_burgers_base_r2048_fp32_strict",
  "families/wno/results/profiles/wno_nsys/wno_burgers_base_r2048_nsys_stats.txt"),
 ("WNO Darcy 141",            "wno_darcy_base_r141",     "wno_s", "wno_darcy_base_r141_fp32_strict",
  "families/wno/results/profiles/wno_nsys/wno_darcy_base_r141_nsys_stats.txt"),
 ("WNO Darcy 141 large",      "wno_darcy_large_r141",    "wno_s", "wno_darcy_large_r141_fp32_strict",
  "families/wno/results/profiles/wno_nsys/wno_darcy_large_r141_nsys_stats.txt"),
 ("Sp2GNO Burgers small",     "sp2gno_burgers_small_r2048", "sp_s", "sp2gno_burgers_small_s2048_fp32_strict",
  "families/sp2gno/results/profiles/sp2gno_nsys/sp2gno_burgers_small_s2048_nsys_stats.txt"),
 ("Sp2GNO Burgers base",      "sp2gno_burgers_base_r2048",  "sp_s", "sp2gno_burgers_base_s2048_fp32_strict",
  "families/sp2gno/results/profiles/sp2gno_nsys/sp2gno_burgers_base_s2048_nsys_stats.txt"),
 ("Sp2GNO Burgers large",     "sp2gno_burgers_large_r2048", "sp_s", "sp2gno_burgers_large_s2048_fp32_strict",
  "families/sp2gno/results/profiles/sp2gno_nsys/sp2gno_burgers_large_s2048_nsys_stats.txt"),
 ("Sp2GNO Darcy 141 small",   "sp2gno_darcy_small_r141", "sp_s", "sp2gno_darcy_small_r141_fp32_strict",
  "families/sp2gno/results/profiles/sp2gno_nsys/sp2gno_darcy_small_r141_nsys_stats.txt"),
 ("Sp2GNO Darcy 141 base",    "sp2gno_darcy_base_r141",  "sp_s", "sp2gno_darcy_base_r141_fp32_strict",
  "families/sp2gno/results/profiles/sp2gno_nsys/sp2gno_darcy_base_r141_nsys_stats.txt"),
 ("Sp2GNO Darcy 141 large",   "sp2gno_darcy_large_r141", "sp_s", "sp2gno_darcy_large_r141_fp32_strict",
  "families/sp2gno/results/profiles/sp2gno_nsys/sp2gno_darcy_large_r141_nsys_stats.txt"),
]
HELD = [
 ("Heat exch. full",      "hx_full",     "heat_s", "full_fp32",
  "families/heat_exchanger/inference_runs/virso_nsys_fixed/full_fp32_nsys_stats.txt"),
 ("Heat exch. spectral",  "hx_spectral", "heat_s", "spectral_fp32",
  "families/heat_exchanger/inference_runs/virso_nsys_fixed/spectral_fp32_nsys_stats.txt"),
 ("Heat exch. 2-layer",   "hx_layer2",   "heat_s", "layer2_fp32",
  "families/heat_exchanger/inference_runs/virso_nsys_fixed/layer2_fp32_nsys_stats.txt"),
]


def build(rows, agg, gh):
    out = []
    for label, gcase, grp, key, stats in rows:
        out.append(dict(label=label, phi=phi(stats),
                        jet=agg[grp][key]['med'], gh=gh[gcase],
                        s=agg[grp][key]['med'] / gh[gcase]))
    return out


def fit(pts):
    x = np.array([p['phi'] for p in pts]); y = np.log([p['s'] for p in pts])
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a + b*x))**2).sum() / ((y - y.mean())**2).sum()
    return a, b, r2, x, y


def main():
    agg = pickle.load(open(os.path.join(ROOT, 'verification/agg_all.pkl'), 'rb'))
    gh = {r['case']: float(r['med_ms_mean']) for r in csv.DictReader(
        open(os.path.join(ROOT, 'gh200/results/gh200_fp32_20260715/gh200_fp32_summary.csv')))}
    F, H = build(FIT, agg, gh), build(HELD, agg, gh)

    print(f"{'configuration':28s} {'phi':>6s} {'Jetson ms':>10s} {'GH200 ms':>9s} {'s':>7s}")
    for p in F + H:
        print(f"{p['label']:28s} {p['phi']*100:6.1f} {p['jet']:10.3f} {p['gh']:9.3f} {p['s']:7.3f}")

    a, b, r2, x, y = fit(F)
    rng = np.random.default_rng(0); bs = []
    for _ in range(10000):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\npooled fit   n={len(F)}  alpha={a:.3f}  beta={b:.3f} [{lo:.3f}, {hi:.3f}]  R2={r2:.3f}")

    worst = max((max(math.exp(a + b*p['phi'])/p['s'], p['s']/math.exp(a + b*p['phi'])), p['label'])
                for p in H)
    print(f"held out     worst ratio-residual {worst[0]:.2f}x  ({worst[1]})")
    for name, sub in (("Darcy only", [p for p in F if 'Darcy' in p['label']]),
                      ("Burgers only", [p for p in F if 'Burgers' in p['label']])):
        print(f"{name:12s} n={len(sub):2d}  R2={fit(sub)[2]:.3f}")

    X = np.column_stack([np.ones(len(F)), x, np.log([p['jet'] for p in F])])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"{'+log latency':12s} n={len(F)}  R2="
          f"{1 - ((y - X@c)**2).sum()/((y - y.mean())**2).sum():.3f}   "
          "(diagnosis of the omitted variable, not a replacement relation)")
    print("\nacceptance criterion R2 >= 0.6 and held-out within 2x: NOT MET")

    import json
    out = {
        "n": len(F), "alpha": a, "beta": b, "beta_ci95_percentile": [float(lo), float(hi)],
        "bootstrap_resamples": 10000, "bootstrap_method": "percentile", "r2": r2,
        "subgroup_r2": {"darcy_only": fit([p for p in F if 'Darcy' in p['label']])[2],
                         "burgers_only": fit([p for p in F if 'Burgers' in p['label']])[2]},
        "r2_with_log_latency": 1 - ((y - X@c)**2).sum()/((y - y.mean())**2).sum(),
        "acceptance": {"r2_min": 0.6, "holdout_max_factor": 2.0, "met": False},
        "held_out": [{"label": p['label'], "phi": p['phi'], "s_measured": p['s'],
                       "s_predicted": math.exp(a + b*p['phi'])} for p in H],
        "points": [{"label": p['label'], "phi": p['phi'], "jetson_ms": p['jet'],
                     "gh200_ms": p['gh'], "s": p['s']} for p in F],
    }
    dest = os.path.join(ROOT, 'verification', 'kernel_class_fit.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
