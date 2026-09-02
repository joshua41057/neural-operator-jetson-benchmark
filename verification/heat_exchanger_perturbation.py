#!/usr/bin/env python3
"""Heat-exchanger reduced-precision perturbation, same estimator as Sp2GNO.

    per_sample = ||y_p - y_fp32||_2 / (||y_fp32||_2 + 1e-12)
    Delta_p    = mean(per_sample)

Reference is the FP32 prediction of the same architectural variant.
"""
import numpy as np, glob, os, sys, itertools, statistics as st

D = sys.argv[1]
VAR = ['full', 'spectral', 'layer2']
PREC = ['tf32', 'bf16', 'fp16_autocast']

def load(v, p, r):
    f = f'{D}/perturb_{v}_{p}_r{r}/outputs/virso_predictions.npy'
    return np.load(f) if os.path.exists(f) else None

print(f"{'variant':10s} {'precision':16s} {'rep1':>10s} {'rep2':>10s} {'rep3':>10s} {'mean':>10s}  refspread")
for v in VAR:
    refs = [load(v, 'fp32', r) for r in (1, 2, 3)]
    refs = [x for x in refs if x is not None]
    ref = refs[0]
    # how much do the three fp32 runs differ from each other?
    spread = max(float(np.mean(np.linalg.norm((a-ref).reshape(a.shape[0],-1),axis=1) /
                               (np.linalg.norm(ref.reshape(ref.shape[0],-1),axis=1)+1e-12)))
                 for a in refs[1:]) if len(refs) > 1 else 0.0
    for p in PREC:
        vals = []
        for r in (1, 2, 3):
            y = load(v, p, r)
            if y is None or y.shape != ref.shape:
                vals.append(float('nan')); continue
            per = np.linalg.norm((y-ref).reshape(y.shape[0],-1), axis=1) / \
                  (np.linalg.norm(ref.reshape(ref.shape[0],-1), axis=1) + 1e-12)
            vals.append(float(per.mean()))
        good = [x for x in vals if x == x]
        m = st.mean(good) if good else float('nan')
        print(f"{v:10s} {p:16s} " + " ".join(f"{x:10.6f}" for x in vals) + f" {m:10.6f}  {spread:.2e}")
