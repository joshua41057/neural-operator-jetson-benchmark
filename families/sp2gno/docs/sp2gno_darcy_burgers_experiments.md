# Sp2GNO on Darcy (2D) and Burgers (1D) — Experiment Log

**Date:** 2026-06-13
**Author:** subhankar
**Environment:** conda `BGCN_torch` (torch 2.0.1+cu117, PyG 2.4.0), GPU 0 = A100 80GB.

## Why these experiments
Benchmark the **batched** Sp2GNO architecture (the model + utilities from
`train_sp2gno_elasticity.py`) on two new PDE operator-learning datasets, reusing
exactly that batched model rather than the original unbatched implementation
(https://github.com/csccm-iitd/Sp2GNO). Both datasets use a single **shared grid
graph** for every sample (unlike elasticity's per-mesh graphs), so the knn graph,
inverse-distance edge weights, Lipschitz embeddings and Laplacian eigenpairs are
built once and cached.

Code:
- `sp2gno_core.py` — batched model (`MLP`, `GraphFourierLayer`, `FrigateConv`,
  `Sp2GNO`) + graph-feature preprocessing, all ported verbatim from
  `train_sp2gno_elasticity.py`; adds a shared-graph batching path (`SharedGraph`)
  and a generic trainer (`train_shared`) with best-on-validation checkpointing and
  resume.
- `train_darcy.py`, `train_sp2gno_burgers.py` — per-dataset drivers.

> **Note on the Burgers framing.** The user originally asked for two Burgers
> scripts (1D time-marching + 2D space-time). `burgers_data_R10.mat` is a *static*
> map `a(x) -> u(x,T=1)` with **no intermediate time axis** (confirmed: `a_x` has
> 8191 points = a spatial finite difference of the single 8192-long axis). Both
> time-resolved framings are therefore unsupported by this file, and the user
> chose to keep a **single 1D static-map script**.

## Common model / training protocol
- Model: `Sp2GNO`, width 48, 6 layers (`GraphFourierLayer ∥ FrigateConv`),
  `num_freq = 64` low Laplacian frequencies, `out_dim = 1`, ≈ 0.93 M params.
- Node features: **raw coordinates + the (z-scored) input field only** — no NeRF
  positional encoding (per user instruction; positional information is carried by
  the graph-Laplacian eigenvectors in the spectral layers).
- Loss: per-sample **relative L2** (`rel_l2`), mean over batch — scale-invariant,
  so targets are left unnormalized.
- Optimizer: Adam, lr 1e-3, weight decay 1e-4; `StepLR(step_size=100, gamma=0.65)`.
- Eval every 5 epochs on val + test; **best checkpoint selected on validation
  rel-L2**; a `last` checkpoint (model+opt+sched+epoch+best+curve) is written every
  eval for `--resume`. Plots saved every 50 epochs and for the final best model.
- Epochs: **1000** each. Seed 0. Device cuda:0.

## 1. Darcy 2D — `train_darcy.py`
- **Data:** `piececonst_r421_N1024_smooth1.mat` (train) /
  `piececonst_r421_N1024_smooth2.mat` (test). Input `coeff` a(x) ∈ {3,12},
  output `sol` u(x).
- **Resolution:** subsample r=5 → s=85, N = s² = **7225** grid nodes.
- **Splits:** train = smooth1[:900], val = smooth1[900:1000] (best-ckpt
  selection), test = smooth2[:200] (canonical Darcy test set).
- **Features (in_dim=3):** [x, y, coeff_norm]; coeff z-scored with train mean/std
  (mean 7.5675, std 4.4995).
- **Graph:** knn k=20 + to_undirected, min-max inverse-distance edge weights,
  16-dim Lipschitz embeddings, eigh of the 7225×7225 sym-normalized Laplacian
  (lowest 64). Cached → `cache/darcy_s85_k20_f64.pt`.
- **Batch size:** 10.

## 2. Burgers 1D — `train_sp2gno_burgers.py`
- **Data:** `burgers_data_R10.mat`, static map a(x) → u(x,T=1), periodic 1D domain.
- **Resolution:** subsample sub=8 → s = **1024** nodes.
- **Splits:** from `Jetson_data/burgers_split.json` — train 1638 / val 205 /
  test 205.
- **Features (in_dim=2):** [x, a_norm]; a z-scored with train mean/std
  (mean ≈ 0, std 0.5918).
- **Graph:** knn k=8 over the 1D x-coordinate + to_undirected, inverse-distance
  edge weights, Lipschitz embeddings, eigh of the 1024×1024 Laplacian (lowest 64).
  Cached → `cache/burgers_s1024_k8_f64.pt`.
- **Batch size:** 20.

## Results
Metric = relative L2 (lower is better). Runs launched 2026-06-13 18:23 in tmux
sessions `darcy_sp2gno` and `burgers_sp2gno`.

| Dataset | best val rel-L2 | final test rel-L2 (best-on-val model) | status |
|---------|-----------------|----------------------------------------|--------|
| Darcy 2D  | **0.007844** (ep 975) | **0.008447** (≈0.84%) | ✅ done (1000 ep, ~2.5 h / 9172 s) |
| Burgers 1D | **0.004301** (ep 990) | **0.004324** (≈0.43%) | ✅ done (1000 ep, ~51 min / 3091 s) |

**Key findings:** Both converge cleanly with the batched Sp2GNO and coords+field
inputs only (no NeRF positional encoding). Burgers 1D reaches ≈0.43% rel-L2;
Darcy 2D reaches ≈0.84% rel-L2 (best test seen during training 0.00837 @ ep ~975).
Darcy train rel-L2 (0.0052 @ ep1000) sits a touch below val/test (~0.008), i.e.
mild, benign overfitting — the best-on-val checkpoint (ep 975) is the one reported.

**Pre-run convergence checks (sanity):**
- Burgers: val 0.0652 @ ep5 → 0.0610 @ ep10 (full split), decreasing steadily.
- Darcy (200-sample smoke, 30 ep): val 1.26 @ ep5 → 0.15 @ ep30 — learns cleanly.

(Final numbers + per-seed curve to be filled from `runs/*/result.json` and
`runs/*/curve.json` once the 1000-epoch runs finish.)

## Artefacts
- Darcy: `runs/darcy/` — `run.log` (training log), `curve.json` (train/val/test
  rel-L2 per eval), `result.json` (final test + best val), `ckpt/darcy_best.pth`,
  `ckpt/darcy_last.pth`, `plots/darcy_pred_*.png` (Input / GT / Pred / Sq-Error,
  4 test samples).
- Burgers: `runs/burgers/` — same layout, `ckpt/burgers_{best,last}.pth`,
  `plots/burgers_pred_*.png` (GT vs Pred overlay + squared error, 4 test samples).
- Graph caches: `cache/darcy_s85_k20_f64.pt`, `cache/burgers_s1024_k8_f64.pt`.
- tmux tee logs: `logs/tmux_darcy.out`, `logs/tmux_burgers.out` (note: `conda run`
  buffers stdout — `runs/*/run.log` is the live source of truth).

## How to resume
```bash
conda activate BGCN_torch
python train_darcy.py --resume                 # picks up runs/darcy/ckpt/darcy_last.pth
python train_sp2gno_burgers.py --resume         # picks up runs/burgers/ckpt/burgers_last.pth
```
