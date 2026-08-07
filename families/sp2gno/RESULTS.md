# Sp2GNO Benchmark Results

Batched Sp2GNO (width 48, 6 layers, 64 graph-Fourier modes, ≈0.93 M params),
trained 1000 epochs with per-sample relative-L2 loss, Adam lr 1e-3 / StepLR(100, 0.65).
Node features = coordinates + input field only (no positional encoding); a single
shared grid graph per dataset. Best checkpoint selected on validation rel-L2.

| Dataset | resolution | split (train/val/test) | best val rel-L2 | **final test rel-L2** | wall time |
|---|---|---|---|---|---|
| **Darcy 2D** | s=85, N=7225 | 900 / 100 / 200 | 0.00784 (ep 975) | **0.00845 (≈0.84%)** | ~2.5 h |
| **Burgers 1D** | s=1024 | 1638 / 205 / 205 | 0.00430 (ep 990) | **0.00432 (≈0.43%)** | ~51 min |

- **Darcy** — `piececonst_r421_N1024_smooth1/2.mat`, map a(x)→u(x); train from smooth1, test on smooth2.
- **Burgers** — `burgers_data_R10.mat`, static 1D map a(x)→u(x,T=1); split from `burgers_split.json`.

Both converged cleanly. Darcy shows mild benign overfitting (train 0.0052 vs test ~0.008), so the
reported value is the best-on-validation checkpoint (ep 975).

**Artifacts:** `runs/darcy/` and `runs/burgers/` — `result.json`, `curve.json`, `run.log`,
`ckpt/*_best.pth`, `ckpt/*_last.pth`, `plots/*_pred_best.png` (4-sample Prediction/GT/Error).
Full setup: `docs/sp2gno_darcy_burgers_experiments.md`.
