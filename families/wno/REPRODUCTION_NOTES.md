# Reproduction Notes — WNO vs. Sp2GNO Variants & Resolution Scaling

Exact configuration, run grid, and design decisions behind the numbers in `RESULTS.md`
and `RESULTS_resolution_scaling.md`. See `README.md` for the quick-start; this file is the
precise reference.

## 1. Scripts & parameter budgets

Four training scripts, each with `--variant {small,base,large}` and a resolution control,
sharing the same data splits and the same per-sample relative-L2 metric:
- `train_wno_burgers.py`, `train_wno_darcy.py` — WNO (Wavelet Neural Operator)
- `train_sp2gno_burgers.py`, `train_sp2gno_darcy.py` — Sp2GNO

Parameter counts are matched to the FNO Table 6 budgets (~72k / 235k / 820k), verified by
instantiation:

| | small | base | large |
|---|---|---|---|
| Sp2GNO (both problems) | 70,658 | 234,371 | 814,370 |
| WNO Burgers (s2048) | 74,859 | 242,457 | 820,695 |
| WNO Darcy (r141) | 91,037 | 232,001 | 813,197 |

Variant knob = **width**. Sp2GNO: 6 layers, 64 graph-Fourier modes, width 13/24/45.
WNO Burgers: 4 layers, level 8, width 22/40/74. WNO Darcy: 4 layers, level 5, width 5/8/15.

## 2. Running

### 2a. Environment
Python 3.10 / CUDA 11.8 — see `requirements.txt` and the install block in `README.md`.
`torch-cluster` is required by Sp2GNO (`knn_graph`); the three wavelet packages
(`PyWavelets`, `ptwt`, `pytorch_wavelets`) are required by WNO.

### 2b. Data
Place the `.mat` files in `Jetson_data/` — see `Jetson_data/README.md`. Sp2GNO builds a
shared graph (knn + Laplacian eigenbasis) per resolution and caches it under `cache/` on
first use; the cache rebuilds automatically if absent.

### 2c. The 18-run grid (the `--ckpt` names are the deliverables)

**Model-scale (RESULTS.md):** Burgers @ s2048, Darcy @ r141.
```
python train_wno_burgers.py    --variant small --res 2048 --ckpt wno_burgers_small_r2048.pth
python train_wno_burgers.py    --variant base  --res 2048 --ckpt wno_burgers_base_r2048.pth
python train_wno_burgers.py    --variant large --res 2048 --ckpt wno_burgers_large_r2048.pth
python train_wno_darcy.py      --variant small --res 141  --ckpt wno_darcy_small_r141.pth
python train_wno_darcy.py      --variant base  --res 141  --ckpt wno_darcy_base_r141.pth
python train_wno_darcy.py      --variant large --res 141  --ckpt wno_darcy_large_r141.pth
python train_sp2gno_burgers.py --variant small --res 2048 --ckpt sp2gno_burgers_small_s2048.pth
python train_sp2gno_burgers.py --variant base  --res 2048 --ckpt sp2gno_burgers_base_s2048.pth
python train_sp2gno_burgers.py --variant large --res 2048 --ckpt sp2gno_burgers_large_s2048.pth
python train_sp2gno_darcy.py   --variant small --res 141  --batch_size 10 --ckpt sp2gno_darcy_small_r141.pth
python train_sp2gno_darcy.py   --variant base  --res 141  --batch_size 10 --ckpt sp2gno_darcy_base_r141.pth
python train_sp2gno_darcy.py   --variant large --res 141  --batch_size 8  --ckpt sp2gno_darcy_large_r141.pth
```

**Resolution scaling (RESULTS_resolution_scaling.md):** base variant, varying grid.
The base@2048 / base@141 runs above double as the low-resolution anchor — not retrained.
```
python train_wno_burgers.py    --variant base --res 4096 --ckpt wno_burgers_base_r4096.pth
python train_wno_burgers.py    --variant base --res 8192 --ckpt wno_burgers_base_r8192.pth
python train_wno_darcy.py      --variant base --res 281  --epochs 500 --ckpt wno_darcy_base_r281.pth
python train_wno_darcy.py      --variant base --res 421  --epochs 500 --ckpt wno_darcy_base_r421.pth
python train_sp2gno_burgers.py --variant base --res 4096 --ckpt sp2gno_burgers_base_s4096.pth
python train_sp2gno_darcy.py   --variant base --res 211  --batch_size 5 --ckpt sp2gno_darcy_base_r211.pth
```

Default `--epochs 1000`; the two largest WNO-Darcy grids (281, 421) used `--epochs 500`
(each ≈170 s/epoch). Runs are independent → one per GPU
(`CUDA_VISIBLE_DEVICES=$i python … &`), or use `run_burgers.sh` / `run_darcy.sh` for a
sequential launch.

### 2d. Collect results
```
python collect_results.py
```
Reads `runs/<tag>/result.json` and regenerates both `RESULTS*.md`. Per-run outputs:
`checkpoints/<tag>.pth`, `runs/<tag>/{result,curve}.json` + `run.log`, `logs/<tag>.log`.
(Prediction `plots/` are produced per run but were omitted from this package to keep it small.)

## 3. Design decisions baked in

- **Darcy resolution selection** uses evenly-spaced indices (`round(linspace(0,420,res))`)
  of the native 421-grid, so `--res 141/211/281/421` are true 141/211/281/421 grids.
- **Sp2GNO Darcy stops at 211×211.** The shared graph needs a dense `eigh` of an N×N
  Laplacian (N = grid²); memory grows as N². 211 → N≈44.5k fits an 80 GB GPU;
  421 → N≈177k does not. WNO Darcy has no eigendecomposition and scales to the full 421.
- **WNO wavelet level scales with resolution** (Darcy `{141:5, 211:6, 281:6, 421:7}`;
  Burgers `8+log2(res/2048)`) to hold the wavelet-mode count — and hence the parameter
  count — roughly constant across resolutions, for a fair resolution-scaling comparison.
- **WNO Darcy "small" is width-5** to honor the ~72k budget (2D WNO is parameter-dense:
  params ≈ width²·modes²·layers) — expected to be the weakest model by budget parity.
- **`--batch_size`** is reduced for the heaviest Sp2GNO-Darcy runs (large@141, base@211)
  to fit memory; it is purely a speed knob — the metric is per-sample relative-L2 and so
  is batch-size invariant.

## 4. Completeness check
A finished sweep has all 18 `checkpoints/*.pth`, a `final_test_rel_l2` in every
`runs/*/result.json`, and no "pending" cells in either `RESULTS*.md`.
