# WNO vs. Sp2GNO — Model-Scale & Resolution-Scaling Benchmarks

Companion code, trained checkpoints, and results for the **WNO (Wavelet Neural Operator)**
vs. **Sp2GNO** comparison on the 1D **Burgers** and 2D **Darcy** benchmarks, at three
parameter-matched model scales (**small / base / large**) and across grid resolutions.

Parameter budgets are matched to the FNO **Small / Base / Large** counts from **Table 6**
of the Jetson edge paper (~72k / 235k / 820k) — see `Jetson_Scifm_Edge_Arxiv.pdf`. Both
operators use the **same data splits** and the **same per-sample relative-L2** metric, so
every number is directly comparable.

> **Note:** the raw training/testing data (`*.mat`, ~4.1 GB) is **not** included in this
> package. See [`Jetson_data/README.md`](Jetson_data/README.md) for the exact files and
> where they go. Everything else needed to read the results or re-train is here.

---

## Results at a glance

Full tables (parameter counts + best-val + final-test rel-L2) are in
[`RESULTS.md`](RESULTS.md) (model-scale) and
[`RESULTS_resolution_scaling.md`](RESULTS_resolution_scaling.md) (resolution scaling).

**Model-scale, final test rel-L2:**

| | small | base | large |
|---|---|---|---|
| WNO Burgers (s2048)   | 3.17% | 2.21% | 2.37% |
| Sp2GNO Burgers (s2048)| 0.59% | 0.53% | 0.39% |
| WNO Darcy (r141)      | 5.37% | 5.22% | 4.49% |
| Sp2GNO Darcy (r141)   | 1.30% | 0.98% | 0.77% |

**Resolution scaling (base variant), final test rel-L2:**

- WNO Burgers: 2048 → 2.21%, 4096 → 2.39%, 8192 → 2.38%
- Sp2GNO Burgers: 2048 → 0.53%, 4096 → 0.54%
- WNO Darcy: 141 → 5.22%, 281 → 2.95%, 421 → 3.50%
- Sp2GNO Darcy: 141 → 0.98%, 211 → 12.80%

---

## Repository layout

```
.
├── README.md                         ← this file
├── requirements.txt                  ← exact env that produced these results
├── RESULTS.md                        ← model-scale table (small/base/large)
├── RESULTS_resolution_scaling.md     ← resolution-scaling table
├── REPRODUCTION_NOTES.md             ← exact run grid + reproduction notes / gotchas
├── Jetson_Scifm_Edge_Arxiv.pdf       ← reference paper (Table 6 = param budgets)
│
├── train_wno_burgers.py              ← WNO 1D Burgers      (--variant, --res, --epochs, --ckpt)
├── train_wno_darcy.py                ← WNO 2D Darcy
├── train_sp2gno_burgers.py           ← Sp2GNO 1D Burgers
├── train_sp2gno_darcy.py             ← Sp2GNO 2D Darcy
├── sp2gno_core.py                    ← Sp2GNO model + shared graph/eigenbasis + utilities
├── collect_results.py               ← regenerates the two RESULTS*.md from runs/*/result.json
├── run_burgers.sh / run_darcy.sh     ← sequential drivers (set $CONDA_ENV, then run all configs)
│
├── sample_codes/                     ← reference WNO building blocks (imported by train_wno_*)
│   ├── wavelet_convolution.py        ←   WaveConv1d / WaveConv2d  (REQUIRED)
│   ├── utils.py                      ←   LpLoss, count_params, normalizers  (REQUIRED)
│   ├── wno1d_Burgers.py              ←   original reference WNO-1D
│   └── wno2d_Darcy_dwt.py            ←   original reference WNO-2D
│
├── checkpoints/                      ← 18 best-on-val checkpoints (<tag>.pth)
├── runs/<tag>/                       ← per-run result.json + curve.json + run.log (plots omitted)
├── logs/<tag>.log                    ← per-run training console log
├── docs/                             ← extended experiment write-ups
└── Jetson_data/                      ← burgers_split.json + README (raw .mat data NOT included)
```

The 18 `<tag>`s are `{wno,sp2gno}_{burgers,darcy}_{small,base,large}_<res>` plus the four
resolution-scaling runs; they match the checkpoint filenames one-to-one.

---

## Reproducing / extending

### 1. Environment
The results were produced with **Python 3.10, CUDA 11.8** (see `requirements.txt`):

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch_geometric==2.6.1 "numpy<2" scipy h5py matplotlib PyWavelets ptwt pytorch_wavelets
```

Notes:
- Pin **`torch_geometric==2.6.1`** — 2.7+/2.8 make `knn_graph` require `pyg-lib>=0.6.0`,
  which has no wheel for torch 2.1.0 (Sp2GNO builds its graph via `knn_graph`).
- Pin **`setuptools<81`** if `pytorch_wavelets` fails on `pkg_resources`.
- `numpy<2` is required by torch 2.1.x.

### 2. Data
Place the `.mat` files in `Jetson_data/` — see [`Jetson_data/README.md`](Jetson_data/README.md).

### 3. Train one configuration
```bash
python train_wno_burgers.py    --variant base --res 2048 --ckpt wno_burgers_base_r2048.pth
python train_sp2gno_darcy.py   --variant base --res 141  --batch_size 10 --ckpt sp2gno_darcy_base_r141.pth
```
Each run writes `checkpoints/<ckpt>`, `runs/<tag>/{result,curve}.json` + `run.log`, and
`logs/<tag>.log`. The **exact 18-run grid** (with the two reduced-epoch high-res runs) is
in [`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md); `run_burgers.sh` / `run_darcy.sh` run them in sequence.

### 4. Regenerate the tables
```bash
python collect_results.py     # rewrites RESULTS.md + RESULTS_resolution_scaling.md
```

---

## Protocol (identical across operators)

1000 epochs (500 for the two largest WNO-Darcy grids, 281/421), Adam (lr 1e-3, wd 1e-6),
StepLR, per-sample relative-L2 loss; the reported number is the **best-on-val** model's
**test** rel-L2. Variant knob = channel **width** (Sp2GNO fixed 6 layers / 64 graph-Fourier
modes; WNO fixed 4 layers, wavelet level scaled with resolution to hold the mode count).
See the `docs/` write-ups and `REPRODUCTION_NOTES.md` for the full configuration and design notes.
