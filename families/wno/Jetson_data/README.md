# Data directory

The raw training/testing data (~4.1 GB total) is **not** shipped in this package to keep
it small. To re-train, place the following files here (`Jetson_data/`, which is the default
`--data_dir` for every training script):

| File | Size | Used by | Notes |
|------|------|---------|-------|
| `burgers_data_R10.mat`            | ~615 MB  | Burgers (WNO + Sp2GNO) | 1D Burgers, a(x)→u(x,T=1), native grid 8192 |
| `piececonst_r421_N1024_smooth1.mat` | ~1.6 GB | Darcy (WNO + Sp2GNO)  | Darcy train/val split (smooth1) |
| `piececonst_r421_N1024_smooth2.mat` | ~1.6 GB | Darcy (WNO + Sp2GNO)  | Darcy test split (smooth2) |
| `burgers_split.json`              | ~19 KB   | Burgers                | **included here** — fixed train/val/test indices |

These are the standard **WNO/FNO Burgers** and **Darcy (piecewise-constant, r421)**
benchmark datasets (as used in the FNO and WNO papers). Obtain them from the original
WNO/FNO data release and drop them in this folder unchanged.

- **Burgers grid** is subsampled from the native 8192 (`--res 2048/4096/8192`).
- **Darcy grid** is selected as evenly-spaced indices of the native 421 grid
  (`--res 141/211/281/421`).
- `burgers_split.json` is kept so the Burgers train/val/test partition is reproduced
  exactly; the Darcy split is deterministic (smooth1[:900] / smooth1[900:1000] /
  smooth2[:200]) and needs no file.
