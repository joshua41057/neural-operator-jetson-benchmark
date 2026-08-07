# Provenance of the measurement records

Every number in the paper comes from a named campaign. Superseded campaigns are **not**
included in this repository, so that no result in `families/*/results/` is ambiguous.

## Canonical campaigns

| Family | Canonical path | Notes |
|---|---|---|
| FNO | `families/fno/results/` | |
| DeepONet | `families/deeponet/results/` | |
| WNO | `families/wno/results/jetson_wno_exact/` | |
| Sp²GNO | `families/sp2gno/inference_runs/`, `families/sp2gno/results/` | June 2026 campaign |
| Heat exchanger | `families/heat_exchanger/inference_runs/` | |

## Deliberately excluded

| Excluded | Reason |
|---|---|
| `WNO_Sp2GNO_Benchmarks/_archive_unused/runs/sp2gno_*` | Sp²GNO campaign superseded when the graph-Laplacian eigenbasis cache was regenerated and the affected Darcy checkpoints were retrained (Table 7 footnote). The canonical Sp²GNO records are the June 2026 campaign above. |
| `WNO_Sp2GNO_Benchmarks/checkpoints/_old_bad_*` | Superseded checkpoints. |
| `EDCNO_DeepONet/archive_noncanonical/` | Non-canonical runs. |
| `sp2gno/_archive_old_runs/`, `_archive_old_scripts/` | Superseded. |

## Two measurement classes

Short-run and sustained records are stored separately and are never merged. A short-run
median and a sustained median of the same configuration are different quantities: the
device occupies a different thermal and clock state. Table captions in the paper label
every value by class.
