# Paper table → data file index

Mapping from each table in the paper to the records it was generated from.
Paths are relative to the repository root.

> Status: main-text tables mapped; appendix tables to be completed before submission.

## Main text

| Table | Content | Source |
|---|---|---|
| 5 | FP32 cost with held-out error | `families/*/results/` (per-family validity + short-run) |
| 6 | Precision-mode executability | `families/*/results/` precision sweeps |
| 7 | Reduced-precision perturbation | `families/*/results/` perturbation campaign |
| 8 | Sustained Jetson telemetry | `families/*/` sustained sweeps (120 s, R=3) |
| 10 | Deployment frontiers | `families/*/results/` + `families/fno/results/` frontier sweep |
| 11 | First-evaluation guidance | derived from Tables 5–10 and §9 |

## Appendices

| Table | Content | Source |
|---|---|---|
| 14–23 | FNO | `families/fno/results/` |
| 24–30 | DeepONet | `families/deeponet/results/` |
| 31–33 | WNO | `families/wno/results/jetson_wno_exact/` |
| 34–38 | Sp²GNO | `families/sp2gno/inference_runs/`, `families/sp2gno/results/` |
| 39–47 | Profiling counters | `families/*/results/profiles/*_exports/` |
| 48–49 | Backend admission | `backend_probe/` |
