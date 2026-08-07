# Shared measurement protocol

Section 4.3 of the paper states that all timing is obtained under **a single protocol
applied identically to every operator family**. This directory is that protocol. The
per-family code under `families/` implements each forward pass; everything that defines
*how it is measured* lives here.

| File | Role |
|---|---|
| `preflight_clocks.sh` | Refuses to measure unless the board is in the declared state: `MAXN_SUPER`, `jetson_clocks` applied, CPU and GPU `min_freq == max_freq`, GPU max `= 1020000000`. Sourced and called by every sweep; a failure returns non-zero and the sweep exits. |
| `run_unified_shortrun.sh` | **Short-run class.** Fixed number of synchronized forward passes after warmup, median and P95, `R = 3`. Basis for the cross-family, model-scale and resolution comparisons of Section 5. |
| `run_unified_sustained.sh` | **Sustained class.** One configuration held under continuous batch-size-one execution for a 120 s window with concurrent `tegrastats`, `R = 3`. Basis for every power, energy, thermal and board-memory quantity in Section 8. |
| `campaign/` | Orchestration for gap-filling and retries within a campaign. |

The two classes are **not interchangeable**: a short-run and a sustained median of the
same configuration are different quantities because the device occupies a different
thermal and clock state. Records are stored separately and labelled by class throughout.

## Why the preflight matters

`jetson_clocks` does not survive a reboot. Without the preflight a sweep can silently
record timings from an unpinned board, which would not be comparable with the rest of
the benchmark. `run_unified_shortrun.sh` and `run_unified_sustained.sh` both begin with

```bash
. preflight_clocks.sh
preflight_clocks || exit 1
```
