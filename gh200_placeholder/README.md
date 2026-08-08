# GH200 server-reference records

The GH200 measurements of Section 8.2 and Table 9 were produced on the Delta AI
platform, not on the Jetson host this repository was assembled from. The records
themselves are not yet included here.

## Audit status

The full GH200 pipeline was independently recomputed against its raw NVML logs.
The audit covered the whole matrix — **27 cases x 3 repetitions = 81 repetition
records**, including the small and large scale groups that Table 9 does not print —
and recomputed every link in the chain rather than checking the endpoints:

| # | Link | Result |
|---|---|---|
| 1 | 27 cases x rep{1,2,3} present, status=success | no missing or failed runs |
| 2 | 81 NVML logs re-parsed independently of the harness | matches stored `avg_w`, error < 0.05 W |
| 3 | sampling interval and window from log timestamps | median exactly 200 ms; window 119.9–123.1 s |
| 4 | `J/inf == avg_power / throughput` per repetition | agrees within 1% |
| 5 | per-repetition native values vs `gh200_fp32_raw.csv` | no mismatch |
| 6 | three-repetition mean vs `gh200_fp32_summary.csv` | no mismatch |
| 7 | summary vs the 44 printed cells of Table 9 | **44/44** |
| 8 | FP32 strictness (`allow_tf32_matmul` and related) | TF32 never enabled |

Each log carries 600–616 samples, consistent with a 120 s window at 200 ms.

**No value error was found.** Table 9 is traceable to raw NVML logs at every step.

## What the audit did change

The audit corrected the environment strings of Table 13, which had been recorded
from a different job than the one that produced the measurements:

| Field | Paper (before) | Measured |
|---|---|---|
| Driver | 590.48.01 | **595.71.05** |
| CUDA | 13.1 | **12.8** |
| PyTorch | 2.11.0+cu130 | **2.11.0+cu128** |

Addressable GPU memory (97,871 MiB) was confirmed correct. Section 8.2's
platform-difference sentence was corrected in the same pass.

## Known cosmetic issue in the raw records

In `gh200_fp32_raw.csv` the nine heat exchanger rows are all labelled `rep=1`,
because the driver did not encode the repetition index in the output filename and
the aggregation step defaulted it. The three repetitions are genuinely distinct
measurements:

| Case | rep medians (ms) |
|---|---|
| `hx_full` | 10.430 / 10.663 / 11.208 |
| `hx_spectral` | 5.650 / 5.734 / 5.739 |
| `hx_layer2` | 2.975 / 3.138 / 3.194 |

`n_reps=3` was used for the mean, so the Table 9 values are unaffected. The labels
should be corrected to `rep=1,2,3` before the raw records are published here.

## What to deposit here

- per-configuration, per-repetition median latency (3 repetitions)
- NVML power series or its per-repetition summary
- `torch.cuda.max_memory_allocated` per configuration
- the job scripts and Slurm batch files
- an environment capture (`nvidia-smi`, module list or `pip freeze`)
- the audit script

Checkpoints and datasets are not needed: they are attached to the release of this
repository. Binary profiler traces are not needed either.
