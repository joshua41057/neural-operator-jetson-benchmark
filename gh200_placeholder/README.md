# GH200 server-reference records

Sustained FP32 telemetry from the NCSA Delta AI GH200 node, backing the
server-reference results in Section 8.2 / Table 9. These sweeps were run on a
Slurm-managed node and were not present on the Jetson host from which the rest
of this repository was assembled. The records are now deposited here.

Run tag: `gh200_fp32_20260715`.

## Layout

```
harness/
  gpu_power.py            NVML sampler + parser (nvidia-smi, 200 ms)
  aggregate_gh200.py      per-rep records -> summary/raw CSV
  audit_gh200.py          re-derive every value from the raw NVML logs
  slurm/                  batch scripts (bench, matrix, submit_all, smoke)
environment/
  environment.txt         GPU, driver, CUDA, torch, protocol
results/gh200_fp32_20260715/
  gh200_fp32_summary.csv  per-case mean/std over reps (reproduces Table 9)
  gh200_fp32_raw.csv      every rep
  {fno,deeponet,wno,sp2gno,hx}/
                          per-rep records + 200 ms NVML logs
```

## Metrics

| Column     | Definition                                                        |
|------------|-------------------------------------------------------------------|
| Med. (ms)  | mean over 3 reps of each rep's per-inference median latency        |
| Avg. (W)   | mean GPU power (NVML `power.draw`, 200 ms sampling)                |
| J/inf      | `avg_power / sustained_throughput` over the same window           |
| CUDA (MB)  | `torch.cuda.max_memory_allocated` (shares the Jetson definition)  |

Protocol: batch size 1, fp32_strict (TF32 disabled), 20 s warmup, 120 s
sustained window, 3 repetitions. See `environment/environment.txt`.

## Reproduce

```
python harness/aggregate_gh200.py --results-root results --run-tag gh200_fp32_20260715
python harness/audit_gh200.py     --results-root results --run-tag gh200_fp32_20260715
```

`aggregate` regenerates the two CSVs from the per-rep records. `audit`
independently re-parses the 200 ms NVML logs and checks every link in the chain
(below); both are included so `Avg. (W)` can be re-derived without trusting the
stored summaries.

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

## What the audit changed in the paper

The audit corrected the environment strings of Table 13, which had been recorded
from a different job than the one that produced the measurements:

| Field | Paper (before) | Measured |
|---|---|---|
| Driver | 590.48.01 | **595.71.05** |
| CUDA | 13.1 | **12.8** |
| PyTorch | 2.11.0+cu130 | **2.11.0+cu128** |

Addressable GPU memory (97,871 MiB) was confirmed correct. Section 8.2's
platform-difference sentence was corrected in the same pass.

## Notes

- **Heat Exchanger rep labels.** The aggregation step originally defaulted the
  nine heat exchanger rows to `rep=1` (the driver did not encode the repetition
  index in the output filename). This is corrected here: the records carry
  `rep=1,2,3`, one per distinct run (`full_rep{1,2,3}`, etc.). `n_reps=3` was
  always used for the mean, so Table 9 was never affected.
- **Path normalization.** Machine-specific prefixes in the records were replaced
  with placeholders (`<REPO>`, `<WORK>`, `<HOME>`, `<CONDA>`, `<JETSON_HPC>`,
  `<JETSON_HOME>`) and the Slurm account/partition with `<ACCOUNT>`/`<PARTITION>`.
  Only path strings were changed; no measured value was altered.
- **Excluded** (not needed to reproduce the telemetry): prediction/target `.npy`
  outputs, checkpoints and datasets (released with the Jetson artifacts), verbose
  per-inference runtime logs, and Slurm stdout.
