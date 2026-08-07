# Neural-Operator Jetson Benchmark

Benchmark harness and measured results for **"Deploying PDE Neural Operators as Trained:
Mechanism-Governed Cost and Backend Reachability on the NVIDIA Jetson Orin Nano."**

The paper evaluates a deployed model not as a trained checkpoint but as an *executed runtime
path* `R = (G_θ, p, b, s, h)` — checkpoint, precision mode, backend, software stack, hardware
target. This repository contains the code that realizes that protocol and the per-configuration
records behind every table in the paper.

## How to read this repository

The paper's claim is that deployed cost is organized by the **dominant computational
mechanism of the executed path**, not by the architectural family label. The directories
under `families/` are named by family because that is how the measurement campaigns were
actually run; the mechanism each one instantiates is the quantity the paper carries
forward.

| Mechanism (the organizing axis) | Family measured here | Stress axis | Reaches TensorRT |
|---|---|---|---|
| FFT / spectral transform | FNO | spatial resolution | no — ONNX opset |
| Dense branch–trunk | DeepONet | output-query count | **yes**, 3.8× |
| Multiresolution wavelet | WNO | resolution + transform depth | no — tracing |
| Graph-spectral + movement | Sp²GNO, heat exchanger | graph size | no — engine build |

Start with `harness/` (the protocol that is identical across all of them),
then `predictions/` (the four falsifiable axes), then `docs/TABLE_INDEX.md`.

## What is here

| Path | Contents |
|---|---|
| `families/fno/` | Fourier Neural Operator: training, deployment sweep, profiling |
| `families/deeponet/` | Branch–trunk DeepONet |
| `families/wno/` | Wavelet Neural Operator |
| `families/sp2gno/` | Spatio-Spectral Graph Neural Operator, Burgers and Darcy |
| `families/heat_exchanger/` | Sp²GNO on an irregular CFD mesh — **deployment measurements only** (see its README) |
| `backend_probe/` | TorchScript / ONNX→TensorRT / torch.compile admission probe (§7, Appendix G) |
| `figures/` | Figure 2–4 generation |
| `harness/` | **The shared measurement protocol** — clock preflight, short-run and sustained sweeps |
| `predictions/` | The four falsifiable predictions of §1 and where each is tested |
| `verification/` | Two checks: table→record and prose→table |
| `docs/TABLE_INDEX.md` | **Paper table → script → data file** mapping |
| `docs/PROVENANCE.md` | Which measurement campaign each result belongs to |

## What is *not* here

| Excluded | Where it lives |
|---|---|
| Trained checkpoints (`.pt`, `.pth`) | Attached to the [latest release](../../releases/latest) |
| Nsight Systems / Nsight Compute binary traces (`.ncu-rep`, `.nsys-rep`) | Not distributed. The per-kernel counters they contain are exported to CSV under `families/*/results/profiles/*_exports/`, which is what Appendix F is built from. |
| Burgers and Darcy datasets | Public release of Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021 |
| Heat exchanger data, checkpoints, architecture and training | Companion study — https://github.com/joshua41057/virso-jetson-inference |
| Superseded measurement campaigns | Intentionally omitted; see `docs/PROVENANCE.md` |

## Measurement protocol

Two measurement classes, never merged:

- **Short-run** — 100 synchronized forward passes after warmup, median and P95, `R = 3`.
- **Sustained** — 120 s continuous batch-size-one execution with concurrent `tegrastats`, `R = 3`.

Every sweep asserts `jetson_clocks` in a preflight check and refuses to record timings when the
clocks are not pinned. The backend-admission probe of `backend_probe/` does *not* assert that
preflight; its latencies are read only as within-run ratios.

## Platform

Jetson Orin Nano SUPER 8 GB · JetPack 6.2.1 (L4T R36.4.7) · CUDA 12.6 ·
PyTorch 2.5.0a0+872d972e41.nv24.08. Full stack in `environment/VERSIONS.md`.

## Reproducing a table

See `docs/TABLE_INDEX.md`, then:

```bash
cd verification && python verify_all.py     # re-checks tabulated values against raw records
```

## Citation

See `CITATION.cff`.
