# Measured software stacks

Matches Tables 12 and 13 of the paper.

## Near-asset target — Jetson Orin Nano SUPER 8 GB

| | |
|---|---|
| JetPack | 6.2.1 (L4T R36.4.7) |
| CUDA | 12.6 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 (NVIDIA build) |
| OS | Ubuntu 22.04 (Jetson Linux) |
| Power / clock mode | `MAXN_SUPER` with `jetson_clocks` enabled |
| Telemetry | `tegrastats` `VDD_IN`, 100 ms sampling |
| Backends | PyTorch eager, TorchScript |
| Backend probe | TensorRT 10.7.0.23, ONNX opset 17, `trtexec` |

`jetson_clocks` does not persist across a reboot. Every sweep asserts it in a preflight
check and refuses to record timings when the governor does not report the pinned minimum
and maximum frequencies on both CPU and GPU.

## Server-reference target — GH200 (Delta AI)

| | |
|---|---|
| Driver / CUDA | 590.48.01 / CUDA 13.1 |
| PyTorch | 2.11.0+cu130 |
| Telemetry | `nvidia-smi` / NVML, 200 ms sampling |

Records to be added; see `gh200_placeholder/`.
