#!/usr/bin/env python3
"""Model-agnostic TensorRT engine latency + correctness runner.

Must be run under the system python (/usr/bin/python3), which is the only
interpreter on this device with a tensorrt build that matches the installed
TensorRT 10.7 libraries. Takes only an .engine path: shapes/dtypes for every
I/O tensor are introspected directly from the engine, so this script has no
project-specific knowledge.

Two modes:
  - Timing (default): inputs filled with random data, since only latency is
    being measured.
  - Correctness (--input-tensors given): loads a real input tensor dict
    (torch.save'd {name: cpu_tensor}) instead of random data, runs once, and
    writes the output tensor(s) to --output-tensors (torch.save'd {name:
    cpu_tensor}). Used to compute the admission-gate predictive-validity
    check (rel-L2 vs FP32 eager, Eq. 22) for a TensorRT runtime path -- the
    timing-only mode alone only checks execution-validity (A_exec), not
    A_pred, per the framework's A(R) = A_pred(R) and A_exec(R).
"""
import argparse
import json
import time

import tensorrt as trt
import torch

TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}
if hasattr(trt.DataType, "INT64"):
    TRT_TO_TORCH_DTYPE[trt.DataType.INT64] = torch.int64
if hasattr(trt.DataType, "BF16"):
    TRT_TO_TORCH_DTYPE[trt.DataType.BF16] = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", required=False)
    ap.add_argument("--input-tensors", required=False,
                     help="path to a torch.save'd {name: cpu_tensor} dict of real inputs")
    ap.add_argument("--output-tensors", required=False,
                     help="path to write the torch.save'd {name: cpu_tensor} dict of outputs")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")
    context = engine.create_execution_context()

    device = torch.device("cuda")
    stream = torch.cuda.Stream()

    real_inputs = None
    if args.input_tensors:
        real_inputs = torch.load(args.input_tensors, map_location="cpu", weights_only=False)

    tensors = {}
    input_names = []
    output_names = []
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in names:
        shape = tuple(engine.get_tensor_shape(name))
        if any(d < 0 for d in shape):
            raise RuntimeError(f"tensor {name} has dynamic shape {shape}; expected static engine")
        trt_dtype = engine.get_tensor_dtype(name)
        torch_dtype = TRT_TO_TORCH_DTYPE.get(trt_dtype)
        if torch_dtype is None:
            raise RuntimeError(f"unsupported TRT dtype {trt_dtype} for tensor {name}")
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        (input_names if is_input else output_names).append(name)
        if is_input and real_inputs is not None:
            t = real_inputs[name].to(device=device, dtype=torch_dtype).contiguous()
            if tuple(t.shape) != shape:
                raise RuntimeError(f"real input {name} shape {tuple(t.shape)} != engine shape {shape}")
        elif torch_dtype.is_floating_point:
            t = torch.randn(shape, dtype=torch_dtype, device=device)
        else:
            t = torch.zeros(shape, dtype=torch_dtype, device=device)
        tensors[name] = t
        context.set_tensor_address(name, t.data_ptr())

    def run_once():
        context.execute_async_v3(stream.cuda_stream)

    if real_inputs is not None:
        # Correctness mode: single deterministic forward on real data, no timing loop.
        with torch.no_grad():
            run_once()
            stream.synchronize()
        outputs = {name: tensors[name].detach().cpu() for name in output_names}
        torch.save(outputs, args.output_tensors)
        print(f"Saved {len(outputs)} output tensor(s) to {args.output_tensors}")
        return

    with torch.no_grad():
        for _ in range(args.warmup):
            run_once()
        stream.synchronize()
        times = []
        for _ in range(args.reps):
            stream.synchronize()
            t0 = time.perf_counter()
            run_once()
            stream.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    p95_idx = int(round(0.95 * (len(times_sorted) - 1)))
    result = {
        "status": "success",
        "median_ms": median,
        "p95_ms": times_sorted[p95_idx],
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n": args.reps,
        "warmup": args.warmup,
        "all_ms": times,
        "io_tensors": {n: {"shape": list(engine.get_tensor_shape(n)),
                            "dtype": str(engine.get_tensor_dtype(n))} for n in names},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "all_ms"}, indent=2))


if __name__ == "__main__":
    main()
