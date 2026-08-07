#!/usr/bin/env python3
"""
Cross-backend batch-size-1 latency benchmark for the Sp2GNO/GraphFNO Heat
Exchanger runtime path (Table 36 in the paper: full / spectral-only / 2-layer)
on Jetson Orin Nano.

Note on repo location: the paper's companion Heat Exchanger checkpoints and
driver do NOT live in VirSO/sp2gno/sp2gno_new_benchmarks_june_2026 (that repo
only has the structured-grid Burgers/Darcy Sp2GNO runs). The actual Heat
Exchanger driver, model (`GraphFNO`, not `Sp2GNO`), and checkpoints live here,
in VirSO/For_Jetson/For_Jetson/. This script imports run_virso_inference_jetson.py
as a library (it only runs its heavy logic inside `main()`, guarded by
`if __name__ == "__main__"`) and reuses its exact data-loading, normalizer,
and GraphDataset construction, so the graph/features are identical to those
used to produce Table 8/9/36's Heat Exchanger rows.

Adds torch.compile and TensorRT (ONNX export -> trtexec -> TensorRT execution
context) on top of the paper's eager-only measurement. Short-run median/P95
over `--reps` reps after `--warmup` warmup calls; not a sustained/telemetry run.

Run inside the `extra_bench_virso` conda env (cloned from wave_gpu_310_test,
the env this project's run_virso_*.sh scripts actually use):
    conda activate extra_bench_virso
    export PYTHONNOUSERSITE=1; unset PYTHONPATH
    cd /home/jetson/VirSO/For_Jetson/For_Jetson
    python extra_inference/bench_extra_backends.py --case full
    python extra_inference/bench_extra_backends.py --case spectral
    python extra_inference/bench_extra_backends.py --case layer2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before `run_virso_inference_jetson` is imported: that module
# defaults to PYTORCH_NO_CUDA_MEMORY_CACHING=1 (a sustained-run OOM-safety
# audit mode) unless this is set, which disables the CUDA caching allocator
# and inflates every forward pass by ~5x (measured: 361ms vs 75ms steady-state
# for the "full" config) -- not representative of the paper's warm,
# synchronized, batch-size-one latency convention (Sec 4.3).
os.environ.setdefault("VIRSO_ALLOCATOR_AUDIT", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/home/jetson/jjyoo3/_extra_bench_common")

import run_virso_inference_jetson as V  # noqa: E402
from harness import time_callable, try_backend  # noqa: E402
from trt_utils import export_onnx, build_engine, run_engine  # noqa: E402

CONFIGS = {
    "full": dict(
        checkpoint=str(REPO_ROOT / "sp2gno_final.pth"),
        num_layers=10, width=48, max_mode=64, spectral=True, spatial=True,
    ),
    "spectral": dict(
        checkpoint="/home/jetson/VirSO/best_model_spectral.pth",
        num_layers=10, width=48, max_mode=64, spectral=True, spatial=False,
    ),
    "layer2": dict(
        checkpoint="/home/jetson/VirSO/best_model_2_layer.pth",
        num_layers=2, width=48, max_mode=40, spectral=True, spatial=True,
    ),
}

WARMUP = 5
REPS = 20

_DATASET_CACHE = {}


def get_test_sample():
    """Builds the GraphDataset exactly as run_virso_inference_jetson.main() does,
    and returns one batch-size-1 sample. Cached across the 3 configs since the
    graph/features (largest_mode=150 default) are shared; each GraphFNO slices
    U down to its own num_freq internally (GraphFourierLayer.no_low_freq)."""
    if "sample" in _DATASET_CACHE:
        return _DATASET_CACHE["sample"], _DATASET_CACHE["input_dim"]

    grid_pos, heat_prof, inlet, target = V.load_raw_data()
    input_dim = heat_prof.shape[1] + inlet.shape[1]
    splits = V.split_data(inlet, heat_prof, target)
    inlet_normalizer, heat_normalizer, output_normalizer_cpu, _ = V.build_normalizers(
        splits["train_inlet"], splits["train_heat"], splits["train_target"]
    )

    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.join(V.ROOT, "results")
    os.makedirs(base_folder, exist_ok=True)

    test_dataset = V.GraphDataset(
        splits["final_test_target"][:1],
        grid_pos,
        splits["final_test_heat"][:1],
        splits["final_test_inlet"][:1],
        inlet_normalizer,
        heat_normalizer,
        output_normalizer_cpu,
        False,
        base_folder,
        timestamp,
        largest_mode=V.dataset_largest_mode,
        custom_name="extra_inference_bench",
        custom_data_name="final",
        graph_type=V.graph_type,
        k=V.k,
        k_lower=V.k_lower,
        r=V.r,
        factor_of_lower=V.factor_of_lower,
        start_radius=V.start_radius,
    )
    sample = test_dataset[0]
    _DATASET_CACHE["sample"] = sample
    _DATASET_CACHE["input_dim"] = input_dim
    return sample, input_dim


class GraphFNOWrapper(torch.nn.Module):
    """Freezes U/edge_index/edge_weight/grid_pos/lip as buffers (batch_size=1
    fixed as a python constant), so the exported/compiled callable takes a
    single `x` tensor input, matching the single-input convention used for
    the other three families' extra_inference scripts."""

    def __init__(self, model, U, edge_index, edge_weight, grid_pos, lip):
        super().__init__()
        self.model = model
        self.register_buffer("U", U)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.register_buffer("grid_pos", grid_pos)
        self.register_buffer("lip", lip)

    def forward(self, x):
        return self.model(x, self.U, self.edge_index, self.edge_weight, self.grid_pos, self.lip, 1)


def load_wrapped_model(case: str):
    cfg = CONFIGS[case]
    sample, input_dim = get_test_sample()

    V.model_file = cfg["checkpoint"]
    V.num_sp2gno_layers = cfg["num_layers"]
    V.width = cfg["width"]
    V.max_mode = cfg["max_mode"]
    V.spectral = cfg["spectral"]
    V.spatial = cfg["spatial"]

    model = V.init_model(input_dim)
    model = V.load_model(model)

    x = sample.input.to("cuda").float()
    U = sample.U.to("cuda").float()
    edge_index = sample.edge_index.to("cuda")
    edge_weight = sample.edge_weight.to("cuda").float()
    grid_pos = sample.grid_pos.to("cuda").float()
    lip = sample.lipschitz_embedding.to("cuda").float()

    wrapper = GraphFNOWrapper(model, U, edge_index, edge_weight, grid_pos, lip).eval().cuda()
    return wrapper, x


def bench_eager(model, x):
    return time_callable(lambda: model(x), warmup=WARMUP, reps=REPS)


def bench_torchscript_trace(model, x):
    """torch.jit.script fails (PyG's MessagePassing/GraphFourierLayer internals
    aren't scriptable on this torch build), but torch.jit.trace was verified to
    actually succeed for all three configs -- checked for real, not assumed."""
    traced = torch.jit.trace(model, (x,))
    y_eager = model(x)
    y_traced = traced(x)
    if not torch.allclose(y_eager, y_traced, atol=1e-3):
        raise RuntimeError("traced output does not match eager output")
    return time_callable(lambda: traced(x), warmup=WARMUP, reps=REPS)


def bench_compile(model, x):
    compiled = torch.compile(model)
    return time_callable(lambda: compiled(x), warmup=WARMUP, reps=REPS)


def bench_tensorrt(model, x, workdir: Path, tag: str):
    onnx_path = workdir / f"{tag}.onnx"
    engine_path = workdir / f"{tag}.engine"
    export_onnx(model, (x,), str(onnx_path), ["x"], ["y"], opset=17)
    ok, log = build_engine(str(onnx_path), str(engine_path))
    if not ok:
        raise RuntimeError(f"trtexec engine build failed:\n{log}")
    result = run_engine(str(engine_path), warmup=WARMUP, reps=REPS)
    if result.get("status") != "success":
        raise RuntimeError(f"TRT engine run failed: {result.get('error')}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CONFIGS), required=True)
    args = ap.parse_args()

    model, x = load_wrapped_model(args.case)

    workdir = Path(__file__).parent / "trt_work"
    workdir.mkdir(exist_ok=True)

    results = {
        "family": "Sp2GNO (GraphFNO) / Heat Exchanger",
        "case": args.case,
        "checkpoint": CONFIGS[args.case]["checkpoint"],
        "input_shape": list(x.shape),
        "warmup": WARMUP,
        "reps": REPS,
        "note": ("Paper (Appendix D/E) states TorchScript is not defined for this runtime path. "
                 "torch.jit.script does fail here, but torch.jit.trace was verified for real "
                 "(see 'torchscript_trace') rather than assumed to fail."),
        "backends": {},
    }

    results["backends"]["eager"] = try_backend("eager", lambda: bench_eager(model, x))
    results["backends"]["torchscript_trace"] = try_backend(
        "torchscript_trace", lambda: bench_torchscript_trace(model, x)
    )
    results["backends"]["torch.compile"] = try_backend("torch.compile", lambda: bench_compile(model, x))
    results["backends"]["tensorrt"] = try_backend(
        "tensorrt", lambda: bench_tensorrt(model, x, workdir, args.case)
    )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.case}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Heat Exchanger / {args.case} ===")
    for name, r in results["backends"].items():
        if r["status"] == "success":
            print(f"  {name:14s} median={r['median_ms']:.3f} ms  p95={r['p95_ms']:.3f} ms")
        else:
            print(f"  {name:14s} FAIL  ({r['error_type']}: {r['error'][:150]})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
