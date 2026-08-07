#!/usr/bin/env python3
"""
Cross-backend batch-size-1 latency benchmark for WNO and Sp2GNO on Jetson Orin Nano.

The paper (Appendix D/E) benchmarks these two families under the eager backend
only (no TorchScript: the wavelet filter-bank buffers / graph caches are not
traceable). This script keeps that constraint and adds two more runtime paths:
torch.compile (Inductor) and TensorRT (ONNX export -> trtexec -> TensorRT
execution context), reusing the same checkpoints and cached graph tensors used
to produce Table 32 (WNO) and Table 35 (Sp2GNO).

Not a sustained/telemetry benchmark: short-run median/P95 over `--reps`
repetitions after `--warmup` warmup calls (Sec 4.3 timing convention, minus
tegrastats telemetry).

Sp2GNO checkpoints carry no architecture config (unlike WNO's), so width/
n_layers/num_freq/k are hardcoded here from the training-script convention
(VARIANTS={'small':13,'base':24,'large':45}, n_layers=6, num_freq=64) and
cross-checked against the checkpoint's own state_dict tensor shapes.

The Burgers-base Sp2GNO graph cache (burgers_s2048_k8_f64.pt) was absent from
this repo's cache/ dir (only the Darcy one shipped here) and has been copied
in from the sibling project VirSO/sp2gno/sp2gno_new_benchmarks_june_2026/cache/,
which builds the identical graph deterministically (seed=0) for the same
2048-node grid.

Run from the repo root, inside the `extra_bench` conda env:
    conda activate extra_bench
    export PYTHONNOUSERSITE=1; unset PYTHONPATH
    cd /home/jetson/jjyoo3/WNO_Sp2GNO_Benchmarks
    python extra_inference/bench_extra_backends.py --case wno_burgers_r2048
    python extra_inference/bench_extra_backends.py --case wno_darcy_r141
    python extra_inference/bench_extra_backends.py --case sp2gno_burgers_r2048
    python extra_inference/bench_extra_backends.py --case sp2gno_darcy_r141
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "_extra_bench_common"))

from train_wno_burgers import WNO1d  # noqa: E402
from train_wno_darcy import WNO2d  # noqa: E402
from sp2gno_core import Sp2GNO, SharedGraph  # noqa: E402
from harness import time_callable, try_backend  # noqa: E402
from trt_utils import export_onnx, build_engine, run_engine  # noqa: E402

BANK_DIR = Path("/home/jetson/data/wno_inference_banks_exact")
WARMUP = 5
REPS = 20


class Sp2GNOWrapper(torch.nn.Module):
    """Freezes the graph tensors (U, edge_index, edge_weight, lips) as buffers
    so the exported/compiled callable takes a single `feats` tensor input,
    matching the FNO/DeepONet/WNO single-input convention used elsewhere."""

    def __init__(self, model, U, edge_index, edge_weight, lips):
        super().__init__()
        self.model = model
        self.register_buffer("U", U)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.register_buffer("lips", lips)

    def forward(self, feats):
        return self.model(feats, self.U, self.edge_index, self.edge_weight, self.lips)


def load_wno_burgers():
    ck = torch.load(REPO_ROOT / "checkpoints/wno_burgers_base_r2048.pth",
                     map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = WNO1d(width=cfg["width"], level=cfg["level"], layers=cfg["layers"],
                  size=cfg["res"], wavelet="db6", in_channel=2, grid_range=1, padding=0)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.eval().cuda()
    bank = torch.load(BANK_DIR / "burgers_r2048_bank.pt", map_location="cpu", weights_only=False)
    x = bank["x"][0:1].float().cuda()  # [1, 2048, 1]
    return model, x


def load_wno_darcy():
    ck = torch.load(REPO_ROOT / "checkpoints/wno_darcy_base_r141.pth",
                     map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = WNO2d(width=cfg["width"], level=cfg["level"], layers=cfg["layers"],
                  size=[cfg["res"], cfg["res"]], wavelet="db6", in_channel=3,
                  grid_range=[1, 1], padding=1)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.eval().cuda()
    bank = torch.load(BANK_DIR / "darcy_r141_bank.pt", map_location="cpu", weights_only=False)
    x = bank["x"][0:1].float().cuda()  # [1, 141, 141, 1]
    return model, x


def load_sp2gno(case: str):
    if case == "burgers":
        ckpt_path = REPO_ROOT / "checkpoints/sp2gno_burgers_base_s2048.pth"
        cache_path = REPO_ROOT / "cache/burgers_s2048_k8_f64.pt"
        in_dim, N = 2, 2048
    else:
        ckpt_path = REPO_ROOT / "checkpoints/sp2gno_darcy_base_r141.pth"
        cache_path = REPO_ROOT / "cache/darcy_s141_k20_f64.pt"
        in_dim, N = 3, 141 * 141

    model = Sp2GNO(in_dim=in_dim, width=24, n_layers=6, N=N, num_freq=64, out_dim=1)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.eval().cuda()

    gfeat = torch.load(cache_path, map_location="cpu", weights_only=False)
    graph = SharedGraph(gfeat, "cuda")
    feats = torch.randn(1, N, in_dim, device="cuda")  # latency-only: shape/dtype match is what matters
    offs = torch.arange(1, device="cuda") * graph.N
    ei = graph.edge_index + offs.repeat_interleave(graph.E)
    ew = graph.edge_weight
    lips = graph.lips
    U = graph.U.unsqueeze(0).contiguous()

    wrapper = Sp2GNOWrapper(model, U, ei, ew, lips).eval().cuda()
    return wrapper, feats


LOADERS = {
    "wno_burgers_r2048": load_wno_burgers,
    "wno_darcy_r141": load_wno_darcy,
    "sp2gno_burgers_r2048": lambda: load_sp2gno("burgers"),
    "sp2gno_darcy_r141": lambda: load_sp2gno("darcy"),
}


def bench_eager(model, x):
    return time_callable(lambda: model(x), warmup=WARMUP, reps=REPS)


def bench_torchscript_trace(model, x):
    """torch.jit.script fails for both families (WNO: numpy/tensor mixing inside
    pytorch_wavelets; Sp2GNO: PyG's MessagePassing internals aren't scriptable).
    torch.jit.trace is attempted for real rather than assumed to fail: it also
    fails for WNO (same numpy/tensor root cause hit during tracing), but it
    actually succeeds for Sp2GNO -- verified against eager output below."""
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
    ap.add_argument("--case", choices=sorted(LOADERS), required=True)
    args = ap.parse_args()

    model, x = LOADERS[args.case]()
    family = "Sp2GNO" if args.case.startswith("sp2gno") else "WNO"

    workdir = Path(__file__).parent / "trt_work"
    workdir.mkdir(exist_ok=True)

    results = {
        "family": family,
        "case": args.case,
        "input_shape": list(x.shape),
        "warmup": WARMUP,
        "reps": REPS,
        "note": ("Paper (Appendix D/E) states TorchScript is not defined for this runtime path. "
                 "torch.jit.script does fail for both families, but torch.jit.trace was verified "
                 "here for real (see 'torchscript_trace') rather than assumed to fail."),
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

    print(f"\n=== {family} / {args.case} ===")
    for name, r in results["backends"].items():
        if r["status"] == "success":
            print(f"  {name:14s} median={r['median_ms']:.3f} ms  p95={r['p95_ms']:.3f} ms")
        else:
            print(f"  {name:14s} FAIL  ({r['error_type']}: {r['error'][:150]})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
