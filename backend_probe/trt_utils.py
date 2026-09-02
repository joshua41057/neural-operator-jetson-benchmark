"""ONNX export + trtexec engine-build helpers.

Runs inside the project's own conda env (needs torch + onnx only).
Engine *inference* is done separately by trt_runner.py under system python,
which is the only interpreter on this device with matching tensorrt+torch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
SYSTEM_PYTHON = "/usr/bin/python3"


def export_onnx(model, inputs, onnx_path: str, input_names: Sequence[str],
                 output_names: Sequence[str], opset: int = 17, export_on_cpu: bool = True) -> None:
    """Export to ONNX. `export_on_cpu` moves model+inputs to CPU for the export call only
    (a standard workaround: torch.onnx's interpolate/shape-inference symbolics can raise a
    spurious cuda/cpu device-mismatch RuntimeError during constant folding on CUDA tensors;
    the exported graph and the later TensorRT engine build/inference are unaffected)."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    model.eval()
    if export_on_cpu:
        orig_device = next(model.parameters()).device
        model_cpu = model.to("cpu")
        inputs_cpu = tuple(t.to("cpu") if torch.is_tensor(t) else t for t in inputs)
    else:
        model_cpu, inputs_cpu = model, inputs
    try:
        with torch.no_grad():
            torch.onnx.export(
                model_cpu,
                inputs_cpu,
                str(onnx_path),
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=None,  # static shapes: matches batch-size-1 fixed-shape deployment
            )
    finally:
        if export_on_cpu:
            model.to(orig_device)


def build_engine(onnx_path: str, engine_path: str, timeout: int = 900,
                  extra_args: Sequence[str] = ()) -> tuple[bool, str]:
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
    ] + list(extra_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return False, f"trtexec build timed out after {timeout}s: {e}"
    log = proc.stdout[-4000:] + "\n---STDERR---\n" + proc.stderr[-4000:]
    ok = proc.returncode == 0 and engine_path.exists()
    return ok, log


def run_engine(engine_path: str, warmup: int = 5, reps: int = 15, timeout: int = 300) -> dict:
    """Invoke the standalone TRT runner under system python (matching tensorrt build)."""
    import json
    import tempfile
    out_path = Path(tempfile.mktemp(suffix=".json"))
    runner = str(Path(__file__).parent / "trt_runner.py")
    cmd = [SYSTEM_PYTHON, runner, "--engine", str(engine_path),
           "--warmup", str(warmup), "--reps", str(reps), "--out", str(out_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return {"status": "fail", "error_type": "TimeoutExpired", "error": str(e)}
    if proc.returncode != 0 or not out_path.exists():
        return {
            "status": "fail",
            "error_type": "TRTRunnerError",
            "error": (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]),
        }
    result = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    return result


def run_engine_correctness(engine_path: str, input_tensors: dict, timeout: int = 120) -> dict:
    """Feeds real input tensor(s) (name -> cpu tensor) through the built TensorRT engine
    and returns the real output tensor(s) (name -> cpu tensor), for the admission-gate
    predictive-validity check (A_pred, Eq. 22) -- as opposed to `run_engine`, which only
    checks that the path executes (A_exec) using random data for latency timing."""
    import tempfile
    in_path = Path(tempfile.mktemp(suffix=".pt"))
    out_path = Path(tempfile.mktemp(suffix=".pt"))
    import torch
    torch.save({k: v.detach().cpu() for k, v in input_tensors.items()}, in_path)
    runner = str(Path(__file__).parent / "trt_runner.py")
    cmd = [SYSTEM_PYTHON, runner, "--engine", str(engine_path),
           "--input-tensors", str(in_path), "--output-tensors", str(out_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return {"status": "fail", "error_type": "TimeoutExpired", "error": str(e)}
    if proc.returncode != 0 or not out_path.exists():
        return {
            "status": "fail",
            "error_type": "TRTRunnerError",
            "error": (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]),
        }
    outputs = torch.load(out_path, map_location="cpu", weights_only=False)
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return {"status": "success", "outputs": outputs}
