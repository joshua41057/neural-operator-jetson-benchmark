"""Shared, model-agnostic timing helpers for the extra_inference benchmarks.

Used inside each project's own conda env (imported via sys.path insertion),
so it only depends on torch -- no project-specific imports here.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import torch


def time_callable(fn: Callable[[], Any], warmup: int = 5, reps: int = 15) -> dict:
    """Warm up, then time `reps` synchronized batch-size-one calls. Returns ms stats."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    p95_idx = int(round(0.95 * (len(times_sorted) - 1)))
    p95 = times_sorted[p95_idx]
    return {
        "median_ms": median,
        "p95_ms": p95,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n": reps,
        "warmup": warmup,
        "all_ms": times,
    }


def try_backend(name: str, fn: Callable[[], dict]) -> dict:
    """Run a backend-producing callable; on failure, record a Fail entry instead of raising."""
    try:
        result = fn()
        result["backend"] = name
        result["status"] = "success"
        return result
    except Exception as e:  # noqa: BLE001 - deliberately broad: this is an admission gate
        return {
            "backend": name,
            "status": "fail",
            "error_type": type(e).__name__,
            "error": str(e)[:2000],
            "traceback_tail": "".join(traceback.format_exc().splitlines()[-8:]),
        }
