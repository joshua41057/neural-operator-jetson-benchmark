"""
gpu_power.py — GH200 GPU power/memory/thermal sampler.

Drop-in replacement for the Jetson `tegrastats` telemetry seam used by the
jetson-hpc benchmark harnesses. Instead of reading Jetson board rails
(VDD_IN mW) via `tegrastats`, this samples the GH200 GPU via `nvidia-smi`
(GPU-level, NVML-backed) in a background subprocess and parses the log.

Design goals:
  * No new Python dependency (uses the `nvidia-smi` CLI, always present on
    the compute node). pynvml is NOT installed in the wno_sp2gno env.
  * Same context-manager / parse shape as the harnesses' TegraMonitor so the
    port is a mechanical swap.
  * Canonical output keys that each harness maps onto the exact key names its
    downstream aggregator already expects (e.g. `vdd_in_mean_w`).

Metrics captured per sample: power.draw (W), memory.used (MiB, process/context
residency), temperature.gpu (C), utilization.gpu (%).

Note on "GPU MB": the *primary* GPU-memory number reported in the paper table
is torch.cuda.max_memory_allocated (captured by each harness directly, matching
the Jetson "CUDA MB" definition). The nvidia-smi memory.used here is the
process residency (includes the CUDA context, ~hundreds of MB) and is reported
as a secondary column.
"""

from __future__ import annotations

import math
import os
import signal
import statistics as _stats
import subprocess
import time
from pathlib import Path
from typing import Any


NVSMI_QUERY = "timestamp,power.draw,memory.used,temperature.gpu,utilization.gpu"


class NvsmiMonitor:
    """Background `nvidia-smi` sampler writing CSV to `log_path`.

    Mirrors the TegraMonitor context-manager interface used by the Jetson
    harnesses: `with NvsmiMonitor(log, interval_ms=200): <timed region>`.
    """

    def __init__(self, log_path, interval_ms: int = 200, gpu_index: int = 0,
                 enabled: bool = True):
        self.log_path = str(log_path)
        self.interval_ms = int(interval_ms)
        self.gpu_index = int(gpu_index)
        self.enabled = enabled
        self.proc = None
        self._f = None

    def __enter__(self):
        if not self.enabled:
            return self
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={NVSMI_QUERY}",
            "--format=csv,noheader,nounits",
            "-lms", str(self.interval_ms),
        ]
        self._f = open(self.log_path, "w")
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self._f,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            # nvidia-smi absent (should not happen on a GPU node). Degrade to
            # a disabled monitor rather than crashing the benchmark.
            self.proc = None
            self.enabled = False
            return self
        # Let at least one sample land before the timed region starts.
        time.sleep(max(0.1, 2.0 * self.interval_ms / 1000.0))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if self._f is not None:
            try:
                self._f.flush()
                self._f.close()
            except Exception:
                pass
        time.sleep(0.1)


def _fnum(s):
    s = s.strip()
    if not s or s.lower().startswith("[n/a") or s.lower() == "n/a":
        return None
    try:
        return float(s)
    except Exception:
        return None


def summarize_nvsmi(log_path) -> dict[str, Any]:
    """Parse an NvsmiMonitor CSV log into canonical summary statistics.

    Returns NaNs when the log is missing/empty so callers can still record a
    result row.
    """
    out = {
        "nvsmi_samples": 0,
        "gpu_power_mean_w": math.nan,
        "gpu_power_median_w": math.nan,
        "gpu_power_min_w": math.nan,
        "gpu_power_max_w": math.nan,
        "gpu_power_p95_w": math.nan,
        "gpu_mem_used_mean_mb": math.nan,
        "gpu_mem_used_peak_mb": math.nan,
        "gpu_temp_peak_c": math.nan,
        "gpu_util_mean_pct": math.nan,
        "gpu_util_peak_pct": math.nan,
    }

    log_path = str(log_path)
    if not os.path.exists(log_path):
        return out

    powers, mems, temps, utils = [], [], [], []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            # fields: timestamp, power.draw(W), memory.used(MiB), temp(C), util(%)
            p = _fnum(parts[1])
            m = _fnum(parts[2])
            t = _fnum(parts[3])
            u = _fnum(parts[4])
            if p is not None:
                powers.append(p)
            if m is not None:
                mems.append(m)
            if t is not None:
                temps.append(t)
            if u is not None:
                utils.append(u)

    out["nvsmi_samples"] = len(powers)
    if powers:
        out["gpu_power_mean_w"] = float(_stats.mean(powers))
        out["gpu_power_median_w"] = float(_stats.median(powers))
        out["gpu_power_min_w"] = float(min(powers))
        out["gpu_power_max_w"] = float(max(powers))
        out["gpu_power_p95_w"] = float(_percentile(powers, 95))
    if mems:
        out["gpu_mem_used_mean_mb"] = float(_stats.mean(mems))
        out["gpu_mem_used_peak_mb"] = float(max(mems))
    if temps:
        out["gpu_temp_peak_c"] = float(max(temps))
    if utils:
        out["gpu_util_mean_pct"] = float(_stats.mean(utils))
        out["gpu_util_peak_pct"] = float(max(utils))
    return out


def _percentile(xs, q):
    if not xs:
        return math.nan
    s = sorted(xs)
    idx = int(round((q / 100.0) * (len(s) - 1)))
    idx = max(0, min(len(s) - 1, idx))
    return s[idx]


def tegrastats_compatible(nvsmi_summary: dict[str, Any]) -> dict[str, Any]:
    """Map canonical nvidia-smi stats onto the tegrastats key names that the
    Jetson harnesses/aggregators consume, so downstream code is unchanged.

    `vdd_in_*_w`  <- GPU power.draw (the "Avg. W" column)
    `board_ram_*` <- GPU memory.used residency (secondary memory column)
    `peak_temp_c` <- GPU temperature
    `gr3d_*_pct`  <- GPU utilization
    plus all canonical `gpu_*` keys retained for provenance.
    """
    s = nvsmi_summary
    mapped = {
        "tegrastats_samples": s.get("nvsmi_samples", 0),
        "vdd_in_mean_w": s.get("gpu_power_mean_w", math.nan),
        "vdd_in_min_w": s.get("gpu_power_min_w", math.nan),
        "vdd_in_max_w": s.get("gpu_power_max_w", math.nan),
        "vdd_in_last_avg_field_w": s.get("gpu_power_mean_w", math.nan),
        "board_ram_mean_mb": s.get("gpu_mem_used_mean_mb", math.nan),
        "board_ram_peak_mb": s.get("gpu_mem_used_peak_mb", math.nan),
        "peak_temp_c": s.get("gpu_temp_peak_c", math.nan),
        "gr3d_mean_pct": s.get("gpu_util_mean_pct", math.nan),
        "gr3d_peak_pct": s.get("gpu_util_peak_pct", math.nan),
    }
    mapped.update(s)
    return mapped
