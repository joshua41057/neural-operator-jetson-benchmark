from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("results/jetson_fno_precision")
OUT_DIR = Path("results/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "precision_runs_raw.csv"
PAIR_CSV = OUT_DIR / "precision_tf32_vs_fp32.csv"
FAIL_CSV = OUT_DIR / "precision_failure_summary.csv"
FAIL_EXAMPLES_CSV = OUT_DIR / "precision_failure_examples.csv"


def classify_failure(error_message: str) -> str:
    msg = error_message or ""

    if "Unsupported dtype BFloat16" in msg:
        return "fft_bf16_unsupported"

    if "cuFFT only supports dimensions whose sizes are powers of two" in msg:
        m = re.search(r"signal size of\[(.*?)\]", msg)
        if m:
            return f"fft_fp16_non_power_of_two_size_{m.group(1)}"
        return "fft_fp16_non_power_of_two_size"

    if "Input type (c10::Half)" in msg or "bias type" in msg:
        return "fp16_dtype_mismatch"

    if "expected scalar type" in msg.lower():
        return "dtype_mismatch"

    if "out of memory" in msg.lower() or "cuda error: out of memory" in msg.lower():
        return "cuda_oom"

    if "not implemented" in msg.lower():
        return "kernel_not_implemented"

    return "other_runtime_failure"


def parse_tag(tag: str) -> dict[str, str]:
    suffixes = [
        "_eager_fp32_strict",
        "_eager_tf32",
        "_eager_bf16_autocast",
        "_eager_fp16_autocast",
        "_eager_fp16_native",
        "_torchscript_fp32_strict",
        "_torchscript_tf32",
        "_torchscript_bf16_autocast",
        "_torchscript_fp16_autocast",
        "_torchscript_fp16_native",
    ]

    for suffix in suffixes:
        if tag.endswith(suffix):
            prefix = tag[: -len(suffix)]
            if suffix.startswith("_eager_"):
                mode = "eager"
                precision_mode = suffix[len("_eager_"):]
            else:
                mode = "torchscript"
                precision_mode = suffix[len("_torchscript_"):]

            # prefix format: experiment_name_seedX
            m = re.match(r"(.+)_seed([0-9]+)$", prefix)
            if m:
                exp = m.group(1)
                seed = m.group(2)
            else:
                exp = prefix
                seed = ""

            return {
                "tag": tag,
                "experiment_name": exp,
                "selected_seed": seed,
                "mode": mode,
                "precision_mode": precision_mode,
            }

    return {
        "tag": tag,
        "experiment_name": "",
        "selected_seed": "",
        "mode": "",
        "precision_mode": "",
    }


def safe_get(d: dict[str, Any], key: str):
    v = d.get(key)
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


rows = []

for p in sorted(RESULTS_DIR.glob("*.json")):
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    tag = p.stem
    meta = parse_tag(tag)
    status = d.get("status", "unknown")
    err = d.get("error_message", "")

    row = {
        **meta,
        "json_path": str(p),
        "status": status,
        "dataset": d.get("dataset"),
        "resolution": json.dumps(d.get("resolution")),
        "parameter_count": d.get("parameter_count"),
        "input_bank": d.get("input_bank"),
        "input_shape": json.dumps(d.get("input_shape")),
        "input_dtype": d.get("input_dtype"),
        "mean_ms": d.get("mean_ms"),
        "median_ms": d.get("median_ms"),
        "p95_ms": d.get("p95_ms"),
        "p99_ms": d.get("p99_ms"),
        "repeat_median_ms_median": d.get("repeat_median_ms_median"),
        "repeat_median_ms_std": d.get("repeat_median_ms_std"),
        "peak_cuda_alloc_mb": d.get("peak_cuda_alloc_mb"),
        "peak_cuda_reserved_mb": d.get("peak_cuda_reserved_mb"),
        "error_type": d.get("error_type"),
        "failure_class": classify_failure(err) if status != "success" else "",
        "error_message_short": (err[:300].replace("\n", " ") if err else ""),
    }
    rows.append(row)


fieldnames = [
    "tag", "experiment_name", "selected_seed", "mode", "precision_mode",
    "json_path", "status", "dataset", "resolution", "parameter_count",
    "input_bank", "input_shape", "input_dtype",
    "mean_ms", "median_ms", "p95_ms", "p99_ms",
    "repeat_median_ms_median", "repeat_median_ms_std",
    "peak_cuda_alloc_mb", "peak_cuda_reserved_mb",
    "error_type", "failure_class", "error_message_short",
]

with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)


# TF32 vs FP32 strict pair table
by_key = {}
for r in rows:
    key = (r["experiment_name"], r["selected_seed"], r["mode"])
    by_key.setdefault(key, {})[r["precision_mode"]] = r

pair_rows = []
for key, vals in sorted(by_key.items()):
    fp32 = vals.get("fp32_strict")
    tf32 = vals.get("tf32")
    if not fp32 or not tf32:
        continue

    fp32_ok = fp32["status"] == "success"
    tf32_ok = tf32["status"] == "success"

    speedup = ""
    delta_pct = ""
    if fp32_ok and tf32_ok and fp32["median_ms"] and tf32["median_ms"]:
        fp32_med = float(fp32["median_ms"])
        tf32_med = float(tf32["median_ms"])
        speedup = fp32_med / tf32_med if tf32_med > 0 else ""
        delta_pct = 100.0 * (tf32_med - fp32_med) / fp32_med if fp32_med > 0 else ""

    pair_rows.append({
        "experiment_name": key[0],
        "selected_seed": key[1],
        "mode": key[2],
        "dataset": fp32.get("dataset") or tf32.get("dataset"),
        "resolution": fp32.get("resolution") or tf32.get("resolution"),
        "parameter_count": fp32.get("parameter_count") or tf32.get("parameter_count"),
        "fp32_status": fp32["status"],
        "tf32_status": tf32["status"],
        "fp32_median_ms": fp32["median_ms"],
        "tf32_median_ms": tf32["median_ms"],
        "tf32_speedup_vs_fp32": speedup,
        "tf32_latency_delta_pct": delta_pct,
        "fp32_peak_cuda_alloc_mb": fp32["peak_cuda_alloc_mb"],
        "tf32_peak_cuda_alloc_mb": tf32["peak_cuda_alloc_mb"],
    })

with open(PAIR_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else [])
    if pair_rows:
        w.writeheader()
        w.writerows(pair_rows)


# Failure summary by dataset/backend/precision/failure_class
counter = Counter()
examples = {}
for r in rows:
    if r["status"] == "success":
        continue
    key = (
        r["dataset"],
        r["mode"],
        r["precision_mode"],
        r["failure_class"],
    )
    counter[key] += 1
    examples.setdefault(key, r)

fail_rows = []
for key, count in sorted(counter.items()):
    dataset, mode, precision_mode, failure_class = key
    fail_rows.append({
        "dataset": dataset,
        "mode": mode,
        "precision_mode": precision_mode,
        "failure_class": failure_class,
        "num_failures": count,
    })

with open(FAIL_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "mode", "precision_mode", "failure_class", "num_failures"])
    w.writeheader()
    w.writerows(fail_rows)

example_rows = []
for key, r in sorted(examples.items()):
    example_rows.append({
        "dataset": key[0],
        "mode": key[1],
        "precision_mode": key[2],
        "failure_class": key[3],
        "example_tag": r["tag"],
        "example_error": r["error_message_short"],
        "json_path": r["json_path"],
    })

with open(FAIL_EXAMPLES_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["dataset", "mode", "precision_mode", "failure_class", "example_tag", "example_error", "json_path"],
    )
    w.writeheader()
    w.writerows(example_rows)

print(f"Wrote {RAW_CSV} with {len(rows)} rows")
print(f"Wrote {PAIR_CSV} with {len(pair_rows)} rows")
print(f"Wrote {FAIL_CSV} with {len(fail_rows)} rows")
print(f"Wrote {FAIL_EXAMPLES_CSV} with {len(example_rows)} rows")

print("\n=== Status counts ===")
for k, v in sorted(Counter((r["precision_mode"], r["status"]) for r in rows).items()):
    print(k, v)

print("\n=== Failure classes ===")
for k, v in sorted(Counter(r["failure_class"] for r in rows if r["status"] != "success").items()):
    print(k, v)
