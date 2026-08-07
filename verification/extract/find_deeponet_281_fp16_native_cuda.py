#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path.home() / "jjyoo3" / "EDCNO_DeepONet"
OUT = Path.home() / "jjyoo3" / "deeponet_281_fp16_native_cuda_result.txt"

FILE_HINTS = re.compile(
    r"(long|sustained|energy|tegrastats|memory|mem|summary|paper|precision|deployment)",
    re.IGNORECASE,
)

KEY_HINTS = re.compile(
    r"(case|family|task|model|resolution|res|backend|precision|mode|"
    r"median|med|p95|energy|j|joule|inf|cuda|alloc|memory|mem|board|ram|"
    r"tegrastats|temp|power|w|latency|ms)",
    re.IGNORECASE,
)

CUDA_KEY = re.compile(r"(peak.*cuda.*alloc|cuda.*alloc|peak_cuda|allocated.*mb|max.*memory)", re.IGNORECASE)
BOARD_KEY = re.compile(r"(ram_used_peak|board.*ram|peak.*ram|tegrastats.*ram|ram.*peak)", re.IGNORECASE)

def flatten(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            k = str(k)
            flatten(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj
        else:
            for i, v in enumerate(obj[:200]):
                flatten(v, f"{prefix}.{i}" if prefix else str(i), out)
    else:
        out[prefix] = obj
    return out

def as_float(x):
    if x is None:
        return None
    s = str(x).strip()
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def load_rows(path):
    rows = []
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader, 1):
                    rows.append((i, {str(k): v for k, v in r.items() if k is not None}))
        elif path.suffix.lower() in [".json", ".jsonl"]:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return rows
            if path.suffix.lower() == ".jsonl":
                for i, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        rows.append((i, flatten(json.loads(line))))
            else:
                obj = json.loads(text)
                if isinstance(obj, list):
                    for i, item in enumerate(obj, 1):
                        rows.append((i, flatten(item)))
                else:
                    rows.append((1, flatten(obj)))
    except Exception:
        pass
    return rows

def compact(row):
    return {k: v for k, v in row.items() if KEY_HINTS.search(str(k))}

def text_of(row):
    return " ".join(f"{k}={v}" for k, v in row.items()).lower()

def score_row(row):
    t = text_of(row)
    score = 0

    # Semantic identifiers
    if "darcy" in t:
        score += 10
    if "281" in t:
        score += 10
    if "fp16" in t and "native" in t:
        score += 15
    elif "fp16_native" in t or "native_fp16" in t:
        score += 15
    if "torchscript" in t:
        score += 5
    if "base" in t:
        score += 3

    # Exact row fingerprints from the table
    for v in row.values():
        fv = as_float(v)
        if fv is None:
            continue
        if abs(fv - 10.550) < 0.02:
            score += 20
        if abs(fv - 11.802) < 0.02:
            score += 20
        if abs(fv - 0.2534) < 0.002:
            score += 25
        if abs(fv - 4456) < 2:
            score += 25
        if abs(fv - 21.364) < 0.05:
            score += 10

    # Memory keys
    if any(CUDA_KEY.search(k) for k in row.keys()):
        score += 20
    if any(BOARD_KEY.search(k) for k in row.keys()):
        score += 10

    return score

def find_cuda_value(row):
    candidates = []
    for k, v in row.items():
        if CUDA_KEY.search(k):
            fv = as_float(v)
            if fv is not None:
                candidates.append((k, fv))
    return candidates

def find_board_value(row):
    candidates = []
    for k, v in row.items():
        if BOARD_KEY.search(k):
            fv = as_float(v)
            if fv is not None:
                candidates.append((k, fv))
    return candidates

def main():
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".csv", ".json", ".jsonl"]:
            if FILE_HINTS.search(str(p)):
                files.append(p)

    hits = []
    for p in files:
        for idx, row in load_rows(p):
            s = score_row(row)
            if s >= 45:
                hits.append((s, p, idx, compact(row), find_cuda_value(row), find_board_value(row)))

    hits.sort(key=lambda x: x[0], reverse=True)

    with OUT.open("w", encoding="utf-8") as f:
        f.write("TARGET ROW:\n")
        f.write("DeepONet Darcy base @281x281 FP16 native sustained run\n")
        f.write("Expected fingerprints: median 10.550, P95 11.802, J/inf 0.2534, board RAM 4456\n\n")

        if not hits:
            f.write("NO MATCH FOUND.\n")
            f.write("Try searching manually for files containing 0.2534 or 4456.\n")
        else:
            for rank, (s, p, idx, row, cuda_vals, board_vals) in enumerate(hits[:15], 1):
                f.write("=" * 100 + "\n")
                f.write(f"RANK {rank} | SCORE {s} | FILE {p} | ROW {idx}\n")
                f.write("- CUDA candidates:\n")
                for k, v in cuda_vals:
                    f.write(f"  {k}: {v:.6f}\n")
                f.write("- Board RAM candidates:\n")
                for k, v in board_vals:
                    f.write(f"  {k}: {v:.6f}\n")
                f.write("- Compact row:\n")
                for k, v in row.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            # Print best final candidate if available.
            for s, p, idx, row, cuda_vals, board_vals in hits:
                if cuda_vals:
                    # Prefer a peak_cuda_alloc-like key over reserved/cached memory.
                    preferred = None
                    for k, v in cuda_vals:
                        lk = k.lower()
                        if "reserved" in lk:
                            continue
                        if "peak" in lk and "alloc" in lk:
                            preferred = (k, v)
                            break
                    if preferred is None:
                        preferred = cuda_vals[0]

                    f.write("\nFINAL_CANDIDATE\n")
                    f.write(f"FILE={p}\n")
                    f.write(f"ROW={idx}\n")
                    f.write(f"KEY={preferred[0]}\n")
                    f.write(f"CUDA_MB={preferred[1]:.2f}\n")
                    break

    print(f"WROTE {OUT}")
    print("Show result with:")
    print(f"cat {OUT}")

if __name__ == "__main__":
    main()
