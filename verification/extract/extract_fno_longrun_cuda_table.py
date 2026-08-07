#!/usr/bin/env python3
import json
import csv
from pathlib import Path

ROOT = Path.home() / "jjyoo3" / "EDCNO"
OUT = Path.home() / "jjyoo3" / "fno_longrun_cuda_table_values.txt"

TARGETS = [
    ("Burgers base @2048", "fp32_strict", ["burgers", "base", "2048", "fp32"]),
    ("Burgers base @2048", "tf32", ["burgers", "base", "2048", "tf32"]),
    ("Darcy base @85x85", "fp32_strict", ["darcy", "base", "85", "fp32"]),
    ("Darcy base @141x141", "fp32_strict", ["darcy", "base", "141", "fp32"]),
    ("Darcy base @141x141", "tf32", ["darcy", "base", "141", "tf32"]),
    ("Darcy base @211x211", "fp32_strict", ["darcy", "base", "211", "fp32"]),
    ("Darcy base @281x281", "fp32_strict", ["darcy", "base", "281", "fp32"]),
    ("Darcy base @421x421 frontier", "fp32_strict", ["darcy", "base", "421", "fp32"]),
    ("Darcy base @421x421 frontier", "tf32", ["darcy", "base", "421", "tf32"]),
    ("Darcy large @141x141", "fp32_strict", ["darcy", "large", "141", "fp32"]),
    ("Darcy large @421x421 frontier", "fp32_strict", ["darcy", "large", "421", "fp32"]),
    ("Darcy large @421x421 frontier", "tf32", ["darcy", "large", "421", "tf32"]),
]

def flatten(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if len(obj) <= 10 and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj
        else:
            for i, v in enumerate(obj[:100]):
                flatten(v, f"{prefix}.{i}" if prefix else str(i), v)
    else:
        out[prefix] = obj
    return out

def load_json(p):
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        return flatten(obj)
    except Exception:
        return None

def load_csv(p):
    rows = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for r in csv.DictReader(f):
                rows.append({str(k): v for k, v in r.items() if k is not None})
    except Exception:
        pass
    return rows

def row_text(row):
    return " ".join(f"{k}={v}" for k, v in row.items()).lower()

def as_float(v):
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return None

def get_value(row, keys):
    for k, v in row.items():
        lk = k.lower()
        if all(term in lk for term in keys):
            fv = as_float(v)
            if fv is not None:
                return fv, k
    return None, None

def get_cuda(row):
    candidates = []
    for k, v in row.items():
        lk = k.lower()
        if "cuda" in lk and "alloc" in lk and "reserved" not in lk:
            fv = as_float(v)
            if fv is not None:
                candidates.append((k, fv))
    return candidates[0] if candidates else (None, None)

def get_board(row):
    for k, v in row.items():
        lk = k.lower()
        if ("ram_used_peak" in lk or "peak_ram" in lk or ("ram" in lk and "peak" in lk)) and "swap" not in lk:
            fv = as_float(v)
            if fv is not None:
                return k, fv
    return None, None

def main():
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".json", ".csv"]:
            name = str(p).lower()
            if any(x in name for x in ["long", "energy", "sustained", "tegrastats", "summary", "paper"]):
                files.append(p)

    rows = []
    for p in files:
        if p.suffix.lower() == ".json":
            r = load_json(p)
            if r:
                rows.append((p, r))
        else:
            for r in load_csv(p):
                rows.append((p, r))

    with OUT.open("w", encoding="utf-8") as f:
        f.write("Case | Precision | CUDA MB | Board RAM MB | Source\n")
        f.write("-" * 120 + "\n")
        for case, precision, terms in TARGETS:
            best = None
            best_score = -1
            for p, r in rows:
                t = (row_text(r) + " " + str(p).lower())
                score = sum(1 for term in terms if term in t)
                if precision.replace("_strict", "") in t:
                    score += 2
                if "peak_cuda" in t or "cuda_alloc" in t:
                    score += 2
                if score > best_score:
                    best_score = score
                    best = (p, r)

            if best is None:
                f.write(f"{case} | {precision} | MISSING | MISSING | NO MATCH\n")
                continue

            p, r = best
            cuda_k, cuda_v = get_cuda(r)
            board_k, board_v = get_board(r)
            cuda_s = "MISSING" if cuda_v is None else f"{cuda_v:.2f}"
            board_s = "MISSING" if board_v is None else f"{board_v:.0f}"
            f.write(f"{case} | {precision} | {cuda_s} | {board_s} | {p}\n")
            if cuda_k or board_k:
                f.write(f"    keys: cuda={cuda_k}, board={board_k}\n")

    print(f"WROTE {OUT}")
    print(f"cat {OUT}")

if __name__ == "__main__":
    main()
