#!/usr/bin/env python3
import re
import json
import csv
from pathlib import Path
from typing import Any, Dict, List

ROOTS = [
    Path.home() / "jjyoo3" / "EDCNO",
    Path.home() / "jjyoo3" / "EDCNO_DeepONet",
]

OUT = Path.home() / "jjyoo3" / "memory_values_for_main_and_appendix_tables.txt"

FILE_HINTS = [
    "sustained", "energy", "tegrastats", "memory", "mem",
    "frontier", "query", "long", "summary", "paper",
    "cuda", "board", "ram", "alloc"
]

TARGET_PAT = re.compile(
    r"(burgers.*base.*2048|darcy.*base.*141|darcy.*base.*281|"
    r"701|841|981|131072|131,072|chunk|4096|query|frontier|"
    r"fp32|tf32|bf16|fp16|native|autocast)",
    re.IGNORECASE,
)

KEEP_COL_PAT = re.compile(
    r"(case|family|task|model|precision|mode|policy|backend|resolution|res|q|query|chunk|"
    r"median|med|mean|p95|p99|latency|ms|energy|j/inf|joule|power|w|vdd|"
    r"cuda|alloc|memory|mem|board|ram|tegrastats|temp|status|duration|inference|rate)",
    re.IGNORECASE,
)

MEM_COL_PAT = re.compile(
    r"(cuda|alloc|memory|mem|board|ram|tegrastats|peak|resident|rss|vdd|power|energy)",
    re.IGNORECASE,
)

def safe_key(k: Any) -> str:
    if k is None:
        return "__NONE__"
    return str(k)

def safe_val(v: Any) -> str:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return "" if v is None else str(v)

def normalize_row(row: Dict[Any, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        sk = safe_key(k)
        if sk == "__NONE__":
            # Extra CSV fields without headers sometimes appear under None.
            sk = "extra_fields"
        out[sk] = v
    return out

def looks_relevant_file(p: Path) -> bool:
    name = str(p).lower()
    return (
        p.suffix.lower() in [".csv", ".json", ".jsonl", ".txt"]
        and any(h in name for h in FILE_HINTS)
    )

def flatten(prefix: str, obj: Any, out: Dict[str, Any]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            sk = safe_key(k)
            flatten(f"{prefix}.{sk}" if prefix else sk, v, out)
    elif isinstance(obj, list):
        if len(obj) <= 12 and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj
        else:
            for i, v in enumerate(obj[:200]):
                flatten(f"{prefix}.{i}" if prefix else str(i), v, out)
    else:
        out[prefix] = obj

def row_text(row: Dict[str, Any]) -> str:
    return " ".join(f"{safe_key(k)}={safe_val(v)}" for k, v in row.items()).lower()

def has_memory_or_energy_cols(row: Dict[str, Any]) -> bool:
    return any(MEM_COL_PAT.search(safe_key(k)) for k in row.keys())

def is_target_row(row: Dict[str, Any]) -> bool:
    txt = row_text(row)
    return TARGET_PAT.search(txt) is not None and has_memory_or_energy_cols(row)

def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    kept = {}
    for k, v in row.items():
        sk = safe_key(k)
        if KEEP_COL_PAT.search(sk):
            kept[sk] = v
    return kept

def read_csv(p: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            for r in reader:
                rows.append(normalize_row(dict(r)))
    except Exception as e:
        rows.append({"read_error": str(e), "file": str(p)})
    return rows

def read_json_any(p: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return rows

        if p.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                flat = {}
                flatten("", obj, flat)
                rows.append(normalize_row(flat))
        else:
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    flat = {}
                    flatten("", item, flat)
                    rows.append(normalize_row(flat))
            else:
                flat = {}
                flatten("", obj, flat)
                rows.append(normalize_row(flat))
    except Exception as e:
        rows.append({"read_error": str(e), "file": str(p)})
    return rows

def read_txt_grep(p: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            low = line.lower()
            if MEM_COL_PAT.search(low) and TARGET_PAT.search(low):
                rows.append({"line": i, "text": line})
    except Exception as e:
        rows.append({"read_error": str(e), "file": str(p)})
    return rows

def main():
    files = []
    for root in ROOTS:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and looks_relevant_file(p):
                    files.append(p)

    results = []

    for p in sorted(files):
        suffix = p.suffix.lower()
        if suffix == ".csv":
            rows = read_csv(p)
        elif suffix in [".json", ".jsonl"]:
            rows = read_json_any(p)
        else:
            rows = read_txt_grep(p)

        hits = []
        for r in rows:
            r = normalize_row(r)
            if is_target_row(r):
                c = compact_row(r)
                if c:
                    hits.append(c)

        if hits:
            results.append((p, hits[:120]))

    with OUT.open("w", encoding="utf-8") as f:
        f.write("# Memory candidate rows for paper tables\n")
        f.write("# Goal: fill BOTH CUDA MB and Board RAM MB for sustained and frontier tables.\n")
        f.write("# Search roots:\n")
        for r in ROOTS:
            f.write(f"#   {r}\n")
        f.write("\n")

        for p, hits in results:
            f.write("\n" + "#" * 100 + "\n")
            f.write(f"# FILE: {p}\n")
            f.write("#" * 100 + "\n")
            for idx, h in enumerate(hits, 1):
                f.write(f"\n[ROW {idx}]\n")
                for k, v in h.items():
                    f.write(f"{k}: {safe_val(v)}\n")

    print(f"WROTE {OUT}")
    print(f"Files scanned: {len(files)}")
    print(f"Files with hits: {len(results)}")

if __name__ == "__main__":
    main()
