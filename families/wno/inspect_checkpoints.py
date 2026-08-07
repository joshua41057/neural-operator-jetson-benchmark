#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
import re
from collections import Counter, OrderedDict

import torch


CKPT_DIR = "checkpoints"
RUNS_DIR = "runs"
LOGS_DIR = "logs"


def safe_torch_load(path: Path):
    """
    Load local trusted PyTorch checkpoint on CPU.
    Do not use this on arbitrary untrusted .pth files.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only
        return torch.load(path, map_location="cpu")


def is_tensor_dict(x):
    return isinstance(x, dict) and any(torch.is_tensor(v) for v in x.values())


def normalize_state_dict_keys(sd):
    out = OrderedDict()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue

        kk = str(k)
        for prefix in ("module.", "_orig_mod.", "model."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]

        out[kk] = v
    return out


def extract_state_dict(obj):
    """
    Handles common checkpoint layouts:
      1) raw state_dict
      2) {'state_dict': ...}
      3) {'model_state_dict': ...}
      4) {'model': ...}
      5) nested dict with tensor values
    """
    if is_tensor_dict(obj):
        return normalize_state_dict_keys(obj), "root"

    if isinstance(obj, dict):
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "ema_state_dict",
        ]

        for key in candidate_keys:
            if key in obj and is_tensor_dict(obj[key]):
                return normalize_state_dict_keys(obj[key]), key

        # Fallback: choose nested dict with the largest number of tensor elements
        best_key = None
        best_sd = None
        best_numel = -1

        for key, val in obj.items():
            if is_tensor_dict(val):
                numel = sum(v.numel() for v in val.values() if torch.is_tensor(v))
                if numel > best_numel:
                    best_key = key
                    best_sd = val
                    best_numel = numel

        if best_sd is not None:
            return normalize_state_dict_keys(best_sd), best_key

    if hasattr(obj, "state_dict"):
        return normalize_state_dict_keys(obj.state_dict()), "object.state_dict()"

    raise RuntimeError("Could not find a tensor state_dict in checkpoint")


def parse_filename(path: Path):
    """
    Example filenames:
      sp2gno_burgers_base_s2048.pth
      sp2gno_darcy_base_r141.pth
      wno_burgers_base_r2048.pth
      wno_darcy_large_r141.pth
    """
    stem = path.stem

    m = re.match(
        r"(?P<family>wno|sp2gno)_(?P<problem>burgers|darcy)_(?P<scale>small|base|large)_(?P<tag>[rs])(?P<n>\d+)$",
        stem,
    )

    info = {
        "case": stem,
        "family": None,
        "problem": None,
        "model_scale": None,
        "resolution_tag": None,
        "resolution_n": None,
        "requested_resolution": None,
    }

    if not m:
        return info

    family = m.group("family")
    problem = m.group("problem")
    scale = m.group("scale")
    tag = m.group("tag")
    n = int(m.group("n"))

    if problem == "burgers":
        # 1D Burgers resolution / sequence length / node count
        requested_resolution = str(n)
    elif problem == "darcy":
        # 2D Darcy grid
        requested_resolution = f"{n}x{n}"
    else:
        requested_resolution = str(n)

    info.update({
        "family": family,
        "problem": problem,
        "model_scale": scale,
        "resolution_tag": tag,
        "resolution_n": n,
        "requested_resolution": requested_resolution,
    })

    return info


def is_likely_buffer_key(k):
    """
    state_dict contains both parameters and buffers.
    This excludes common non-trainable buffers.
    In this repo, if no BatchNorm-like buffers exist, param_count_est ~= exact params.
    """
    buffer_suffixes = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
        "total_ops",
        "total_params",
    )

    if k.endswith(buffer_suffixes):
        return True

    lower = k.lower()
    buffer_words = (
        "mask",
        "grid",
        "pos",
        "coord",
        "edge_index",
        "laplacian",
        "eigenvec",
        "eigenval",
    )

    # Be conservative: do not automatically exclude all grids unless you want a strict estimate.
    # Here we only mark them separately, but still report all tensor elements too.
    return any(w in lower for w in buffer_words)


def summarize_state_dict(sd):
    all_tensors = [(k, v) for k, v in sd.items() if torch.is_tensor(v)]

    total_tensor_numel = sum(v.numel() for _, v in all_tensors)
    total_tensor_bytes = sum(v.numel() * v.element_size() for _, v in all_tensors)

    likely_param_tensors = [
        (k, v) for k, v in all_tensors
        if not is_likely_buffer_key(k)
    ]

    param_count_est = sum(v.numel() for _, v in likely_param_tensors)
    param_bytes_est = sum(v.numel() * v.element_size() for _, v in likely_param_tensors)

    dtype_counter = Counter(str(v.dtype).replace("torch.", "") for _, v in all_tensors)

    top = sorted(
        [(k, tuple(v.shape), v.numel(), str(v.dtype).replace("torch.", "")) for k, v in all_tensors],
        key=lambda x: x[2],
        reverse=True,
    )[:8]

    return {
        "num_state_tensors": len(all_tensors),
        "total_tensor_numel": total_tensor_numel,
        "total_tensor_MB": total_tensor_bytes / 1024**2,
        "param_count_est": param_count_est,
        "param_MB_est": param_bytes_est / 1024**2,
        "dtype_summary": ";".join(f"{k}:{v}" for k, v in sorted(dtype_counter.items())),
        "largest_tensors": top,
    }


def flatten_json_for_result(d):
    """
    Pull useful result fields if result.json exists.
    This is intentionally flexible because benchmark result schemas vary.
    """
    if not isinstance(d, dict):
        return {}

    out = {}

    interesting_patterns = [
        "median",
        "p50",
        "p95",
        "mean",
        "lat",
        "ms",
        "param",
        "rel",
        "l2",
        "error",
        "memory",
        "cuda",
        "power",
        "energy",
        "resolution",
    ]

    def visit(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            key_lower = prefix.lower()
            if any(p in key_lower for p in interesting_patterns):
                out[prefix] = obj

    visit("", d)
    return out


def load_result_json(root: Path, case: str):
    p = root / RUNS_DIR / case / "result.json"
    if not p.exists():
        return {}, None

    try:
        with open(p, "r") as f:
            d = json.load(f)
        return flatten_json_for_result(d), p
    except Exception as e:
        return {"result_json_error": str(e)}, p


def infer_latency_fields(result_flat):
    """
    Try to normalize latency fields into median_ms and p95_ms.
    """
    median_ms = None
    p95_ms = None

    for k, v in result_flat.items():
        kl = k.lower()
        if not isinstance(v, (int, float)):
            continue

        if median_ms is None and ("median" in kl or "p50" in kl) and ("lat" in kl or "ms" in kl):
            median_ms = float(v)

        if p95_ms is None and "p95" in kl and ("lat" in kl or "ms" in kl):
            p95_ms = float(v)

    return median_ms, p95_ms


def simple_checkpoint_metadata(obj):
    """
    Extract simple non-tensor metadata from checkpoint, if present.
    """
    if not isinstance(obj, dict):
        return {}

    meta = {}

    for k, v in obj.items():
        if k in ("state_dict", "model_state_dict", "model", "net", "network", "module"):
            continue

        if isinstance(v, (int, float, str, bool)) or v is None:
            meta[k] = v
        elif isinstance(v, dict):
            simple = {}
            for kk, vv in v.items():
                if isinstance(vv, (int, float, str, bool)) or vv is None:
                    simple[str(kk)] = vv
            if simple:
                meta[k] = simple

    return meta


def write_markdown(rows, out_md: Path):
    cols = [
        "case",
        "family",
        "problem",
        "model_scale",
        "requested_resolution",
        "param_count_est",
        "total_tensor_numel",
        "total_tensor_MB",
        "dtype_summary",
        "median_ms",
        "p95_ms",
    ]

    with open(out_md, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")

        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = f"{v:.6g}"
                vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root")
    ap.add_argument("--ckpt-dir", type=str, default=CKPT_DIR)
    ap.add_argument("--out-csv", type=str, default="checkpoint_inventory.csv")
    ap.add_argument("--out-md", type=str, default="checkpoint_inventory.md")
    ap.add_argument("--out-detail-json", type=str, default="checkpoint_details.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ckpt_dir = root / args.ckpt_dir

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    rows = []
    details = {}

    for ckpt_path in sorted(ckpt_dir.glob("*.pth")):
        row = parse_filename(ckpt_path)
        row["checkpoint_path"] = str(ckpt_path.relative_to(root))
        row["checkpoint_size_MB"] = ckpt_path.stat().st_size / 1024**2

        try:
            obj = safe_torch_load(ckpt_path)
            sd, sd_source = extract_state_dict(obj)
            summary = summarize_state_dict(sd)

            row["state_dict_source"] = sd_source
            row.update({k: v for k, v in summary.items() if k != "largest_tensors"})

            meta = simple_checkpoint_metadata(obj)
            if meta:
                row["embedded_metadata_json"] = json.dumps(meta, sort_keys=True)

            result_flat, result_path = load_result_json(root, row["case"])
            median_ms, p95_ms = infer_latency_fields(result_flat)

            row["median_ms"] = median_ms if median_ms is not None else ""
            row["p95_ms"] = p95_ms if p95_ms is not None else ""
            row["result_json"] = str(result_path.relative_to(root)) if result_path else ""

            details[row["case"]] = {
                "checkpoint": str(ckpt_path.relative_to(root)),
                "state_dict_source": sd_source,
                "filename_info": parse_filename(ckpt_path),
                "summary": summary,
                "embedded_metadata": meta,
                "result_fields": result_flat,
            }

        except Exception as e:
            row["error"] = repr(e)
            details[row["case"]] = {
                "checkpoint": str(ckpt_path.relative_to(root)),
                "error": repr(e),
            }

        rows.append(row)

    # CSV
    all_cols = []
    for r in rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    with open(root / args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            rr = {}
            for k, v in r.items():
                if isinstance(v, float):
                    rr[k] = f"{v:.10g}"
                else:
                    rr[k] = v
            w.writerow(rr)

    # Markdown table for paper
    write_markdown(rows, root / args.out_md)

    # Detailed JSON for audit
    with open(root / args.out_detail_json, "w") as f:
        json.dump(details, f, indent=2)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_detail_json}")
    print()
    print("Preview:")
    for r in rows:
        print(
            f"{r.get('case'):35s} "
            f"res={str(r.get('requested_resolution')):8s} "
            f"params_est={r.get('param_count_est')} "
            f"median_ms={r.get('median_ms', '')} "
            f"p95_ms={r.get('p95_ms', '')}"
        )


if __name__ == "__main__":
    main()
