from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path("artifacts/summaries")
AGG_DIR = Path("artifacts/aggregates")
CKPT_DIR = Path("artifacts/checkpoints")
OUT_CSV = Path("results/artifacts/fno_quality_validity_table.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


ERROR_KEYS = [
    "test_rel_l2",
    "test_relative_l2",
    "rel_l2",
    "relative_l2",
    "test_loss",
    "best_val_loss",
    "val_loss",
    "best_metric",
]

PARAM_KEYS = [
    "parameter_count",
    "num_parameters",
    "params",
]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_key_recursive(obj: Any, keys: list[str]):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        for v in obj.values():
            found = find_first_key_recursive(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_first_key_recursive(v, keys)
            if found is not None:
                return found
    return None


def parse_name(name: str) -> dict[str, Any]:
    # Examples:
    # burgers_fno_base_seed3_summary
    # darcy_fno_base_r281_seed1_summary
    # burgers_fno_base_modes12_seed1_summary
    stem = name
    stem = stem.replace("_summary", "")
    stem = stem.replace("_aggregate_summary", "")

    dataset = stem.split("_")[0] if "_" in stem else None

    scale = None
    for s in ["small", "base", "large"]:
        if f"_fno_{s}" in stem:
            scale = s

    seed = None
    m = re.search(r"_seed(\d+)", stem)
    if m:
        seed = int(m.group(1))

    resolution = None
    r = re.search(r"_r(\d+)", stem)
    if r:
        rr = int(r.group(1))
        if dataset == "darcy":
            resolution = f"[{rr}, {rr}]"
        else:
            resolution = f"[{rr}]"
    elif dataset == "burgers":
        resolution = "[2048]"
    elif dataset == "darcy":
        resolution = "[141, 141]"

    variant = "baseline"
    for key in ["modes12", "modes16", "modes24", "modes32", "nocoords", "pad0", "pad15", "pad40"]:
        if key in stem:
            variant = key
    if "_r" in stem:
        variant = "resolution_scaling"

    return {
        "experiment_name": stem,
        "dataset": dataset,
        "scale": scale,
        "resolution_inferred": resolution,
        "variant": variant,
        "seed": seed,
    }


def main():
    rows = []

    for path in sorted(SUMMARY_DIR.glob("*_summary.json")):
        d = load_json(path)
        meta = parse_name(path.stem)

        quality = find_first_key_recursive(d, ERROR_KEYS)
        params = find_first_key_recursive(d, PARAM_KEYS)

        # Try checkpoint for parameter count if summary did not include it.
        ckpt_name = meta["experiment_name"] + "_best.pt"
        ckpt_path = CKPT_DIR / ckpt_name
        ckpt_exists = ckpt_path.exists()

        rows.append({
            **meta,
            "summary_path": str(path),
            "checkpoint_path_inferred": str(ckpt_path) if ckpt_exists else "",
            "checkpoint_exists": ckpt_exists,
            "parameter_count_summary": params,
            "quality_metric_value": quality,
            "quality_metric_source_keys": "|".join(ERROR_KEYS),
            "included_for_deployment": ckpt_exists,
            "notes": "Use this as validity gate; verify exact metric name from training config before final manuscript.",
        })

    if not rows:
        raise SystemExit(f"No summary JSON found in {SUMMARY_DIR}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
