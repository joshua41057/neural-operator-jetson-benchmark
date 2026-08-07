from __future__ import annotations

import csv
import json
import re
from pathlib import Path

OUTDIR = Path("results/jetson_fno_profile_nsys")
SUMMARY_CSV = OUTDIR / "nsys_profile_summary.csv"
KERNEL_CSV = OUTDIR / "nsys_top_kernels.csv"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_nvtx_forward_avg_ns(text: str):
    for line in text.splitlines():
        if ":forward" in line:
            nums = re.findall(r"[\d,]+\.\d|[\d,]+", line)
            # Expected columns:
            # Time(%) Total Time(ns) Instances Avg(ns) Med(ns) Min(ns) Max(ns) StdDev(ns)
            if len(nums) >= 8:
                return float(nums[3].replace(",", ""))
    return None


def parse_cuda_api_sync_pct(text: str):
    in_cuda_api = False
    for line in text.splitlines():
        if "CUDA API Summary" in line or "cuda_api_sum" in line:
            in_cuda_api = True
            continue
        if in_cuda_api:
            if "Processing [" in line and "cuda_api_sum" not in line:
                break
            if "cudaDeviceSynchronize" in line:
                m = re.match(r"\s*([\d\.]+)\s+", line)
                if m:
                    return float(m.group(1))
    return None


def parse_mem_mb(text: str, op_name: str):
    in_mem_size = False
    for line in text.splitlines():
        if "CUDA GPU MemOps Summary (by Size)" in line or "cuda_gpu_mem_size_sum" in line:
            in_mem_size = True
            continue
        if in_mem_size:
            if "Processing [" in line and "cuda_gpu_mem_size_sum" not in line:
                break
            if op_name in line:
                nums = re.findall(r"[\d,]+\.\d|[\d,]+", line)
                # Total(MB), Count, Avg(MB), ...
                if len(nums) >= 3:
                    total_mb = float(nums[0].replace(",", ""))
                    count = int(float(nums[1].replace(",", "")))
                    avg_mb = float(nums[2].replace(",", ""))
                    return total_mb, count, avg_mb
    return None, None, None


def parse_top_kernels(text: str, max_rows: int = 8):
    rows = []
    in_kernel_section = False
    seen_table_header = False

    for line in text.splitlines():
        # Enter only when the actual kernel-summary heading appears
        if "CUDA GPU Kernel Summary" in line:
            in_kernel_section = True
            seen_table_header = False
            continue

        if not in_kernel_section:
            continue

        # If another report starts, stop
        if "Processing [" in line and "cuda_gpu_kern_sum" not in line:
            break

        stripped = line.strip()

        # Skip blank lines before/inside header
        if not stripped:
            continue

        # Skip decorative/header lines until real rows start
        if stripped.startswith("Time (%)"):
            seen_table_header = True
            continue
        if stripped.startswith("--------"):
            continue
        if stripped.startswith("**"):
            continue

        if not seen_table_header:
            continue

        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 9:
            continue

        try:
            time_pct = float(parts[0])
            total_ns = float(parts[1].replace(",", ""))
            instances = int(float(parts[2].replace(",", "")))
            avg_ns = float(parts[3].replace(",", ""))
            kernel_name = parts[-1]
        except Exception:
            continue

        rows.append(
            {
                "time_pct": time_pct,
                "total_ns": total_ns,
                "instances": instances,
                "avg_ns": avg_ns,
                "kernel_name": kernel_name,
            }
        )

        if len(rows) >= max_rows:
            break

    return rows


def main():
    summary_rows = []
    kernel_rows = []

    for json_path in sorted(OUTDIR.glob("*.json")):
        tag = json_path.stem
        stats_path = OUTDIR / f"{tag}_nsys_stats.txt"
        if not stats_path.exists():
            print(f"[WARN] Missing stats file for {tag}: {stats_path}")
            continue

        j = read_json(json_path)
        stats_text = stats_path.read_text(encoding="utf-8", errors="ignore")

        forward_avg_ns = parse_nvtx_forward_avg_ns(stats_text)
        sync_pct = parse_cuda_api_sync_pct(stats_text)
        d2d_total_mb, d2d_count, d2d_avg_mb = parse_mem_mb(
            stats_text, "[CUDA memcpy Device-to-Device]"
        )
        h2d_total_mb, h2d_count, h2d_avg_mb = parse_mem_mb(
            stats_text, "[CUDA memcpy Host-to-Device]"
        )

        summary_rows.append(
            {
                "tag": tag,
                "dataset": j.get("dataset"),
                "resolution": j.get("resolution"),
                "mode": j.get("mode"),
                "precision": j.get("precision"),
                "mean_ms": j.get("mean_ms"),
                "p95_ms": j.get("p95_ms"),
                "cuda_peak_allocated_mb": j.get("cuda_peak_allocated_mb"),
                "forward_avg_ms_from_nvtx": None if forward_avg_ns is None else forward_avg_ns / 1e6,
                "cudaDeviceSynchronize_time_pct": sync_pct,
                "d2d_total_mb": d2d_total_mb,
                "d2d_count": d2d_count,
                "d2d_avg_mb": d2d_avg_mb,
                "h2d_total_mb": h2d_total_mb,
                "h2d_count": h2d_count,
                "h2d_avg_mb": h2d_avg_mb,
                "input_bank": j.get("input_bank"),
                "source": j.get("source"),
            }
        )

        top_kernels = parse_top_kernels(stats_text, max_rows=8)
        if not top_kernels:
            print(f"[WARN] No kernels parsed for {tag}")

        for rank, row in enumerate(top_kernels, start=1):
            kernel_rows.append(
                {
                    "tag": tag,
                    "rank": rank,
                    **row,
                }
            )

    if not summary_rows:
        raise RuntimeError(f"No summary rows parsed from {OUTDIR}")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if kernel_rows:
        with open(KERNEL_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(kernel_rows[0].keys()))
            writer.writeheader()
            writer.writerows(kernel_rows)
    else:
        print(f"[WARN] No kernel rows parsed; skipping {KERNEL_CSV}")

    print(f"Wrote {SUMMARY_CSV}")
    if kernel_rows:
        print(f"Wrote {KERNEL_CSV}")


if __name__ == "__main__":
    main()