from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("results/artifacts/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_resolution_area(x):
    if pd.isna(x):
        return None
    try:
        v = ast.literal_eval(str(x))
        if isinstance(v, list):
            if len(v) == 1:
                return int(v[0])
            if len(v) == 2:
                return int(v[0]) * int(v[1])
    except Exception:
        return None
    return None


def savefig(name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def plot_resolution_scaling():
    path = Path("results/artifacts/paper_fno_resolution_scaling_table.csv")
    if not path.exists():
        print(f"[WARN] missing {path}")
        return

    df = pd.read_csv(path)
    df["resolution_area"] = df["resolution"].apply(parse_resolution_area)
    df["median_ms"] = pd.to_numeric(df["median_ms"], errors="coerce")

    for dataset in sorted(df["dataset"].dropna().unique()):
        sub = df[df["dataset"] == dataset].dropna(subset=["resolution_area", "median_ms"])
        if sub.empty:
            continue
        sub = sub.sort_values("resolution_area")

        plt.figure(figsize=(6.5, 4.0))
        plt.plot(sub["resolution_area"], sub["median_ms"], marker="o")
        plt.xlabel("Input resolution size")
        plt.ylabel("Median latency (ms)")
        plt.title(f"FNO resolution scaling on Jetson: {dataset}")
        plt.grid(True, alpha=0.3)
        savefig(f"fno_resolution_scaling_{dataset}")


def plot_backend_speedup():
    path = Path("results/artifacts/paper_fno_backend_speedup_table.csv")
    if not path.exists():
        print(f"[WARN] missing {path}")
        return

    df = pd.read_csv(path)
    df["torchscript_speedup_vs_eager"] = pd.to_numeric(df["torchscript_speedup_vs_eager"], errors="coerce")
    df = df.dropna(subset=["torchscript_speedup_vs_eager"])
    if df.empty:
        return

    # Keep most interpretable cases
    df = df.sort_values("torchscript_speedup_vs_eager", ascending=False).head(20)

    plt.figure(figsize=(8.0, 5.0))
    plt.barh(df["case"], df["torchscript_speedup_vs_eager"])
    plt.xlabel("TorchScript speedup over eager")
    plt.ylabel("FNO case")
    plt.title("Backend effect for FNO deployment on Jetson")
    plt.gca().invert_yaxis()
    savefig("fno_backend_speedup_top20")


def plot_energy():
    path = Path("results/artifacts/paper_fno_long_energy_table.csv")
    if not path.exists():
        print(f"[WARN] missing {path}")
        return

    df = pd.read_csv(path)
    df["energy_per_inf_j"] = pd.to_numeric(df["energy_per_inf_j"], errors="coerce")
    df = df.dropna(subset=["energy_per_inf_j"])
    if df.empty:
        return

    df = df.sort_values("energy_per_inf_j")

    plt.figure(figsize=(8.0, 5.0))
    plt.barh(df["tag"], df["energy_per_inf_j"])
    plt.xlabel("Energy per inference (J)")
    plt.ylabel("FNO case")
    plt.title("Long-run energy per inference on Jetson")
    plt.gca().invert_yaxis()
    savefig("fno_long_energy_per_inference")


def plot_precision_tf32():
    path = Path("results/artifacts/paper_fno_precision_tf32_table.csv")
    if not path.exists():
        print(f"[WARN] missing {path}")
        return

    df = pd.read_csv(path)
    # Try common possible column names
    speed_cols = [c for c in df.columns if "speedup" in c.lower()]
    if not speed_cols:
        print("[WARN] no speedup column in TF32 table")
        return

    speed_col = speed_cols[0]
    df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce")
    df = df.dropna(subset=[speed_col])
    if df.empty:
        return

    label_col = "tag" if "tag" in df.columns else df.columns[0]
    df = df.sort_values(speed_col, ascending=False).head(20)

    plt.figure(figsize=(8.0, 5.0))
    plt.barh(df[label_col].astype(str), df[speed_col])
    plt.xlabel(speed_col)
    plt.ylabel("FNO case")
    plt.title("TF32 vs FP32 strict precision effect")
    plt.gca().invert_yaxis()
    savefig("fno_tf32_vs_fp32_speedup")


def main():
    plot_resolution_scaling()
    plot_backend_speedup()
    plot_energy()
    plot_precision_tf32()


if __name__ == "__main__":
    main()
