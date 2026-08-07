"""Common style setup for all figures in the deployment benchmark paper.

Style targets: journal-quality (Nature/IEEE Trans.), clean sans-serif,
light grid, categorical bright colors, log scales where appropriate.
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams


def setup_style():
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.size"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["axes.labelsize"] = 10
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 9
    rcParams["axes.linewidth"] = 0.8
    rcParams["xtick.major.width"] = 0.8
    rcParams["ytick.major.width"] = 0.8
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.grid"] = True
    rcParams["grid.linewidth"] = 0.4
    rcParams["grid.alpha"] = 0.5
    rcParams["grid.color"] = "#B0B0B0"
    rcParams["legend.frameon"] = False
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150
    rcParams["savefig.bbox"] = "tight"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


# Family colors -- validated with the dataviz skill's validate_palette.js
# (light mode, all checks pass). Deliberately a different hue quartet from
# PRECISION_COLORS below (yellow/green/magenta/orange vs blue/aqua/violet/
# red) since family and precision now co-occur as two separate encodings
# in the same figure (family = group label/divider, precision = bar fill)
# and must stay visually distinguishable from each other.
FAMILY_COLORS = {
    "FNO":      "#eda100",  # yellow
    "DeepONet": "#008300",  # green
    "WNO":      "#e87ba4",  # magenta
    "Sp2GNO":   "#eb6834",  # orange
}

# Precision colors -- FP32 is the trusted reference/baseline and is
# deliberately neutral gray (no hue), exempt from the chroma-floor check
# by the same convention as KERNEL_COLORS_UNIFIED's "Other". The four
# reduced-precision paths get validated categorical hues (dataviz skill's
# validate_palette.js, light mode: lightness band [0.43,0.77], chroma
# floor >=0.1, worst adjacent CVD dE 21.6 -- well clear of the 12 floor).
PRECISION_COLORS = {
    "FP32":         "#4d4d4d",  # neutral gray -- reference, not a "series"
    "TF32":         "#2a78d6",  # blue
    "BF16 auto":    "#1baf7a",  # aqua
    "FP16 auto":    "#4a3aa7",  # violet
    "FP16 native":  "#e34948",  # red -- most aggressive quantization
}

# Status colors for heatmap
STATUS_COLORS = {
    "success":  "#2ca02c",
    "partial":  "#f0c419",
    "fail":     "#c0392b",
}

# Kernel-class colors for stacked-bar mechanism plots
KERNEL_COLORS = {
    # DeepONet class labels
    "Dense/GEMM":         "#1f4e79",
    "Activation":         "#d95f02",
    "Materialization":    "#7b3294",
    "Interp./Cast":       "#2ca02c",
    "Other":              "#7f7f7f",
    # FNO class labels
    "FFT/IFFT":           "#1f4e79",
    "Dense":              "#d95f02",
    "Fused elementwise":  "#2ca02c",
    "Movement":           "#7b3294",
}

# Unified 7-class taxonomy shared across FNO / DeepONet / WNO / Sp2GNO
# (used by both the nsys class-share figure, as fill color, and the ncu
# roofline figure, as marker-edge color, so "class" reads as the same
# color in both figures -- occupancy in the roofline figure is a separate,
# orthogonal encoding carried by marker *fill*, never by this palette).
# Validated with the dataviz skill's validate_palette.js (light mode):
# the 6 hued classes pass lightness band [0.43,0.77], chroma floor >=0.1,
# and CVD separation (worst adjacent dE 24.2); "Other" is a deliberate
# neutral gray and is exempt from the chroma-floor check by design.
KERNEL_COLORS_UNIFIED = {
    "FFT/Spectral":             "#2a78d6",  # blue
    "Conv/Wavelet":             "#1baf7a",  # aqua
    "Dense/GEMM":               "#eda100",  # yellow
    "Activation/Elementwise":   "#008300",  # green
    "Movement/Materialization": "#4a3aa7",  # violet
    "Reduction/Norm":           "#e34948",  # red
    "Other":                    "#7f7f7f",  # gray (neutral, no hue)
}


def save_fig(fig, name, outdir="/home/jetson/jjyoo3/edge_figures/figures"):
    """Save figure as both PNG (150 DPI for viewing) and PDF (vector)."""
    fig.savefig(f"{outdir}/{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight",
                facecolor="white")
    # also save a 400-DPI print-quality version for the journal
    fig.savefig(f"{outdir}/{name}_400dpi.png", dpi=400, bbox_inches="tight",
                facecolor="white")
    print(f"  saved: {name}.png (150dpi), {name}.pdf, {name}_400dpi.png")
