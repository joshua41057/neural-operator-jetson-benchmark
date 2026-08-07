"""Figure: nsys kernel-class-share decomposition, all four operator families.

Four panels (FNO, DeepONet, WNO, Sp2GNO), 2x2, square overall aspect.
Each panel is a stacked bar chart of the share of profiled CUDA kernel
time (nsys `cuda_gpu_kern_sum`) occupied by each operation class, across
a representative resolution/capacity sweep for that family.

Source: nsys stats --report cuda_gpu_kern_sum on batch-size-one forward
passes, kernel names grouped into a shared 7-class taxonomy (FFT/Spectral,
Conv/Wavelet, Dense/GEMM, Activation/Elementwise, Movement/Materialization,
Reduction/Norm, Other). Percentages are share of profiled CUDA kernel time,
not wall-clock latency; they are not the sustained batch-size-one deployment
latency reported in the paper's main tables.

Each family's dominant kernel class differs in kind, not just degree:
FNO is FFT/Spectral-dominated, DeepONet is Dense/GEMM-dominated, WNO is
Conv/Wavelet-dominated (57-86%, share growing with resolution -- its
second-largest class shrinks from a roughly even Activation/Movement split
at r2048 to Activation-led at high resolution, as fixed per-call movement
overhead is amortized over a larger convolution), and Sp2GNO is
Movement/Materialization-dominated (38-57% across cases) -- the graph
message-passing path's index_select/scatter/gather kernels for
neighborhood aggregation outweigh its Dense/GEMM (18-39%) and
Activation/Elementwise (13-18%) time, consistent with a graph-structured
architecture whose bottleneck is data movement over the graph rather than
either spectral transforms or dense compute.

The Sp2GNO panel's last three cases are the VirSO Heat Exchanger request
path (full, spectral-only, 2-layer) -- same architecture family, a fixed
real-world CFD mesh rather than a resolution sweep. Full and 2-layer
reproduce the same Movement/Materialization dominance (57.4%, 54.3%) seen
on Burgers/Darcy; spectral-only (spatial message-passing disabled) is the
one case in the whole panel where that class collapses to under 1%, with
Reduction/Norm and Dense/GEMM taking over instead -- a direct
architecture-level control on the same checkpoint family, not just a
different workload.
"""
import matplotlib.pyplot as plt
import numpy as np

from _style import setup_style, save_fig, KERNEL_COLORS_UNIFIED

setup_style()

CLASS_ORDER = [
    "FFT/Spectral", "Conv/Wavelet", "Dense/GEMM", "Activation/Elementwise",
    "Movement/Materialization", "Reduction/Norm", "Other",
]

# ---------------------------------------------------------------------
# Data: nsys cuda_gpu_kern_sum class shares (%), one representative
# resolution/capacity sweep per family. Values verified against
# results/*/profiles/*_nsys/*_nsys_stats.txt for each case. Sp2GNO's
# three "Heat" cases (VirSO Heat Exchanger, full/spectral/2-layer) come
# from /home/jetson/VirSO/For_Jetson/For_Jetson/inference_runs/
# virso_nsys_fixed/*_nsys_stats.txt instead (allocator-cache-enabled
# reruns; see that directory for provenance).
# ---------------------------------------------------------------------

FNO = {
    "cases": ["Burgers\n2048", "Burgers\n4096", "Burgers\n8192",
              "Darcy\n141$^2$", "Darcy\n211$^2$", "Darcy\n421$^2$"],
    "data": {
        "FFT/Spectral":             [38.0, 52.4, 54.9, 30.1, 37.1, 53.5],
        "Dense/GEMM":               [15.1, 11.3, 10.8, 18.2, 15.1,  7.9],
        "Activation/Elementwise":   [38.4, 29.7, 28.4, 51.3, 47.5, 26.4],
        "Movement/Materialization": [ 0.6,  0.4,  0.2,  0.2,  0.2,  0.1],
        "Other":                    [ 8.0,  6.2,  5.5,  0.1,  0.1, 11.8],
    },
}

DEEPONET = {
    "cases": ["Burgers\n2048", "Burgers\n8192", "Darcy\n141$^2$",
              "Darcy\n281$^2$", "Darcy\n421$^2$", "Darcy\n141$^2$ large"],
    "data": {
        "Dense/GEMM":               [80.3, 84.6, 83.1, 82.8, 80.4, 87.9],
        "Activation/Elementwise":   [17.5, 14.4, 16.0, 16.3, 18.9, 11.6],
        "Movement/Materialization": [ 0.5,  0.2,  0.4,  0.2,  0.2,  0.2],
        "Reduction/Norm":                [ 0.3,  0.1,  0.0,  0.0,  0.0,  0.0],
        "Other":                    [ 1.5,  0.5,  0.5,  0.3,  0.4,  0.1],
    },
}

WNO = {
    "cases": ["Burgers\n2048", "Burgers\n4096", "Burgers\n8192",
              "Darcy\n141$^2$", "Darcy\n281$^2$", "Darcy\n421$^2$"],
    "data": {
        "Conv/Wavelet":             [57.0, 64.3, 70.3, 80.6, 81.1, 78.7],
        "Dense/GEMM":               [ 2.9,  2.7,  2.7,  2.7,  3.0,  3.4],
        "Activation/Elementwise":   [21.2, 17.9, 15.5,  9.8, 10.5, 12.9],
        "Movement/Materialization": [19.0, 15.1, 11.5,  6.8,  5.4,  4.9],
    },
}

SP2GNO = {
    "cases": ["Burgers\ns2048", "Burgers\ns4096", "Darcy\nr141$^2$",
              "Darcy\nr211$^2$", "Darcy L\nr141$^2$",
              "Heat\nfull", "Heat\nspectral", "Heat\n2-layer"],
    "data": {
        "Dense/GEMM":               [30.2, 27.5, 24.6, 23.9, 17.9, 22.4, 39.3, 25.1],
        "Activation/Elementwise":   [14.9, 13.5, 13.3, 14.0, 13.9, 15.9, 14.9, 16.0],
        "Movement/Materialization": [37.6, 41.7, 54.3, 53.6, 48.1, 57.4,  0.7, 54.3],
        "Reduction/Norm":           [17.5, 17.4,  7.9,  8.4,  2.6,  4.0, 45.2,  4.7],
        "Other":                    [ 0.0,  0.0,  0.0,  0.0, 17.6,  0.0,  0.0,  0.0],
    },
}

PANELS = [
    ("(a) FNO", FNO),
    ("(b) DeepONet", DEEPONET),
    ("(c) WNO", WNO),
    ("(d) Sp2GNO", SP2GNO),
]


def stacked_bars(ax, cases, data, title, tick_fontsize=8):
    n_cases = len(cases)
    x = np.arange(n_cases)
    bar_w = 0.62
    bottoms = np.zeros(n_cases)

    classes_here = [c for c in CLASS_ORDER if c in data]
    for cls in classes_here:
        vals = np.array(data[cls])
        ax.bar(x, vals, bar_w, bottom=bottoms,
               color=KERNEL_COLORS_UNIFIED[cls],
               edgecolor="white", linewidth=0.8,
               label=cls, zorder=3)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 6.0:
                ax.text(xi, b + v / 2, f"{v:.0f}",
                        ha="center", va="center",
                        color="white", fontsize=7.5,
                        fontweight="bold", zorder=4)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=tick_fontsize)
    ax.set_ylim(0, 100)
    ax.set_title(title, loc="left", pad=6, fontweight="bold", fontsize=10.5)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4, zorder=0)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0)


fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
(ax_fno, ax_deep), (ax_wno, ax_sp2) = axes

for ax, (title, panel) in zip([ax_fno, ax_deep, ax_wno, ax_sp2], PANELS):
    # Sp2GNO now carries 8 cases (vs 6 for the others) after adding the
    # Heat Exchanger cases -- narrower bars need a smaller tick label to
    # avoid adjacent multi-line labels colliding.
    fs = 6.6 if panel is SP2GNO else 8
    stacked_bars(ax, panel["cases"], panel["data"], title, tick_fontsize=fs)

ax_fno.set_ylabel("Share of profiled CUDA kernel time (%)")
ax_wno.set_ylabel("Share of profiled CUDA kernel time (%)")

# One shared legend for all seven classes across the whole figure
handles = [plt.Rectangle((0, 0), 1, 1, color=KERNEL_COLORS_UNIFIED[c])
           for c in CLASS_ORDER]
fig.legend(handles, CLASS_ORDER, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9,
           handlelength=1.4, handleheight=1.0, columnspacing=1.2)

plt.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig, "fig_nsys_class_share_4panel")
plt.close(fig)
print("nsys 4-panel class-share figure done.")
