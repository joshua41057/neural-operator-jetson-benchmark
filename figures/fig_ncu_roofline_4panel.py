"""Figure: ncu roofline-style kernel-level evidence, all four families.

Four panels (FNO, DeepONet, WNO, Sp2GNO), 2x2, square overall aspect,
matching the layout/tone of fig_nsys_class_share_4panel.py.

Unlike the nsys figure (share of total time), this figure answers a
different question per kernel: is it compute-bound, memory-bound, or
occupancy-limited? Each point is one profiled kernel from
`ncu --set detailed --replay-mode kernel`:

  x = memory throughput %  (gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed)
  y = SM (compute) throughput %  (sm__throughput.avg.pct_of_peak_sustained_elapsed)
  color (fill) = achieved occupancy % (sm__warps_active.avg.pct_of_peak_sustained_active),
          sequential colormap -- a continuous magnitude, not a category
  marker shape + edge color = kernel operation class (categorical identity;
          edge color reuses the exact hex from KERNEL_COLORS_UNIFIED, the same
          palette the nsys class-share figure uses as fill -- so "class" reads
          as the same color in both figures, while occupancy is a separate,
          orthogonal channel carried only by marker fill in this figure)
  marker size = kernel duration in ms (sqrt-scaled)

Points above the y=x diagonal are compute-bound (SM%>Mem%); below it, memory-bound.
Points clustered near the origin are occupancy/latency-limited regardless
of the memory/compute split -- neither pipe is saturated because the
batch-size-one grid is too small to fill the GPU.

FNO's FFT kernels sit high on the memory axis; DeepONet's dominant GEMM
tile sits compute-heavy but at conspicuously low occupancy (dark points);
WNO's conv/wavelet kernels sit compute/cache-bound (low Mem%); Sp2GNO's
kernels are the most scattered of the four, consistent with no single
dominant mechanism.

The Sp2GNO panel also carries the VirSO Heat Exchanger cases (full,
spectral-only, 2-layer -- same architecture family, a fixed real-world CFD
mesh rather than the Burgers/Darcy resolution sweep). Their indexSelect/
scatter-gather points land in the same low-Mem%/high-occupancy region as
the Burgers/Darcy movement kernels; the spectral-only case (spatial
message-passing disabled) contributes no such points at all, since that
class of kernel disappears from its profile entirely -- the same
mechanism-flip visible in fig_nsys_class_share_4panel.py.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _style import setup_style, save_fig, KERNEL_COLORS_UNIFIED

setup_style()

with open("ncu_scatter_data.json") as f:
    DATA = json.load(f)

CLASS_MARKERS = {
    "FFT/Spectral":             "o",
    "Conv/Wavelet":             "s",
    "Dense/GEMM":               "^",
    "Activation/Elementwise":   "D",
    "Movement/Materialization": "v",
    "Reduction/Norm":           "P",
    "Other":                    "X",
}

CMAP = plt.get_cmap("viridis")
OCC_NORM = plt.Normalize(vmin=0, vmax=100)

PANELS = [("(a) FNO", "FNO"), ("(b) DeepONet", "DeepONet"),
          ("(c) WNO", "WNO"), ("(d) Sp2GNO", "sp2gno")]


def size_from_ms(ms_array):
    # sqrt scale so a 100x duration range doesn't produce a 100x area range
    return 25 + 340 * np.sqrt(np.clip(ms_array, 0, None) / max(ms_array.max(), 1e-6))


def draw_panel(ax, points, title):
    # background guide: diagonal + corner labels
    ax.plot([0, 100], [0, 100], color="#999999", linewidth=0.9,
             linestyle="--", zorder=1)
    # Quadrant labels sit in the exact corners where extreme (i.e.
    # interesting) kernels tend to land, so they get a translucent backing
    # box and a zorder above the scatter -- legible even when a marker
    # falls directly behind the text.
    label_kw = dict(fontsize=7.5, color="#777777", style="italic", zorder=5,
                     bbox=dict(facecolor="white", edgecolor="none",
                               alpha=0.72, pad=1.5))
    ax.text(97, 6, "memory-bound", ha="right", va="bottom", **label_kw)
    ax.text(3, 94, "compute-bound", ha="left", va="top", **label_kw)
    ax.text(3, 6, "occupancy-\nlimited", ha="left", va="bottom", **label_kw)

    ms = np.array([p["ms"] for p in points])
    mem = np.array([p["mem"] for p in points])
    sm = np.array([p["sm"] for p in points])
    occ = np.array([p["occ"] for p in points])
    sizes = size_from_ms(ms)

    for cls, marker in CLASS_MARKERS.items():
        idx = [i for i, p in enumerate(points) if p["class"] == cls]
        if not idx:
            continue
        sc = ax.scatter(mem[idx], sm[idx], s=sizes[idx], c=occ[idx],
                         cmap=CMAP, norm=OCC_NORM, marker=marker,
                         edgecolor=KERNEL_COLORS_UNIFIED[cls], linewidth=1.1,
                         alpha=0.92, zorder=3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title, loc="left", pad=6, fontweight="bold", fontsize=10.5)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.35, alpha=0.35, zorder=0)
    return sc


fig = plt.figure(figsize=(9.4, 11.2))
gs = fig.add_gridspec(
    4, 2,
    height_ratios=[1.0, 1.0, 0.22, 0.045],
    hspace=0.30, wspace=0.22,
    top=0.965, bottom=0.02, left=0.09, right=0.98,
)
ax_fno = fig.add_subplot(gs[0, 0])
ax_deep = fig.add_subplot(gs[0, 1])
ax_wno = fig.add_subplot(gs[1, 0])
ax_sp2 = fig.add_subplot(gs[1, 1])
lax = fig.add_subplot(gs[2, :])
cax = fig.add_subplot(gs[3, :])
lax.axis("off")

last_sc = None
for ax, (title, proj) in zip([ax_fno, ax_deep, ax_wno, ax_sp2], PANELS):
    last_sc = draw_panel(ax, DATA[proj], title)

ax_fno.set_ylabel("SM (compute) throughput %")
ax_wno.set_ylabel("SM (compute) throughput %")
ax_wno.set_xlabel("Memory throughput %")
ax_sp2.set_xlabel("Memory throughput %")

# shared marker-shape legend for kernel class, in its own reserved row.
# Edge color matches KERNEL_COLORS_UNIFIED (same hex as the nsys figure's
# bar fills) so class identity reads consistently across both figures;
# face stays neutral since fill is reserved for occupancy in the panels.
legend_handles = [
    Line2D([0], [0], marker=m, color="#555555", linestyle="none",
           markerfacecolor="#e8e8e8", markeredgecolor=KERNEL_COLORS_UNIFIED[cls],
           markeredgewidth=1.3, markersize=7.5, label=cls)
    for cls, m in CLASS_MARKERS.items()
]
class_legend = lax.legend(handles=legend_handles, loc="upper center", ncol=4,
                           frameon=False, fontsize=8.5,
                           handletextpad=0.5, columnspacing=1.1,
                           bbox_to_anchor=(0.5, 1.28))
lax.add_artist(class_legend)

# size legend for kernel duration (ms), same sqrt scaling as the panels
all_ms = np.concatenate([[p["ms"] for p in pts] for pts in DATA.values()])
ms_max = all_ms.max()
ref_ms = [0.05, 0.5, 5.0]
size_handles = [
    Line2D([0], [0], marker="o", color="#555555", linestyle="none",
           markerfacecolor="none", markeredgecolor="#555555",
           markersize=np.sqrt(25 + 340 * np.sqrt(v / ms_max)) * 0.5,
           label=f"{v:g} ms")
    for v in ref_ms
]
lax.legend(handles=size_handles, loc="lower center", ncol=3,
           frameon=False, fontsize=8.5, title="Kernel duration",
           title_fontsize=8.5, handletextpad=1.0, columnspacing=1.6,
           bbox_to_anchor=(0.5, -0.35))

# shared colorbar for occupancy, in its own reserved row below the legend
cbar = fig.colorbar(last_sc, cax=cax, orientation="horizontal")
cbar.set_label("Achieved occupancy %", fontsize=9.5)
cbar.ax.tick_params(labelsize=8.5)

save_fig(fig, "fig_ncu_roofline_4panel")
plt.close(fig)
print("ncu roofline 4-panel figure done.")
