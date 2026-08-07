"""Figure 2: Deployment overview across neural-operator families on the
Jetson Orin Nano -- model-scale sensitivity and resolution scaling, all
under the admitted FP32 runtime path.

Four conceptual axes are folded into four panels (2x2):
  family (4, color) x problem (2, panel split) x {model scale (3) |
  resolution (5)} (x-axis) x statistic (2: median line, P95 as a shaded
  band).

Panel (a),(b): model-scale sensitivity (small/base/large) at the reference
           resolution, Burgers r2048 / Darcy 141x141 -- plotted against each
           point's actual parameter count (log x), not an evenly spaced
           Small/Base/Large category.
Panel (c),(d): resolution scaling of the base checkpoint, Burgers r512-r8192
           / Darcy 85x85-421x421.

An earlier draft carried a fifth panel -- a cross-family bar chart at the
base/reference point -- ahead of these two. It was cut: its 8 bars are
exactly the "Base" tick already sitting inside (a)/(b), so it added a
second encoding of the same 8 numbers without a new axis of information,
at the cost of panel budget and an asymmetry (model scale got a dedicated
summary panel, resolution didn't). Four panels with no repeated content
inside the figure was judged the stronger design; the "Base" values that
panel used to headline are still the first labeled point in (a)/(b).

Why parameter count, not a Small/Base/Large category, is the x-axis in
(a)/(b): the paper's own central claim in this section is that parameter
count does not predict deployment cost (stated in 5.1, 5.2, and again in
the conclusion). An evenly spaced category axis hides *how much* the
parameter count actually grows between points -- e.g. Burgers large is
~11x the params of Burgers small for FNO/WNO/Sp2GNO, but Darcy large is
~42x base for FNO alone. Plotting against the real, log-scaled parameter
count makes the mismatch the reader is supposed to take away -- large
horizontal movement, flat or barely-rising lines for three of the four
families -- visible in the geometry itself rather than requiring the
reader to hold specific numbers from the text in their head. Marker size
is uniform and matches (c)/(d) -- the x-axis position already carries the
small/base/large information, so a redundant size encoding was dropped.

Design note on P95: median and P95 track within ~1-4% of each other for
WNO and Sp2GNO everywhere (a tight, deterministic single-sample harness),
but diverge by 8-50% for FNO and DeepONet at several resolutions (cuFFT
plan/allocator jitter). That divergence is real signal, not noise, so P95
is carried as a shaded band (median -> P95) in every panel rather than
dropped -- a family with a fat band has a less predictable per-call
latency than the median alone would suggest, which matters for a
real-time deployment budget.

Value labels: placed by a renderer-driven collision solver (LabelPlacer)
that measures every candidate's actual pixel bounding box against every
line segment and already-placed label in that axes (not just "is my
nearest same-x neighbor close" -- a label can just as easily collide with
a *different* family's line passing through on its way elsewhere). Each
point tries a fixed priority list of offset positions (right, above,
below, further out); the first collision-free one wins, and a point with
no collision-free option is left unlabeled rather than forced. Exact
values for every point are in the appendix tables regardless.

All values are Jetson Orin Nano, batch size one, FP32. TorchScript backend
for FNO/DeepONet (their admitted runtime path); eager for WNO/Sp2GNO (no
TorchScript path exists for their runtime in this harness). See
build_fig2_full_data.py for the exact source CSV/JSON per family and the
regenerated fig2_full_data.json for the numbers plotted here -- every
point is traceable to one appendix-table row.

Usage:
    python build_fig2_full_data.py   # regenerate fig2_full_data.json
    python plot_figure2_deployment_overview.py
Produces:
    figures/fig2_deployment_overview.png  (150 dpi)
    figures/fig2_deployment_overview.pdf  (vector)
    figures/fig2_deployment_overview_400dpi.png  (print quality)
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from _style import setup_style, FAMILY_COLORS, save_fig

FAMILIES = ["FNO", "DeepONet", "WNO", "Sp2GNO"]
FAMILY_DISPLAY = {"FNO": "FNO", "DeepONet": "DeepONet", "WNO": "WNO", "Sp2GNO": r"Sp$^2$GNO"}

DATA = json.load(open("/home/jetson/jjyoo3/edge_figures/fig2_full_data.json"))

BURGERS_RES = [512, 1024, 2048, 4096, 8192]
DARCY_RES = [85, 141, 211, 281, 421]


# =============================================================================
# Renderer-driven label placement: try candidate offsets, keep the first one
# whose actual pixel bbox clears every line and already-placed label.
# =============================================================================
class LabelPlacer:
    def __init__(self, fig, ax, pad_px=2.5):
        self.fig, self.ax, self.pad = fig, ax, pad_px
        self.renderer = fig.canvas.get_renderer()
        self.placed = []  # list of Bbox, in display (pixel) coords
        self.obstacles = self._collect_obstacles()

    def _collect_obstacles(self):
        obs = []
        for line in self.ax.get_lines():
            disp = self.ax.transData.transform(line.get_xydata())
            for i in range(len(disp) - 1):
                obs.append(("seg", disp[i], disp[i + 1]))
        return obs

    @staticmethod
    def _seg_hits_bbox(bbox, p0, p1):
        x0, x1, y0, y1 = bbox.x0, bbox.x1, bbox.y0, bbox.y1
        (px0, py0), (px1, py1) = p0, p1
        dx, dy = px1 - px0, py1 - py0
        tmin, tmax = 0.0, 1.0
        for p, q in [(-dx, px0 - x0), (dx, x1 - px0), (-dy, py0 - y0), (dy, y1 - py0)]:
            if p == 0:
                if q < 0:
                    return False
            else:
                t = q / p
                if p < 0:
                    tmin = max(tmin, t)
                else:
                    tmax = min(tmax, t)
        return tmin <= tmax

    def _collides(self, bbox):
        expanded = Bbox.from_extents(bbox.x0 - self.pad, bbox.y0 - self.pad,
                                      bbox.x1 + self.pad, bbox.y1 + self.pad)
        for b in self.placed:
            if expanded.overlaps(b):
                return True
        for kind, a, b in self.obstacles:
            if kind == "seg" and self._seg_hits_bbox(expanded, a, b):
                return True
        return False

    def place(self, x, y, text, candidates, fontsize=7.2, color="#333333"):
        """candidates: list of (dx_pts, dy_pts, ha, va) tried in priority order."""
        for dx, dy, ha, va in candidates:
            t = self.ax.annotate(text, xy=(x, y), xycoords="data",
                                   xytext=(dx, dy), textcoords="offset points",
                                   ha=ha, va=va, fontsize=fontsize, color=color, zorder=5)
            bbox = t.get_window_extent(renderer=self.renderer)
            if not self._collides(bbox):
                self.placed.append(bbox)
                return True
            t.remove()
        return False


RIGHT_FIRST = [(6, 0, "left", "center"), (6, 8, "left", "bottom"), (6, -8, "left", "top"),
               (0, 9, "center", "bottom"), (0, -10, "center", "top"),
               (11, 0, "left", "center")]
ABOVE_FIRST = [(0, 7, "center", "bottom"), (0, -9, "center", "top"),
               (8, 6, "left", "bottom"), (-8, 6, "right", "bottom"),
               (8, -7, "left", "top"), (-8, -7, "right", "top")]


def _param_scale_panel(fig, ax, panel_key, xlabel, title, ylim=None):
    """Model-scale panel with x = actual parameter count (log), not an evenly
    spaced Small/Base/Large category. See module docstring for rationale."""
    fam_data = {}
    for fam in FAMILIES:
        entries = DATA[fam][panel_key]  # [(median, p95, params), ...] x3
        params = np.array([e[2] for e in entries], dtype=float)
        med = np.array([e[0] for e in entries])
        p95 = np.array([e[1] for e in entries])
        fam_data[fam] = (params, med, p95)

    for fam in FAMILIES:
        params, med, p95 = fam_data[fam]
        ax.plot(params, med, "-o", color=FAMILY_COLORS[fam], linewidth=1.8,
                 markersize=5.2, markeredgecolor="white", markeredgewidth=0.8,
                 label=FAMILY_DISPLAY[fam], zorder=3)
        ax.fill_between(params, med, p95, color=FAMILY_COLORS[fam], alpha=0.20, linewidth=0, zorder=1)

    xmin = min(fam_data[f][0].min() for f in FAMILIES)
    xmax = max(fam_data[f][0].max() for f in FAMILIES)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin / 1.7, xmax * 2.6)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)

    fig.canvas.draw()
    placer = LabelPlacer(fig, ax)
    priority = ["WNO", "Sp2GNO", "FNO", "DeepONet"]
    for i in range(3):
        cands = RIGHT_FIRST if i == 2 else ABOVE_FIRST
        for fam in priority:
            params, med, p95 = fam_data[fam]
            placer.place(params[i], med[i], f"{med[i]:.1f}", cands, fontsize=7.2)


def _scaling_panel(fig, ax, x_vals, panel_key, xlabel, log_x, title, ylim=None):
    xs = np.array(x_vals) if log_x else np.arange(len(x_vals))
    series = {fam: np.array([p[0] for p in DATA[fam][panel_key]]) for fam in FAMILIES}
    p95s = {fam: np.array([p[1] for p in DATA[fam][panel_key]]) for fam in FAMILIES}

    for fam in FAMILIES:
        ax.plot(xs, series[fam], "-o", color=FAMILY_COLORS[fam], linewidth=1.8,
                 markersize=5.2, markeredgecolor="white", markeredgewidth=0.8,
                 label=FAMILY_DISPLAY[fam], zorder=3)
        ax.fill_between(xs, series[fam], p95s[fam], color=FAMILY_COLORS[fam],
                         alpha=0.20, linewidth=0, zorder=1)

    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlim(x_vals[0] / 1.35, x_vals[-1] * 1.9)
    else:
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels(x_vals)
        ax.set_xlim(-0.4, len(x_vals) - 1 + 0.55)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)

    fig.canvas.draw()
    placer = LabelPlacer(fig, ax)
    n_pts = len(x_vals)
    # Label priority: families with the widest typical separation (WNO, Sp2GNO)
    # get first pick at each x, since FNO/DeepONet are the ones that cross.
    priority = ["WNO", "Sp2GNO", "FNO", "DeepONet"]
    for i in range(n_pts):
        is_last = i == n_pts - 1
        for fam in priority:
            v = series[fam][i]
            cands = RIGHT_FIRST if is_last else ABOVE_FIRST
            placer.place(xs[i], v, f"{v:.1f}", cands, fontsize=7.2)


def main():
    setup_style()
    plt.rcParams["grid.alpha"] = 0.4  # match the softer grid used in the sibling nsys/ncu figures

    fig = plt.figure(figsize=(10.5, 7.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 1.0],
        hspace=0.58, wspace=0.28,
        left=0.075, right=0.98, top=0.92, bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _param_scale_panel(fig, ax_a, "b", "Parameter count", title="(a) Burgers r2048", ylim=(0.9, 90))
    ax_a.set_ylabel("Median latency (ms)")

    _param_scale_panel(fig, ax_b, "c", "Parameter count", title="(b) Darcy 141$\\times$141", ylim=(1.4, 160))

    _scaling_panel(fig, ax_c, BURGERS_RES, "d", "Grid resolution", log_x=True, title="(c) Burgers, base", ylim=(1, 100))
    ax_c.set_ylabel("Median latency (ms)")

    _scaling_panel(fig, ax_d, DARCY_RES, "e", "Grid resolution (per side)", log_x=True, title="(d) Darcy, base", ylim=(2, 900))

    # Single shared legend, centered in the gap between row 1 and row 2.
    handles = [Line2D([0], [0], marker="o", linestyle="-", color=FAMILY_COLORS[f],
                       markeredgecolor="white", markeredgewidth=0.8, markersize=7,
                       linewidth=2.2, label=FAMILY_DISPLAY[f]) for f in FAMILIES]
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=10.5,
               handlelength=1.9, columnspacing=1.9, bbox_to_anchor=(0.53, 0.495))

    save_fig(fig, "fig2_deployment_overview")
    plt.close(fig)


if __name__ == "__main__":
    main()
