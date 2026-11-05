# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

"""Draw docs/images/problems-{light,dark}.png from the examples' output.

    ./build/examples/air3d_brt && ./build/examples/time_to_reach_2d
    ./build/examples/wildfire/wildfire_voronoi wf
    python docs/images/make_readme_figure.py . wf docs/images

Needs numpy and matplotlib.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
wf = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "wf")
out = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "docs/images")

V = np.load(src / "air3d_V.npy")                 # (x, y, psi) on [-20, 20]^2 x [0, 2 pi)
T = np.load(src / "ttr.npy")                     # [-1, 1]^2, wind (0.4, 0), target r = 0.1 at origin
lab = np.load(wf / "gusty_labels.npy")           # [0, 20 km]^2, drone index per cell
ttr = np.load(wf / "gusty_ttr.npy")
fx, fy = np.load(wf / "gusty_flowx.npy"), np.load(wf / "gusty_flowy.npy")
agents = np.loadtxt(wf / "agents.txt")
AGENT_COLORS = ["#ffaa3c", "#be82ff", "#3cd2ff", "#96eb5a", "#ff6e78", "#fae65a"]

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d1d9e0", unsafe="#e5534b", line="#1f2328"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#3d444d", unsafe="#f47067", line="#e6edf3"),
}


def draw(theme):
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "text.color": c["fg"],
                         "axes.labelcolor": c["muted"], "xtick.color": c["muted"], "ytick.color": c["muted"],
                         "axes.edgecolor": c["grid"]})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), facecolor=c["bg"])
    for ax in axes:
        ax.set_facecolor(c["bg"])
        ax.set_aspect("equal")
        ax.tick_params(length=2.5, labelsize=10.5)

    # 1. Avoid: Air3D unsafe set, relative heading pi (head-on).
    ax = axes[0]
    k = V.shape[2] // 2
    s = V[:, :, k].T
    ext = (-20, 20, -20, 20)
    ax.contourf(s, levels=[s.min(), 0], colors=[c["unsafe"]], alpha=0.35, extent=ext)
    ax.contour(s, levels=[0], colors=[c["unsafe"]], linewidths=1.8, extent=ext)
    ax.add_patch(plt.Circle((0, 0), 5, fill=False, ls="--", lw=1.2, color=c["muted"]))
    ax.annotate("", xy=(4.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c["line"], lw=1.6))
    ax.text(0, -7.2, "you, heading +x\n(collision radius 5 m)", ha="center", va="top", fontsize=10.5, color=c["muted"])
    ax.text(-18.5, 17.2, "unsafe: a collision can\nbe forced within 2 s", fontsize=10.5, color=c["unsafe"], va="top")
    ax.set_xlabel("relative x (m)")
    ax.set_ylabel("relative y (m)")
    ax.set_title("Avoid\nAir3D, intruder head-on: 9 ms", loc="left", fontsize=13.5, color=c["fg"], linespacing=1.45)

    # 2. Reach: minimum time-to-reach with a wind blowing to the right.
    ax = axes[1]
    Tm = np.where(np.isfinite(T), T, np.nan).T
    im = ax.imshow(Tm, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis_r", vmin=0, vmax=1.5)
    ax.contour(np.linspace(-1, 1, T.shape[0]), np.linspace(-1, 1, T.shape[1]), Tm,
               levels=[0.25, 0.5, 0.75, 1.0, 1.25], colors="white", linewidths=0.7, alpha=0.8)
    ax.add_patch(plt.Circle((0, 0), 0.1, color="white"))
    for y in (-0.7, 0.0, 0.7):
        ax.annotate("", xy=(-0.55, y + 0.12), xytext=(-0.95, y + 0.12),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4))
    ax.text(-0.95, 0.88, "wind", color="white", fontsize=10.5)
    ax.text(0.0, -0.18, "target", color="white", fontsize=10.5, ha="center", va="top")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("time to reach target", color=c["muted"])
    cb.outline.set_edgecolor(c["grid"])
    cb.ax.tick_params(labelsize=10, colors=c["muted"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Reach\ntime-to-reach in a wind field: 7 to 14 ms", loc="left", fontsize=13.5, color=c["fg"], linespacing=1.45)

    # 3. Dispatch: which drone reaches each cell first in a gusty wind.
    ax = axes[2]
    L = 20.0
    cm = ListedColormap(AGENT_COLORS)
    shade = np.clip(ttr / np.nanpercentile(ttr[np.isfinite(ttr)], 98), 0, 1)
    rgb = cm(lab.astype(int).T)[..., :3]
    base = np.array([1, 1, 1]) if theme == "light" else np.array([0.05, 0.07, 0.09])
    rgb = rgb * (1 - 0.45 * shade.T[..., None]) + base * 0.45 * shade.T[..., None]
    ax.imshow(rgb, origin="lower", extent=(0, L, 0, L), interpolation="bilinear")
    edge = (np.abs(np.diff(lab, axis=0, prepend=lab[:1])) + np.abs(np.diff(lab, axis=1, prepend=lab[:, :1]))) > 0
    ax.contour(np.linspace(0, L, lab.shape[0]), np.linspace(0, L, lab.shape[1]), edge.T.astype(float),
               levels=[0.5], colors=[c["line"]], linewidths=0.8)
    g = np.linspace(0, L, fx.shape[0])
    ax.streamplot(g, g, fx.T, fy.T, color=(1, 1, 1, 0.55), density=0.75, linewidth=0.7, arrowsize=0.8)
    for x, y in agents / 1000.0:
        ax.plot(x, y, "o", ms=8, mfc="white", mec="#1f2328", mew=1.8)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Dispatch\nfirst to arrive, 6 drones: 17 ms", loc="left", fontsize=13.5, color=c["fg"], linespacing=1.45)

    fig.tight_layout(w_pad=2.2)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"problems-{theme}.png", dpi=110, facecolor=c["bg"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


for th in THEMES:
    draw(th)
print("wrote", out)
