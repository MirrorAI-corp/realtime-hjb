# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

"""Draw fleet_sim frames: the capture-basin partition, the wind, the drones
and the fires.

    python examples/wildfire/plot_frame.py wildfire_fleet/frame_00450.bin
    python examples/wildfire/plot_frame.py wildfire_fleet/frame_00*.bin

Writes a .png next to each frame. Needs numpy and matplotlib.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

L_KM = 20.0  # fleet_sim's domain is [0, 20 km] x [0, 20 km]
DRONE_STATUS = ["patrol", "en route", "suppressing", "returning"]
HEADER = np.dtype([("magic", "<u4"), ("version", "<u4"), ("nx", "<i4"), ("ny", "<i4"),
                   ("drones", "<i4"), ("fires", "<i4"), ("t", "<f8")])
DRONE = np.dtype([("x", "<f8"), ("y", "<f8"), ("home_x", "<f8"), ("home_y", "<f8"),
                  ("arrival_t", "<f8"), ("status", "u1"), ("fire", "i1")])
FIRE = np.dtype([("x", "<f8"), ("y", "<f8"), ("t_appear", "<f8"), ("t_dispatched", "<f8"),
                 ("t_suppressed", "<f8"), ("state", "u1"), ("drone", "i1")])


def read_frame(path):
    buf = open(path, "rb").read()
    h = np.frombuffer(buf, HEADER, count=1)[0]
    if h["magic"] != 0x57464C54 or h["version"] != 2:
        sys.exit(f"{path}: not a version-2 fleet_sim frame")
    nx, ny, off = int(h["nx"]), int(h["ny"]), HEADER.itemsize
    labels = np.frombuffer(buf, np.uint8, nx * ny, off).reshape(nx, ny)
    off += nx * ny
    fx = np.frombuffer(buf, "<f4", nx * ny, off).reshape(nx, ny)
    off += 4 * nx * ny
    fy = np.frombuffer(buf, "<f4", nx * ny, off).reshape(nx, ny)
    off += 4 * nx * ny
    drones = np.frombuffer(buf, DRONE, int(h["drones"]), off)
    off += DRONE.itemsize * int(h["drones"])
    fires = np.frombuffer(buf, FIRE, int(h["fires"]), off)
    return float(h["t"]), labels, fx, fy, drones, fires


def plot(path):
    t, labels, fx, fy, drones, fires = read_frame(path)
    nx, ny = labels.shape
    fig, ax = plt.subplots(figsize=(6.4, 6))
    part = np.ma.masked_equal(labels, 0xFF).T  # arrays are indexed [x][y]
    ax.imshow(part, origin="lower", extent=(0, L_KM, 0, L_KM), cmap="tab10",
              vmin=0, vmax=9, alpha=0.45, interpolation="nearest")
    s = max(1, nx // 20)
    xs = (np.arange(nx) + 0.5) * L_KM / nx
    ys = (np.arange(ny) + 0.5) * L_KM / ny
    ax.quiver(xs[::s], ys[::s], fx[::s, ::s].T, fy[::s, ::s].T, color="0.35", width=0.002)
    for i, d in enumerate(drones):
        c = plt.cm.tab10(i % 10)
        ax.plot(d["home_x"] / 1e3, d["home_y"] / 1e3, "s", mfc="none", mec=c, ms=7)
        ax.plot(d["x"] / 1e3, d["y"] / 1e3, "o", color=c, mec="k", ms=8,
                label=f"drone {i}: {DRONE_STATUS[d['status']]}")
    for f in fires:
        if f["state"] == 0:  # not started yet
            continue
        ax.plot(f["x"] / 1e3, f["y"] / 1e3, "*", ms=16, mec="k",
                color="red" if f["state"] == 1 else "0.6")
    ax.set_xlim(0, L_KM)
    ax.set_ylim(0, L_KM)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"t = {t:.0f} s   (colour: first drone to arrive; star: fire)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        plot(p)
