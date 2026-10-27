# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

"""Plot the output of the examples.

    python examples/plot_slice.py air3d_V.npy          # Air3D: three heading slices
    python examples/plot_slice.py ttr.npy              # 2D time-to-reach map

Needs numpy and matplotlib.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1]
V = np.load(path)

if V.ndim == 3:  # Air3D: (x, y, psi) on [-20, 20]^2 x [0, 2 pi)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    n = V.shape[2]
    for ax, k in zip(axes, (0, n // 4, n // 2)):
        ax.contourf(V[:, :, k].T, levels=[V.min(), 0], colors=["tab:red"], extent=(-20, 20, -20, 20), alpha=0.5)
        ax.contour(V[:, :, k].T, levels=[0], colors="k", extent=(-20, 20, -20, 20))
        ax.set_title(f"psi = {2 * np.pi * k / n:.2f} rad")
        ax.set_aspect("equal")
    fig.suptitle("Air3D backward reachable tube (red: V <= 0)")
elif V.ndim == 2:  # time-to-reach on [-1, 1]^2
    T = np.where(np.isfinite(V), V, np.nan)
    plt.imshow(T.T, origin="lower", extent=(-1, 1, -1, 1))
    plt.colorbar(label="time to reach")
    plt.title("Time-to-reach the target (blank: not reached)")
else:
    sys.exit(f"don't know how to plot a {V.ndim}-D array")

out = path.rsplit(".", 1)[0] + ".png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
