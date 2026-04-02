#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import SIMULATORS

PDES = ["burgers", "kdv", "ks"]

for pde in PDES:
    print(f"  Simulating {pde}...")
    x, t, fields = SIMULATORS[pde]()
    U = fields["u"]
    vr = np.max(np.abs(U))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(t, x, U, shading="auto", cmap="RdBu_r", vmin=-vr, vmax=vr)
    ax.set_xlabel("time", fontsize=24)
    ax.set_ylabel("space", fontsize=24)
    ax.tick_params(labelsize=24)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax).ax.tick_params(labelsize=24)

    plt.tight_layout()
    out = f"plot_{pde}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out}")
