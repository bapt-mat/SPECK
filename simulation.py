#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import SIMULATORS

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="discovered_pde_*.txt")
parser.add_argument("--dt_factor", type=float, default=1.0, help="Multiply original dt by this factor (e.g. 0.1 for 10x smaller step)")
parser.add_argument("--vmax_diff", type=float, default=None, help="Fix the error colorbar range to [-vmax_diff, vmax_diff]")

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Parse discovered PDE file — use first kernel found
# ─────────────────────────────────────────────────────────────────────────────
pde_name = None
BASE = None
kernel_name = None
b = W = H = None
H_rows = []
reading_H = False

with open(args.file) as f:
    for line in f:
        s = line.strip()
        if s.startswith("pde:"):
            pde_name = s.split(":", 1)[1].strip()
        elif s.startswith("base:"):
            BASE = s.split(":", 1)[1].strip().split()
        elif s.startswith("[") and s.endswith("]") and kernel_name is None:
            kernel_name = s[1:-1]
            H_rows = []; reading_H = False
        elif kernel_name is None:
            continue
        elif s.startswith("b:"):
            b = float(s.split(":", 1)[1])
            reading_H = False
        elif s.startswith("W:"):
            W = np.array([float(v) for v in s.split(":", 1)[1].split()])
            reading_H = False
        elif s == "H:":
            reading_H = True
        elif reading_H and s:
            H_rows.append([float(v) for v in s.split()])
            if len(H_rows) == len(BASE):
                break

H = np.array(H_rows)
D = len(BASE)

W = np.where(np.abs(W) >= 5e-7, W, 0.0)
H = np.where(np.abs(H) >= 5e-7, H, 0.0)

print(f"  PDE    : {pde_name}")
print(f"  Kernel : {kernel_name}")

# ─────────────────────────────────────────────────────────────────────────────
# Original simulation (same domain and IC)
# ─────────────────────────────────────────────────────────────────────────────
print(f"  Loading original {pde_name} simulation...")
x, t, fields = SIMULATORS[pde_name]()
U_orig = fields["u"]
N = len(x)
t_end = float(t[-1])
dt_orig = float(t[1] - t[0])
dt = dt_orig * args.dt_factor
nt_inner = max(1, round(dt_orig / dt))
dt = dt_orig / nt_inner
nt = len(t)
L_dom = float(x[-1] - x[0] + (x[1] - x[0]))
k = np.fft.rfftfreq(N, d=L_dom / N) * 2 * np.pi

if nt_inner > 1:
    print(f"  dt_factor={args.dt_factor} → {nt_inner} sub-steps per output step (dt={dt:.6g})")

deriv_order = [name.count('x') for name in BASE]

# ─────────────────────────────────────────────────────────────────────────────
# ETD1 setup
# ─────────────────────────────────────────────────────────────────────────────
L_hat = sum(W[j] * (1j * k) ** deriv_order[j] for j in range(D)).astype(complex)
exp_L = np.exp(L_hat * dt)
safe_L = np.where(np.abs(L_hat) < 1e-14, 1.0, L_hat)
coef_L = np.where(np.abs(L_hat) < 1e-14, dt, (exp_L - 1.0) / safe_L)

def basis(u):
    u_hat = np.fft.rfft(u)
    return [u if o == 0 else np.fft.irfft((1j * k) ** o * u_hat, n=N) for o in deriv_order]

def nonlinear(u):
    fs = basis(u)
    nl = np.full(N, b)
    for i in range(D):
        for j in range(D):
            nl += H[i, j] * fs[i] * fs[j]
    return nl

# ─────────────────────────────────────────────────────────────────────────────
# Forward simulation
# ─────────────────────────────────────────────────────────────────────────────
u = U_orig[:, 0].copy()
U_disc = np.zeros((N, nt))
U_disc[:, 0] = u

print("  Running ETD1 forward simulation with discovered PDE...")
for n in range(nt - 1):
    for _ in range(nt_inner):
        u_hat = np.fft.rfft(u)
        nl_hat = np.fft.rfft(nonlinear(u))
        u_hat = exp_L * u_hat + coef_L * nl_hat
        u = np.fft.irfft(u_hat, n=N)
    U_disc[:, n + 1] = u

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
diff = U_disc - U_orig
mae = float(np.mean(np.abs(diff)))
vr = float(np.max(np.abs(U_orig)))
vd = float(args.vmax_diff) if args.vmax_diff is not None else float(np.max(np.abs(diff)))
print(f"  vmax_diff: {vd:.4e}")

fig, axes = plt.subplots(2, 1, figsize=(6, 9), sharex=True)

for ax, U_plot, vv, show_xlabel, cmap in [
    (axes[0], U_disc, vr, False, "RdBu_r"),
    (axes[1], diff, vd, True, "Greys_r"),
]:
    im = ax.pcolormesh(t, x, U_plot, shading="auto", cmap=cmap, vmin=-vv, vmax=vv)
    if show_xlabel:
        ax.set_xlabel("time", fontsize=24)
    ax.set_ylabel("space", fontsize=24)
    ax.tick_params(labelsize=24)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax).ax.tick_params(labelsize=24)

axes[1].text(0.98, 0.97, f"MAE = {mae:.2e}", transform=axes[1].transAxes, fontsize=14, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(left=0.13)
stem = os.path.splitext(os.path.basename(args.file))[0]
out = f"simulation_{stem}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  Plot saved → {out}")
