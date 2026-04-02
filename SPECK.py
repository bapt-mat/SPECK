#!/usr/bin/env python3
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

from utils import (PDE_CONFIG, SIMULATORS, make_interp, median_gamma, make_kernels, gram_matrix, extract, build_WH_basis)

jax.config.update("jax_enable_x64", True)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pde", choices=["burgers", "kdv", "ks"], required=True)
parser.add_argument("--n_colloc", type=int, default=100)
parser.add_argument("--ridge_gamma", type=float, default=1e-8)
parser.add_argument("--lam", type=float, default=0.0)
parser.add_argument("--colloc_type", choices=["random", "grid"], default="grid")
parser.add_argument("--c_bias", type=float, default=1.0)
parser.add_argument("--kernel", choices=["poly", "rbf", "matern", "expdot"], default=None, help="Kernel to use (default: all four)")
args = parser.parse_args()

KERNEL_ALIASES = {
    "poly": "Polynomial",
    "rbf": "RBF",
    "matern": "Matérn 9/2",
    "expdot": "Exp. Dot-Product",
}

cfg = PDE_CONFIG[args.pde]
BASE = cfg["base"]
D = len(BASE)
N_COLLOC = args.n_colloc
RIDGE_GAMMA = args.ridge_gamma
LAM = args.lam
C_BIAS = args.c_bias

KEEP_KERNELS = ({KERNEL_ALIASES[args.kernel]} if args.kernel else set(KERNEL_ALIASES.values()))

# ─────────────────────────────────────────────────────────────────────────────
# Simulate
# ─────────────────────────────────────────────────────────────────────────────
x, t, fields = SIMULATORS[args.pde]()

# ─────────────────────────────────────────────────────────────────────────────
# Collocation points
# ─────────────────────────────────────────────────────────────────────────────
if args.colloc_type == "random":
    pts = np.column_stack([np.random.uniform(x.min(), x.max(), N_COLLOC), np.random.uniform(t.min(), t.max(), N_COLLOC)])
else:
    n_side = int(np.round(np.sqrt(N_COLLOC)))
    xg, tg = np.meshgrid(np.linspace(x.min(), x.max(), n_side), np.linspace(t.min(), t.max(), n_side), indexing='ij')
    pts = np.column_stack([xg.ravel(), tg.ravel()])

def interp(key):
    return make_interp(x, t, fields[key])(pts)

S = np.column_stack([interp(f) for f in BASE])
u_t_c = interp("u_t")
N_C = len(pts)

GAMMA, sq_dists = median_gamma(S)

KERNELS = [(n, k) for n, k in make_kernels(GAMMA, C_BIAS) if n in KEEP_KERNELS]

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  Sparse extraction — {args.pde.upper()}")
print(f"  GT: {cfg['gt_label']}")
print(f"  λ = {LAM}")
print("=" * 65)

kernel_names = []
results = {}

for name, kfn in KERNELS:
    print(f"\n{'─'*65}")
    print(f"  Kernel: {name}")
    print(f"{'─'*65}")

    K_k = gram_matrix(name, S, sq_dists, GAMMA, C_BIAS)

    print("  Building W/H bases via JAX...")
    W_basis, H_basis = build_WH_basis(kfn, S, N_C, D)
    W_coeff = W_basis.T
    H_coeff = H_basis.reshape(N_C, D * D).T

    s0 = jnp.zeros(D)
    b_coeff = np.array([kfn(s0, jnp.array(S[i])) for i in range(N_C)])

    β = cp.Variable(N_C)
    b_expr = b_coeff @ β
    W_expr = W_coeff @ β
    H_expr = cp.reshape(H_coeff @ β, (D, D), order='C')

    objective = cp.sum_squares(K_k @ β - u_t_c) + RIDGE_GAMMA * cp.quad_form(β, cp.psd_wrap(K_k + 1e-12 * np.eye(N_C))) + LAM * cp.norm1(H_expr) + LAM * cp.norm1(W_expr)

    prob = cp.Problem(cp.Minimize(objective))
    prob.solve(solver=cp.CLARABEL, verbose=False)
    print(f"  status: {prob.status}  |  obj: {prob.value:.4e}")

    if β.value is None:
        print("  Solver failed, skipping.")
        continue

    W_out = W_coeff @ β.value
    H_out = (H_coeff @ β.value).reshape(D, D, order='C')
    b_out = extract(kfn, β.value, S, D)[0]

    print(f"\n  b  = {b_out:+.6f}")
    print(f"  W (linear terms):")
    for j in range(D):
        print(f"    {BASE[j]:8s}: {W_out[j]:+.6f}")
    print("  H (quadratic terms, symmetrised):")
    for i in range(D):
        for j in range(i, D):
            sym = H_out[i, j] + (H_out[j, i] if i != j else 0.0)
            print(f"    {BASE[i]}·{BASE[j]:8s}: {sym:+.6f}")

    kernel_names.append(name)
    results[name] = {"b": b_out, "W": W_out, "H": H_out}

print(f"\n{'='*65}")

# ─────────────────────────────────────────────────────────────────────────────
# Save discovered PDE coefficients
# ─────────────────────────────────────────────────────────────────────────────
kernel_tag = f"_{args.kernel}" if args.kernel else ""
out_txt = f"discovered_pde_{args.pde}{kernel_tag}.txt"
with open(out_txt, "w") as fout:
    fout.write(f"pde: {args.pde}\n")
    fout.write(f"base: {' '.join(BASE)}\n")
    for kname in kernel_names:
        res = results[kname]
        fout.write(f"\n[{kname}]\n")
        fout.write(f"b: {res['b']:+.10e}\n")
        fout.write(f"W: {' '.join(f'{v:+.10e}' for v in res['W'])}\n")
        fout.write("H:\n")
        for row in res['H']:
            fout.write(f"  {' '.join(f'{v:+.10e}' for v in row)}\n")
print(f"  Discovered PDE saved → {out_txt}")
