#!/usr/bin/env python3
import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth
# ─────────────────────────────────────────────────────────────────────────────

def median_gamma(S):
    sq = cdist(S, S, metric='sqeuclidean')
    gamma = 1.0 / (2.0 * np.median(sq[sq > 0]))
    return gamma, sq


# ─────────────────────────────────────────────────────────────────────────────
# JAX kernel factories
# ─────────────────────────────────────────────────────────────────────────────

def make_kernels(gamma, alpha_rq=2.0, c_bias=1.0):
    def rbf(s, si):
        return jnp.exp(-gamma * jnp.dot(s - si, s - si))

    def exp_dot(s, si):
        return jnp.exp(gamma * jnp.dot(s, si))

    def polynomial(s, si):
        return (jnp.dot(s, si) + c_bias) ** 2

    def matern92(s, si):
        r2 = gamma * jnp.dot(s - si, s - si)
        r = jnp.sqrt(9.0 * r2 + 1e-30)
        return (1.0 + r + 3.0*r**2/7.0 + 2.0*r**3/21.0 + r**4/105.0) * jnp.exp(-r)

    return [
        ("Polynomial", polynomial),
        ("RBF", rbf),
        ("Matérn 9/2", matern92),
        ("Exp. Dot-Product", exp_dot),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Gram matrices (numpy)
# ─────────────────────────────────────────────────────────────────────────────

def gram_matrix(name, S, sq_dists, gamma, alpha_rq=2.0, c_bias=1.0):
    if name == "RBF":
        return np.exp(-gamma * sq_dists)
    elif name == "Polynomial":
        return (S @ S.T + c_bias) ** 2
    elif name == "Exp. Dot-Product":
        return np.exp(gamma * (S @ S.T))
    elif name == "Matérn 9/2":
        r = np.sqrt(9.0 * gamma * sq_dists)
        return (1 + r + 3*r**2/7 + 2*r**3/21 + r**4/105) * np.exp(-r)
    else:
        raise ValueError(f"Unknown kernel '{name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Universal Taylor extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract(kernel_fn, beta, S, D):
    S_j = jnp.array(S)
    beta_j = jnp.array(beta)
    s0_j = jnp.zeros(D)

    def f(s):
        return jnp.sum(beta_j * jax.vmap(lambda si: kernel_fn(s, si))(S_j))

    b = float(f(s0_j))
    W = np.array(jax.grad(f)(s0_j))
    H = 0.5 * np.array(jax.hessian(f)(s0_j))
    return b, W, H


# ─────────────────────────────────────────────────────────────────────────────
# W/H basis precomputation
# ─────────────────────────────────────────────────────────────────────────────

def build_WH_basis(kernel_fn, S, N_C, D):
    s0 = jnp.zeros(D)
    S_j = jnp.array(S)
    grad_k = jax.grad(kernel_fn)
    hess_k = jax.hessian(kernel_fn)
    W_basis = np.array([grad_k(s0, S_j[i]) for i in range(N_C)])
    H_basis = np.array([0.5 * hess_k(s0, S_j[i]) for i in range(N_C)])
    return W_basis, H_basis


# ─────────────────────────────────────────────────────────────────────────────
# PDE configuration registry
# ─────────────────────────────────────────────────────────────────────────────

PDE_CONFIG = {
    "burgers": {
        "base": ["u", "u_x", "u_xx"],
        "gt_label": "W[u_xx]=0.05, H[u·u_x]=-1  (ν=0.05)",
    },
    "kdv": {
        "base": ["u", "u_x", "u_xx", "u_xxx"],
        "gt_label": "W[u_xxx]=-1, H[u·u_x]=-6",
    },
    "ks": {
        "base": ["u", "u_x", "u_xx", "u_xxx", "u_xxxx"],
        "gt_label": "W[u_xx]=-1, W[u_xxxx]=-1, H[u·u_x]=-1",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def spectral_deriv(U, k, p):
    N, nt = U.shape
    return np.array([np.fft.irfft((1j*k)**p * np.fft.rfft(U[:, n]), n=N) for n in range(nt)]).T


def make_interp(x, t, F):
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((x, t), F, method='linear', bounds_error=False, fill_value=None)


# ─────────────────────────────────────────────────────────────────────────────
# Per-PDE simulators
# ─────────────────────────────────────────────────────────────────────────────

def simulate_burgers(nu=0.05):
    N = 512; L = 2*np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.rfftfreq(N, d=L/N) * 2*np.pi
    dt = 0.0025; nt = 800
    t = np.linspace(0, nt*dt, nt)
    u0 = np.sin(x) + 0.5*np.sin(2*x) + 0.3*np.sin(3*x)
    lin = -nu*k**2; exp_l = np.exp(lin*dt)
    safe = np.where(np.abs(lin) < 1e-14, 1.0, lin)
    coef = np.where(np.abs(lin) < 1e-14, dt, (exp_l - 1.0) / safe)
    U = np.zeros((N, nt)); U[:, 0] = u0; u = u0.copy()
    print("  Simulating Burgers...")
    for n in range(nt - 1):
        u_hat = np.fft.rfft(u)
        u_hat = exp_l * u_hat + coef * (-0.5j * k * np.fft.rfft(u**2))
        u = np.fft.irfft(u_hat, n=N); U[:, n + 1] = u
    U_x = spectral_deriv(U, k, 1)
    U_xx = spectral_deriv(U, k, 2)
    U_xxx = spectral_deriv(U, k, 3)
    U_xxxx = spectral_deriv(U, k, 4)
    U_t = -U * U_x + nu * U_xx
    return x, t, {"u": U, "u_x": U_x, "u_xx": U_xx, "u_xxx": U_xxx, "u_xxxx": U_xxxx, "u_t": U_t}


def simulate_kdv():
    N = 512; L = 2*np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.rfftfreq(N, d=L/N) * 2*np.pi
    dt = 1.0 / (1200 - 1); nt = 1200
    t = np.linspace(0, 1.0, nt)
    u0 = 0.5*np.cos(x) + 0.3*np.cos(2*x + 0.5)
    lin = 1j*k**3; exp_l = np.exp(lin*dt)
    safe = np.where(np.abs(lin) < 1e-14, 1.0, lin)
    coef = np.where(np.abs(lin) < 1e-14, dt, (exp_l - 1.0) / safe)
    U = np.zeros((N, nt)); U[:, 0] = u0; u = u0.copy()
    print("  Simulating KdV...")
    for n in range(nt - 1):
        u_hat = np.fft.rfft(u)
        u_hat = exp_l * u_hat + coef * (-6 * 0.5j * k * np.fft.rfft(u**2))
        u = np.fft.irfft(u_hat, n=N); U[:, n + 1] = u
    U_x = spectral_deriv(U, k, 1)
    U_xx = spectral_deriv(U, k, 2)
    U_xxx = spectral_deriv(U, k, 3)
    U_xxxx = spectral_deriv(U, k, 4)
    U_t = -6 * U * U_x - U_xxx
    return x, t, {"u": U, "u_x": U_x, "u_xx": U_xx, "u_xxx": U_xxx, "u_xxxx": U_xxxx, "u_t": U_t}


def simulate_ks():
    N = 1024; L = 22.0
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.rfftfreq(N, d=L/N) * 2*np.pi
    dt = 50.0 / (1000 - 1); nt = 1000
    t = np.linspace(0, 50.0, nt)
    u0 = np.cos(2*np.pi*x/L) + 0.5*np.cos(4*np.pi*x/L + 0.3)
    lin = k**2 - k**4; exp_l = np.exp(lin*dt)
    safe = np.where(np.abs(lin) < 1e-14, 1.0, lin)
    coef = np.where(np.abs(lin) < 1e-14, dt, (exp_l - 1.0) / safe)
    dealias = np.abs(k) < (2/3) * k.max()
    U = np.zeros((N, nt)); U[:, 0] = u0; u = u0.copy()
    print("  Simulating KS...")
    for n in range(nt - 1):
        u_hat = np.fft.rfft(u)
        nl = 1j * k * np.fft.rfft(-0.5 * u**2) * dealias
        u_hat = exp_l * u_hat + coef * nl
        u = np.fft.irfft(u_hat, n=N); U[:, n + 1] = u
    U_x = spectral_deriv(U, k, 1)
    U_xx = spectral_deriv(U, k, 2)
    U_xxx = spectral_deriv(U, k, 3)
    U_xxxx = spectral_deriv(U, k, 4)
    U_t = -U * U_x - U_xx - U_xxxx
    return x, t, {"u": U, "u_x": U_x, "u_xx": U_xx, "u_xxx": U_xxx, "u_xxxx": U_xxxx, "u_t": U_t}


_SIM_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_cache")

def _cached_simulator(name, sim_fn):
    def wrapper(*args, **kwargs):
        os.makedirs(_SIM_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(_SIM_CACHE_DIR, f"{name}.npz")
        if os.path.exists(cache_file):
            print(f"  Loading cached {name} simulation from {cache_file}...")
            data = np.load(cache_file)
            x = data["x"]; t = data["t"]
            fields = {k: data[k] for k in data.files if k not in ("x", "t")}
            return x, t, fields
        x, t, fields = sim_fn(*args, **kwargs)
        np.savez(cache_file, x=x, t=t, **fields)
        print(f"  Cached {name} simulation → {cache_file}")
        return x, t, fields
    return wrapper


SIMULATORS = {
    "burgers": _cached_simulator("burgers", simulate_burgers),
    "kdv": _cached_simulator("kdv", simulate_kdv),
    "ks": _cached_simulator("ks", simulate_ks),
}
