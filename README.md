# SPECK - Sparse Partial differential Equation disCovery with Kernels

## Overview

SPECK discovers the governing PDE of a dynamical system from simulation data. It represents the right-hand side of `u_t = F(u, u_x, u_xx, ...)` as a kernel expansion and recovers sparse coefficients (constant, linear, quadratic) via convex optimisation.

Supported PDEs: **Burgers**, **KdV**, **Kuramoto-Sivashinsky (KS)**.

## Files

| File | Description |
|------|-------------|
| `SPECK.py` | Main script: simulate, extract PDE coefficients, save results |
| `simulation.py` | Forward-simulate the discovered PDE and compare to ground truth |
| `original_pdes.py` | Plot the ground-truth PDE solutions |
| `utils.py` | Kernels, simulators, feature extraction utilities |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Step 1 — Extract PDE coefficients:**
```bash
python SPECK.py --pde burgers
python SPECK.py --pde kdv
python SPECK.py --pde ks
```

This saves `discovered_pde_{pde}.txt`.

**Step 2 — Simulate the discovered PDE:**
```bash
python simulation.py --file discovered_pde_burgers.txt
```

**Plot ground-truth fields:**
```bash
python original_pdes.py
```

## Key options for `SPECK.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--pde` | required | `burgers`, `kdv`, or `ks` |
| `--n_colloc` | 100 | Number of collocation points |
| `--lam` | 0.0 | L1 sparsity weight |
| `--ridge_gamma` | 1e-8 | Ridge regularisation |
| `--colloc_type` | `grid` | `grid` or `random` |
| `--kernel` | all | `poly`, `rbf`, `matern`, `expdot` |
