#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Figure 2: Global stability analysis of Feedforward Hebbian learning.
Solves the generalized eigenvalue problem across beta, N_in, and alpha_J with 
spatial resolution ds = 0.1 cm and k = 1e-20, saving results and parameter 
dictionary to ../figure2-stability-ff-data/.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from tqdm import tqdm

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.theory import (
    learned_U_std,
    learned_weight_std,
    theoretical_distance,
    theoretical_speed,
)


class Params:
    def __init__(
        self,
        ds=0.1,
        N=512,
        L=100.0,
        m=0.15,
        tau=0.015,
        tauv=0.6,
        k=0.01,
        beta=0.5,
        eta=5e-4,
        alpha_J=1.0,
        A_R=30.0,
        sigma_R=5.0,
    ):
        self.ds = ds
        self.N = N
        self.L = L
        self.m = m
        self.tau = tau
        self.tauv = tauv
        self.k = k
        self.beta = beta
        self.eta = eta
        self.alpha_J = alpha_J
        self.A_R = A_R
        self.sigma_R = sigma_R

    def derive_relevant_parameters(self):
        self.rho = self.N / self.L
        self.sigma_U = learned_U_std(self.beta, self.sigma_R)
        self.sigma_J = learned_weight_std(self.beta, self.sigma_R)

        self.Cbeta = np.sqrt((2 * np.pi * self.sigma_J**2) ** (1 - self.beta) / self.beta)

        self.v = theoretical_speed(self.m, self.tau, self.tauv, self.sigma_R)
        self.d = theoretical_distance(self.m, self.tau, self.tauv, self.sigma_R, self.v)

        self.A_J = (self.A_R / (self.alpha_J * self.Cbeta)) ** (1 / self.beta)
        self.A_u = self.rho * self.A_R * self.A_J / (1 + self.tau * self.v / self.d)
        self.B = 1 + self.k * self.rho * self.A_u**2 / np.sqrt(4 * np.pi * self.sigma_U**2)
        self.A_r = self.A_u**2 / np.sqrt(4 * np.pi * self.sigma_U**2) / self.B

        self.s_max = self.L
        self.Ngrid = int(np.round(2.0 * self.s_max / self.ds)) + 1


def gaussian_1d(x, mu, sigma):
    """N(x; mu, sigma): 1D non-periodic Gaussian evaluated at x."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def phi(s, s_pp, p):
    sR2 = p.sigma_R**2
    sU2 = p.sigma_U**2
    denom = 2.0 * sR2 * (2.0 * sU2 + sR2)
    return ((s**2 + s_pp**2) * (sU2 + sR2) + 2.0 * s * s_pp * sU2) / denom


def T1_kernel(s, s_pp, p):
    prefactor = (
        p.A_R**2
        * p.A_u
        / (2.0 * np.pi * p.sigma_R * np.sqrt(2.0 * p.sigma_U**2 + p.sigma_R**2))
    )
    return prefactor * np.exp(-phi(s, s_pp, p))


def T2_kernel(s, s_pp, p):
    sig1 = p.sigma_J / np.sqrt(p.beta)
    sig2 = np.sqrt(p.sigma_R**2 + p.sigma_U**2)
    return p.A_R**2 * p.A_u * gaussian_1d(s, 0.0, sig1) * gaussian_1d(s_pp, 0.0, sig2)


def J_kernel(s, p):
    return p.A_J * gaussian_1d(s, 0.0, p.sigma_J)


def tau_J(s, p):
    return J_kernel(s, p) ** (1.0 - p.beta) / (p.beta * p.eta * p.alpha_J * p.A_r)


def K_tilde(s, s_pp, p):
    prefactor = (
        2.0
        * p.rho
        / (p.B * (p.m + 1.0))
        * J_kernel(s, p) ** (1.0 - p.beta)
        / (p.beta * p.alpha_J * p.A_r)
    )
    return prefactor * (T1_kernel(s, s_pp, p) - T2_kernel(s, s_pp, p))


def build_matrices(p):
    s_grid = np.linspace(-p.s_max, p.s_max, p.Ngrid)
    ds = s_grid[1] - s_grid[0]

    S, S_pp = np.meshgrid(s_grid, s_grid, indexing="ij")
    K_mat = K_tilde(S, S_pp, p) * ds
    tau_vec = tau_J(s_grid, p)
    L_mat = K_mat - np.eye(p.Ngrid)
    M_mat = (1.0 / tau_vec[:, None]) * L_mat

    return s_grid, ds, K_mat, tau_vec, L_mat, M_mat


def solve_stability2(p):
    s_grid, ds, K_mat, tau_vec, L_mat, M_mat = build_matrices(p)

    J_vec = J_kernel(s_grid, p)
    J_max = J_vec.max()
    J_normalized = J_vec / J_max
    physical_mask = J_normalized > 1e-5

    s_phys = s_grid[physical_mask]
    L_phys = L_mat[np.ix_(physical_mask, physical_mask)]
    tau_phys = np.diag(tau_vec[physical_mask])

    eigenvalues, eigenvectors = eig(L_phys, tau_phys)
    max_Re = np.max(eigenvalues.real)
    is_stable = max_Re < 0

    return eigenvalues, eigenvectors, is_stable, s_phys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Figure 2 stability eigenvalue sweep over beta, N_in, and alpha_J."
    )
    parser.add_argument("--ds", type=float, default=0.1, help="Spatial resolution Delta s (cm)")
    parser.add_argument("--m", type=float, default=0.15, help="Adaptation strength")
    parser.add_argument("--tau", type=float, default=0.015, help="U time constant (s)")
    parser.add_argument("--tauv", type=float, default=0.6, help="V time constant (s)")
    parser.add_argument("--k", type=float, default=0.01, help="Global inhibition (cm^4 s^2 / syn)")
    parser.add_argument("--etaJ", type=float, default=5e-4, help="FF learning rate (Hz)")
    parser.add_argument("--A_R", type=float, default=30.0, help="Input amplitude (Hz)")
    parser.add_argument("--sigma_R", type=float, default=5.0, help="Input width (cm)")
    parser.add_argument("--N_min", type=int, default=512, help="Minimum N_in")
    parser.add_argument("--N_max", type=int, default=5000, help="Maximum N_in")
    parser.add_argument("--num_N", type=int, default=10, help="Number of N_in points")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        help="List of beta values",
    )
    parser.add_argument(
        "--alphaJ_list",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
        help="List of alpha_J values (0.001 to 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    repo_root = Path(__file__).resolve().parent.parent
    save_dir = repo_root.parent / "figure2-stability-ff-data"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving Figure 2 stability sweep data to: {save_dir}")

    # 2. Sweep Grids Matching Figure 2 Caption
    beta_list = np.array(args.betas)
    N_list = np.linspace(args.N_min, args.N_max, args.num_N, dtype=int)
    alphaJ_list = np.array(args.alphaJ_list)

    # Shape: (len(beta_list), len(N_list), len(alphaJ_list))
    eigenvalues_mat = np.zeros((len(beta_list), len(N_list), len(alphaJ_list)))
    L_list = np.zeros(len(beta_list))

    # 3. Compute Generalized Eigenvalue Sweep
    total_iters = len(beta_list) * len(N_list) * len(alphaJ_list)
    with tqdm(total=total_iters, desc="Solving Eigenvalues") as pbar:
        for l, beta in enumerate(beta_list):
            # L = 20 * sqrt((2*beta + 2) / (2 - beta)) * sigma_R = 20 * learned_U_std(beta, sigma_R)
            L = 20.0 * learned_U_std(beta, args.sigma_R)
            L_list[l] = L
            for i, n in enumerate(N_list):
                for j, alpha in enumerate(alphaJ_list):
                    p = Params(
                        ds=args.ds,
                        N=int(n),
                        beta=float(beta),
                        L=float(L),
                        m=args.m,
                        tau=args.tau,
                        tauv=args.tauv,
                        k=args.k,
                        eta=args.etaJ,
                        alpha_J=float(alpha),
                        A_R=args.A_R,
                        sigma_R=args.sigma_R,
                    )
                    p.derive_relevant_parameters()
                    E, _, _, _ = solve_stability2(p)
                    eigenvalues_mat[l, i, j] = E.real.max()
                    pbar.update(1)

    # Dictionary containing all simulation parameters
    sim_params = {
        "ds": args.ds,
        "m": args.m,
        "tau": args.tau,
        "tauv": args.tauv,
        "k": args.k,
        "etaJ": args.etaJ,
        "A_R": args.A_R,
        "sigma_R": args.sigma_R,
        "N_min": args.N_min,
        "N_max": args.N_max,
        "num_N": args.num_N,
        "beta_list": beta_list,
        "N_list": N_list,
        "alphaJ_list": alphaJ_list,
        "L_list": L_list,
    }

    # 4. Save Computed Data and Parameter Dictionary
    np.save(save_dir / "sim_params.npy", sim_params)
    np.savez_compressed(
        save_dir / "stability_sweep_results.npz",
        eigenvalues_mat=eigenvalues_mat,
        params=sim_params,
    )
    print(f"[*] Computation complete. Data saved to {save_dir / 'stability_sweep_results.npz'}")


if __name__ == "__main__":
    main()