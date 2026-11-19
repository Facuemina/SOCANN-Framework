#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the computed eigenvalue stability sweep data from ../figure2-stability-ff-data/
and plots/saves Figure 2 (-ln(-lambda_max) vs. rho_in across beta and alpha_J).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.theory import learned_U_std


def main():
    # 1. Locate the Data Directory
    repo_root = Path(__file__).resolve().parent.parent
    # Uncomment if data if outside the repo 
    # data_dir = repo_root.parent / "figure2-stability-ff-data" / "k_1e-20"
    # Comment if data if outside the repo 
    data_dir = repo_root / "data" / "figure2-stability-ff-data" / "k_1e-2"
    data_file = data_dir / "stability_sweep_results.npz"
    base_fig_dir = repo_root  / "figures"

    if not data_file.exists():
        print(data_dir)
        print(f"[!] Error: Data file {data_file} not found.")
        print("    Make sure you have run 'scripts/figure2_compute_stability.py' first.")
        return

    # 2. Load Data and Parameters Dictionary
    loaded = np.load(data_file, allow_pickle=True)
    eigenvalues_mat = loaded["eigenvalues_mat"]
    sim_params = loaded["params"].item()

    beta_list = sim_params["beta_list"]
    N_list = sim_params["N_list"]
    alphaJ_list = sim_params["alphaJ_list"]
    sigma_R = sim_params["sigma_R"]

    # 3. Generate the 3x2 Figure Matching Figure 2
    f = 0.5
    fig = plt.figure(figsize=(10 * f, 15 * f))

    handles, labels = [], []
    num_alphas = len(alphaJ_list)

    for ib, beta in enumerate(beta_list):
        ax = fig.add_subplot(3, 2, ib + 1)
        L = 20.0 * learned_U_std(beta, sigma_R)
        rho_in = N_list / L
        var2plot = -np.log(-eigenvalues_mat[ib, :, :])

        for j, alpha in enumerate(alphaJ_list):
            color = [j / (num_alphas - 1), 0, 0]
            lbl = rf"$\alpha_J$={alpha}"
            (line,) = ax.plot(rho_in, var2plot[:, j], color=color, lw=1.8, label=lbl)
            if ib == 0:
                handles.append(line)
                labels.append(lbl)

        ax.set_xlabel(r"$\rho_{in}$")
        ax.set_ylabel(r"$-\ln(-\lambda_{max})$")
        ax.set_title(rf"$\beta$={beta}")

        M = int(np.max(var2plot))
        m = int(np.min(var2plot))
        ax.set_yticks([m, (m + M) / 2, M])

        rho_min = int(rho_in.min())
        rho_max = int(rho_in.max())
        ax.set_xticks([rho_min, (rho_min + rho_max) / 2, rho_max])

    # Shared alpha_J legend at the bottom
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # 4. Save Figure
    fig_path = base_fig_dir / "TheoreticalStabilityLambdas_rho2.png"
    fig.savefig(fig_path, format="png")
    print(f"[*] Saved Figure 2 to:\n    {fig_path}\n ")

    plt.show()


if __name__ == "__main__":
    main()