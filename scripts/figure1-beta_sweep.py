#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the compressed snapshots from the Figure 1 feedforward beta-sweep,
computes the learned synaptic width (sigma_J) and amplitude (A_J) statistics
across neurons, and plots/saves them against the theoretical predictions.
"""

import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.helpers import sort_connectivity_matrix, circ_std
from src.theory import learned_weight_std, learned_weight_amplitude


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 1(b-c) and sorted matrices from FF beta sweep data."
    )
    parser.add_argument("--N", type=int, default=512, help="Number of neurons matching the sweep")
    parser.add_argument("--total_steps", type=int, default=2000000, help="Total simulation steps matching the sweep")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        help="List of beta_J values swept over",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Locate the Data Directory 
    repo_root = Path(__file__).resolve().parent.parent
    
    # uncomment if data is outside the repo
    # base_data_dir = repo_root.parent / "figure1-learning-ff-data"
    
    # comment if data is outside the repo
    base_data_dir = repo_root / "data" / "figure1-learning-ff-data"
    
    base_fig_dir = repo_root  / "figures"

    if not base_data_dir.exists():
        print(f"[!] Error: Data directory {base_data_dir} not found.")
        print("    Make sure you have run 'scripts/scripts-figure1_learning_ff.py' first.")
        return

    betas = np.array(args.betas)
    data_std = np.zeros((len(betas), 5))
    data_J = np.zeros((len(betas), 5))

    sim_params = None
    # 2. Extract Statistics from Final Snapshots across Beta Sweep
    for ib, beta in enumerate(betas):
        run_dir = base_data_dir / f"beta_{beta:.2f}_N{args.N}_steps{args.total_steps}"
        snapshot_files = sorted(glob.glob(str(run_dir / "snapshot_*.npz")))

        if not snapshot_files:
            raise FileNotFoundError(f"[!] No snapshots found in {run_dir}")

        final_file = snapshot_files[-1]
        print(f"[*] Loading beta = {beta:.2f} final state: {Path(final_file).name}")

        final_data = np.load(final_file, allow_pickle=True)
        W = final_data["W_ff"]

        # Load simulation parameters dictionary
        if "params" in final_data:
            sim_params = final_data["params"].item()
        else:
            sim_params = np.load(run_dir / "sim_params.npy", allow_pickle=True).item()

        L = sim_params["L"]

        # Circular standard deviation across neurons
        std = circ_std(W, L)
        data_std[ib, 0] = std.mean()
        data_std[ib, 1] = std.std()
        data_std[ib, 2] = np.median(std)
        data_std[ib, 3] = np.percentile(std, 25)
        data_std[ib, 4] = np.percentile(std, 75)

        # Learned weight amplitude A_J across neurons
        A_J = W.max(axis=1) * np.sqrt(2 * np.pi * data_std[ib, 2] ** 2)
        data_J[ib, 0] = A_J.mean()
        data_J[ib, 1] = A_J.std()
        data_J[ib, 2] = np.median(A_J)
        data_J[ib, 3] = np.percentile(A_J, 25)
        data_J[ib, 4] = np.percentile(A_J, 75)

        # Plot and save sorted final connectivity matrix for each beta
        W_sorted, _ = sort_connectivity_matrix(jnp.array(W))
        fig_w, ax_w = plt.subplots()
        im = ax_w.imshow(np.array(W_sorted), aspect="auto")
        ax_w.set_title(rf"Sorted $W_{{\mathrm{{ff}}}}$ ($\beta = {beta:.2f}$)")
        fig_w.colorbar(im, ax=ax_w)
        fig_w.tight_layout()
        #fig_w.savefig(run_dir / f"W_ff_sorted_beta_{beta:.2f}.svg", format="svg")

    # 3. Plot Figure 1(b) and 1(c): Learned Width and Amplitude vs. Theory
    FONTSIZE = 20
    fig = plt.figure(figsize=(8, 4))

    input_std = sim_params["input_std"]
    A_R = sim_params["A_R"]
    alphaJ = sim_params["alphaJ"]

    th_betas = np.linspace(0.01, 1.6, 50)
    theoretical_std = learned_weight_std(th_betas, input_std)
    theoretical_AJ = learned_weight_amplitude(th_betas, input_std, R0=A_R, alpha=alphaJ)

    # --- Panel (b): Learned Standard Deviation sigma_J ---
    median_std = data_std[:, 2]
    q1_std = data_std[:, 3]
    q3_std = data_std[:, 4]
    asymmetric_error_std = np.vstack((median_std - q1_std, q3_std - median_std))

    ax1 = fig.add_subplot(121)
    ax1.plot(th_betas, theoretical_std, "k--", label="Theory")
    ax1.errorbar(
        betas,
        median_std,
        yerr=asymmetric_error_std,
        fmt="*",
        color="r",
        capsize=5,
        label="Simulation",
    )
    ax1.set_xlabel(r"$\beta$", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$\sigma_J$ (cm)", fontsize=FONTSIZE)
    ax1.set_yticks([0.0, 5, 10, 15])
    ax1.set_xticks([0, 0.5, 1, 1.5])
    ax1.tick_params(axis="both", labelsize=FONTSIZE)
    ax1.legend(fontsize=int(0.75 * FONTSIZE))

    # --- Panel (c): Learned Weight Amplitude A_J ---
    median_J = data_J[:, 2]
    q1_J = data_J[:, 3]
    q3_J = data_J[:, 4]
    IQR_J = q3_J - q1_J

    lower_error_J = np.maximum(-q1_J + 1.5 * IQR_J + median_J, 0)
    upper_error_J = np.maximum(-median_J + (1.5 * IQR_J + q3_J), 0)
    asymmetric_error_J = np.vstack((lower_error_J, upper_error_J))

    ax2 = fig.add_subplot(122)
    ax2.plot(th_betas[7:], theoretical_AJ[7:], "k--", label="Theory")
    ax2.errorbar(
        betas,
        median_J,
        yerr=asymmetric_error_J,
        fmt="*",
        color="r",
        capsize=5,
        label="Simulation",
    )
    ax2.set_xlabel(r"$\beta$", fontsize=FONTSIZE)
    ax2.set_ylabel(r"$A_J$ (a.u.)", fontsize=FONTSIZE)
    ax2.set_yticks([25, 50, 75, 100])
    ax2.set_xticks([0, 0.5, 1, 1.5])
    ax2.tick_params(axis="both", labelsize=FONTSIZE)
    ax2.legend(fontsize=int(0.75 * FONTSIZE))

    fig.tight_layout()

    # Save SVG and EPS outputs in ../figure1-learning-ff-data/
    fig_path = base_fig_dir / "figure1-beta_sweep.png"
    fig.savefig(fig_path, format="png")
    print(f"[*] Saved figures to:\n    {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()