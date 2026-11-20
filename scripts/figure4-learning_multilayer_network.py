#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the multilayer feedforward simulation data from ../figure4-learning-multilayer-ff-data/,
computes the learning correlations across snapshots, evaluates the sorted bump profiles
and predictive shifts, and saves the combined Figure 4 (panels b, c, d layout).
"""

import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import run_multilayer_ff_evaluation
from src.helpers import center_of_mass, sort_connectivity_matrix
from src.theory import gamma, learned_U_std, learned_weight_std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 4 (panels b, c, d) from saved multilayer FF data."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional explicit path to a run folder inside ../figure4-learning-multilayer-ff-data/",
    )
    parser.add_argument("--N", type=int, default=512, help="Number of neurons matching the run")
    parser.add_argument("--L", type=float, default=70.0, help="Domain length matching the run")
    parser.add_argument("--beta", type=float, default=0.5, help="beta_J matching the run")
    parser.add_argument("--v", type=float, default=20.0, help="Speed v (cm/s) matching the run")
    parser.add_argument("--m", type=float, default=0.25, help="Adaptation m matching the run")
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Optional upper limit in hours for the correlation x-axis (e.g., 500)",
    )
    parser.add_argument(
        "--recompute_corrs",
        action="store_true",
        help="Force recomputation of snapshot correlations instead of loading corrs_cache.npz",
    )
    return parser.parse_args()


def normalize_rows(W):
    """Safe row-wise normalization avoiding division by zero."""
    row_sums = np.sum(W, axis=1, keepdims=True)
    return W / np.where(row_sums == 0, 1.0, row_sums)


def main():
    args = parse_args()

    # 1. Locate Run Directory
    repo_root = Path(__file__).resolve().parent.parent
    base_data_dir = repo_root.parent / "figure4-learning-multilayer-ff-data"

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = (
            base_data_dir
            / f"N{args.N}_L{args.L:g}_beta{args.beta:g}_v{args.v:g}_m{args.m:g}"
        )

    if not run_dir.exists():
        print(f"[!] Error: Run directory {run_dir} not found.")
        print("    Make sure you have run 'scripts/figure4_learning_multilayer_ff.py' first.")
        return

    # 2. Load Parameters & Snapshots
    sim_params = np.load(run_dir / "sim_params.npy", allow_pickle=True).item()
    snapshot_files = sorted(glob.glob(str(run_dir / "snapshot_*.npz")))

    if len(snapshot_files) < 2:
        print(f"[!] Error: Not enough snapshots found in {run_dir}.")
        return

    num_layers = sim_params["num_layers"]
    n = sim_params["N"]
    L = sim_params["L"]
    dt = sim_params["dt"]
    tau = sim_params["tau"]
    tauv = sim_params["tauv"]
    beta = sim_params["betaJ"]
    m = sim_params["m"]
    v = sim_params["speed"]
    k = sim_params["k"]
    input_std = sim_params["input_std"]
    R0 = sim_params["A_R"]
    u_std = learned_U_std(beta, input_std)

    colors = ["red", "orange", "green", "m"]
    if num_layers > len(colors):
        cmap_extra = plt.cm.tab10(np.linspace(0, 1, num_layers))
        colors = [cmap_extra[l] for l in range(num_layers)]

    # 3. Sort Final Connectivity Matrices & Save 2x2 Diagnostic Figure
    final_snap = np.load(snapshot_files[-1], allow_pickle=True)
    W_final = final_snap["W_ff"]  # shape: (num_layers, N, N)

    idx = np.arange(n)
    final_idx_list = []
    fig_w, axes_w = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes_w.flatten()

    for l in range(num_layers):
        W = W_final[l]
        sorted_W, idx_jnp = sort_connectivity_matrix(jnp.array(W[:, idx]))
        idx = np.array(idx_jnp)
        final_idx_list.append(idx)

        if l < len(axes_flat):
            im = axes_flat[l].imshow(np.array(sorted_W), cmap="jet", aspect="auto")
            axes_flat[l].set_title(f"Layer {l + 1} | J max = {W.max():.2e}")
            fig_w.colorbar(im, ax=axes_flat[l])

    fig_w.tight_layout()
    fig_w.savefig(run_dir / "W_ff_sorted_final.png", dpi=200)
    plt.close(fig_w)
    print(f"[*] Loaded {len(snapshot_files)} snapshots. Final snapshot: {Path(snapshot_files[-1]).name}")

    # 4. Compute (or Load Cached) Learning Correlations Across Snapshots
    corr_cache_file = run_dir / "corrs_cache.npz"
    use_cache = (
        corr_cache_file.exists()
        and not args.recompute_corrs
    )

    if use_cache:
        cached = np.load(corr_cache_file)
        if len(cached["times_hr"]) == len(snapshot_files):
            corrs = cached["corrs"]
            times_hr = cached["times_hr"]
            print(f"[*] Loaded cached correlations from {corr_cache_file.name}")
        else:
            use_cache = False

    if not use_cache:
        print("[*] Computing correlations against theoretical Gaussian profile...")
        x_grid = np.linspace(0, L, n, endpoint=False).reshape(-1, 1)
        dx = np.minimum(np.abs(x_grid - x_grid.T), L - np.abs(x_grid - x_grid.T))
        sJ2 = learned_weight_std(beta, input_std) ** 2
        J_th = np.exp(-0.5 * (dx ** 2) / sJ2)
        J_th_flat = normalize_rows(J_th).flatten()

        corrs = np.zeros((num_layers, len(snapshot_files)))
        times_hr = np.zeros(len(snapshot_files))

        for s_idx, snap_path in enumerate(tqdm(snapshot_files, desc="Correlations")):
            snap = np.load(snap_path, allow_pickle=True)
            W_all = snap["W_ff"]
            if "time_hours" in snap:
                times_hr[s_idx] = float(snap["time_hours"])
            else:
                chunk_num = int(Path(snap_path).stem.split("_")[1])
                times_hr[s_idx] = chunk_num * sim_params["steps_per_chunk"] * dt / 3600.0

            idx_curr = np.arange(n)
            for l in range(num_layers):
                W_norm = normalize_rows(W_all[l])
                sorted_W, idx_jnp = sort_connectivity_matrix(jnp.array(W_norm[:, idx_curr]))
                idx_curr = np.array(idx_jnp)
                corrs[l, s_idx] = np.corrcoef(np.array(sorted_W).flatten(), J_th_flat)[0, 1]

        np.savez_compressed(corr_cache_file, corrs=corrs, times_hr=times_hr)
        print(f"[*] Cached correlations to {corr_cache_file.name}")

    # 5. Load Evaluation Data (or Run Evaluation Fallback if Missing)
    eval_file = run_dir / "eval_results.npz"
    pos = np.linspace(0, L, n, endpoint=False)

    if eval_file.exists():
        eval_data = np.load(eval_file, allow_pickle=True)
        x_eval = eval_data["x_eval"]    # (eval_positions,)
        I0_eval = eval_data["I0_eval"]  # (eval_positions, N)
        fU_eval = eval_data["fU_eval"]  # (eval_positions, num_layers, N)
    else:
        print("[*] eval_results.npz not found; running fixed-weight evaluation on latest snapshot...")
        neural_params = (
            1.0 / tau,
            1.0 / tauv,
            m,
            k,
            sim_params["g"],
            sim_params["gain"],
        )
        input_params = (dt, input_std, L, v, R0, sim_params["av"])
        pos_jnp = jnp.array(pos).reshape((n, 1))

        U_init = tuple(jnp.array(u) for u in final_snap["U"])
        V_init = tuple(jnp.array(vv) for vv in final_snap["V"])
        W_init = tuple(jnp.array(w) for w in W_final)
        x_init = float(final_snap["x_stimulus"])

        T_lap = int(np.round(L / (v * dt)))
        v_eval = jnp.ones(T_lap) * v
        _, (x_hist, I0_hist, fU_hist) = run_multilayer_ff_evaluation(
            U_init, V_init, W_init, x_init, v_eval, neural_params, input_params, pos_jnp, T=T_lap
        )

        eval_positions = sim_params.get("eval_positions", 100)
        sample_idx = np.linspace(0, T_lap - 1, eval_positions, endpoint=False, dtype=int)
        x_eval = np.array(x_hist)[sample_idx]
        I0_eval = np.array(I0_hist)[sample_idx, :, 0]
        fU_eval = np.array(fU_hist)[sample_idx, :, :, 0]

    # Permute evaluated firing rates into sorted spatial order using final chained indices
    fU_sorted = np.zeros_like(fU_eval)
    for l in range(num_layers):
        fU_sorted[:, l, :] = fU_eval[:, l, final_idx_list[l]]

    # 6. Compute Center-of-Mass Predictive Shifts Over Evaluated Positions
    n_eval = len(x_eval)
    COM = np.zeros((num_layers + 1, n_eval))
    COM[0, :] = x_eval
    pos_jnp_1d = jnp.array(pos)

    for ii in range(n_eval):
        for l in range(num_layers):
            COM[l + 1, ii] = float(
                center_of_mass(pos_jnp_1d, jnp.array(fU_sorted[ii, l]), L)
            )

    Deltas = np.minimum(np.abs(COM[1:, :] - COM[0, :]), L - np.abs(COM[1:, :] - COM[0, :]))
    mD = np.median(Deltas, axis=1)
    Q1 = np.quantile(Deltas, 0.25, axis=1)
    Q3 = np.quantile(Deltas, 0.75, axis=1)
    IQR = Q3 - Q1

    # Pick example position closest to center of domain (L / 2) to avoid periodic wrap in plot
    ex_idx = int(np.argmin(np.abs(x_eval - L / 2.0)))
    xf0 = float(x_eval[ex_idx])

    # 7. Combined Figure (Panels b, c, d layout without panel letters)
    FONTSIZE = 25
    MARKERSIZE = 20
    LINEWIDTH = 3

    fig = plt.figure(figsize=(20, 10))

    # --- Left Panel (121): Learning Correlation vs. Time (hr) ---
    ax_corr = fig.add_subplot(121)
    if args.max_hours is not None:
        t_mask = times_hr <= args.max_hours
    else:
        t_mask = np.ones(len(times_hr), dtype=bool)

    for l in range(num_layers):
        ax_corr.plot(
            times_hr[t_mask],
            corrs[l, t_mask],
            color=colors[l],
            label=f"Layer {l + 1}",
            linewidth=LINEWIDTH,
        )

    ax_corr.set_xlabel("Time (hr)", fontsize=FONTSIZE)
    ax_corr.set_ylabel("corr", fontsize=FONTSIZE)
    ax_corr.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    max_t_plot = float(times_hr[t_mask].max())
    if max_t_plot <= 600:
        ax_corr.set_xticks(np.arange(0, 600, 100))
    ax_corr.tick_params(axis="both", labelsize=FONTSIZE)
    ax_corr.legend(fontsize=FONTSIZE - 5)

    # --- Top-Right Panel (222): Example Normalized Firing Rate Bumps ---
    ax_bump = fig.add_subplot(222)
    I0_ex = I0_eval[ex_idx]
    ax_bump.plot(
        pos,
        I0_ex / (I0_ex.max() + 1e-12),
        "b",
        label="Input Layer",
        linewidth=LINEWIDTH,
    )
    for l in range(num_layers):
        fU_ex = fU_sorted[ex_idx, l]
        ax_bump.plot(
            pos,
            fU_ex / (fU_ex.max() + 1e-12),
            color=colors[l],
            label=f"Layer {l + 1}",
            linewidth=LINEWIDTH,
        )

    ax_bump.legend(fontsize=FONTSIZE - 5)
    ax_bump.set_xlim([xf0 - 5 * input_std, xf0 + 5 * input_std])
    ax_bump.set_yticks([0, 1])
    if np.isclose(L, 70.0):
        ax_bump.set_xticks([20, 30, 40, 50, 60])
    ax_bump.tick_params(axis="both", labelsize=FONTSIZE)
    ax_bump.set_ylabel("Normalized Firing Rate", fontsize=FONTSIZE)
    ax_bump.set_xlabel("x (cm)", fontsize=FONTSIZE)

    # --- Bottom-Right Panel (224): Predictive Shift Delta vs. Layer ---
    ax_shift = fig.add_subplot(224)
    # Dummy off-axis entry to give the legend a neutral gray Simulation marker
    ax_shift.errorbar(
        [10],
        [10],
        [5],
        capsize=10,
        linestyle="",
        marker="*",
        color="gray",
        markersize=MARKERSIZE,
        label="Simulation",
    )

    g_asym = gamma(m, v * tau / (np.sqrt(2) * u_std), tauv / tau) * np.sqrt(2)
    layer_indices = np.arange(num_layers + 1)
    ax_shift.plot(
        layer_indices,
        g_asym * u_std * layer_indices,
        "k--",
        label="Theory",
        linewidth=LINEWIDTH,
    )
    ax_shift.plot([0], [0], "bo", markersize=MARKERSIZE)

    for l in range(1, num_layers + 1):
        lower_err = max(mD[l - 1] - Q1[l - 1] + 1.5 * IQR[l - 1], 0.0)
        upper_err = max(Q3[l - 1] + 1.5 * IQR[l - 1] - mD[l - 1], 0.0)
        ax_shift.errorbar(
            [l],
            [mD[l - 1]],
            yerr=np.array([[lower_err], [upper_err]]),
            capsize=10,
            marker="*",
            color=colors[l - 1],
            markersize=MARKERSIZE,
        )

    ax_shift.set_yticks([0, 2, 4, 6])
    ax_shift.set_xticks(layer_indices)
    ax_shift.set_xticklabels(
        ["Input"] + [f"Layer {l}" for l in range(1, num_layers + 1)],
        fontsize=FONTSIZE,
    )
    ax_shift.tick_params(axis="y", labelsize=FONTSIZE)
    ax_shift.set_ylabel(r"$\Delta$ (cm)", fontsize=FONTSIZE)
    ax_shift.legend(fontsize=FONTSIZE - 5, loc="upper left")
    ax_shift.set_xlim([-0.2, num_layers + 0.2])
    ax_shift.set_ylim([-0.2, 8.0])

    fig.tight_layout()

    # 8. Save Combined Figure
    svg_path = run_dir / "Multilayer_Figure4.svg"
    png_path = run_dir / "Multilayer_Figure4.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"[*] Saved Figure 4 to:\n    {svg_path}\n    {png_path}")

    plt.show()


if __name__ == "__main__":
    main()