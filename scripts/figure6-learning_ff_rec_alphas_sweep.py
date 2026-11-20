#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads Figure 6 2D (alpha_J, alpha_W) sweep snapshots, computes FF/REC
correlations and autonomous recurrent speeds (via src.engine.run_basic_simulation),
and plots phase heatmaps, smoothed REC/FF stability contours, and 3x3 example
weight matrices to figures/.
"""

import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import run_basic_simulation
from src.helpers import circ_std, rectify_positions, sort_connectivity_matrix
from src.theory import gaussian, learned_U_std, learned_weight_std, theoretical_speed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot Figure 6: 2D (alpha_J, alpha_W) stability sweep."
    )
    parser.add_argument("--recompute", action="store_true", help="Recompute metrics and speeds")
    parser.add_argument("--eval_dt", type=float, default=0.005, help="Evaluation time step (s)")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm-up steps with FF+REC")
    parser.add_argument("--eval_steps", type=int, default=3000, help="Autonomous evaluation steps (FF off)")
    parser.add_argument("--tail_steps", type=int, default=1500, help="Trailing steps for mean speed")
    parser.add_argument("--corr_threshold_speed", type=float, default=0.8, help="Min REC corr to run speed test")
    parser.add_argument("--k_eval", type=float, default=1e-15, help="Global inhibition k during autonomous test")
    parser.add_argument("--u_boost", type=float, default=5.0, help="State multiplier at autonomous onset")
    parser.add_argument("--template_beta", type=float, default=.5, help="Optional beta for template width (0.5=legacy)")
    parser.add_argument("--exclude_last_alphaJ", type=int, default=1, help="Skip top N alpha_J rows (1=legacy)")
    parser.add_argument("--speed_threshold", type=float, default=15.0, help="Speed cutoff (cm/s) for REC contour")
    parser.add_argument("--ff_corr_threshold", type=float, default=0.97046238, help="Corr cutoff for FF contour")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Locate the Data Directory
    repo_root = Path(__file__).resolve().parent.parent
    # Uncomment if data is outside the repo
    # base_data_dir = repo_root.parent / "figure6-learning-ff-rec-alphas-data"
    # Comment if data is outside the repo
    base_data_dir = repo_root / "data" / "figure6-learning-ff-rec-alphas-data"
    base_fig_dir = repo_root / "figures"
    base_fig_dir.mkdir(parents=True, exist_ok=True)

    if not (base_data_dir / "sim_params.npy").exists():
        print(f"[!] Error: {base_data_dir / 'sim_params.npy'} not found.")
        print("    Make sure you have run 'scripts/script-figure6-learning_ff_rec_alphas.py' first.")
        return

    sim_params = np.load(base_data_dir / "sim_params.npy", allow_pickle=True).item()
    N = int(sim_params["N"])
    L = float(sim_params["L"])
    tau = float(sim_params["tau"])
    tauv = float(sim_params["tauv"])
    m = float(sim_params["m"])
    k = float(sim_params["k"])
    g = float(sim_params["g"])
    gain = float(sim_params["gain"])
    input_std = float(sim_params["input_std"])
    A_R = float(sim_params["A_R"])
    v = float(sim_params["v_eq"])
    alphaJ_vals = np.array(sim_params["alphaJ_vals"])
    alphaW_vals = np.array(sim_params["alphaW_vals"])
    v_th = float(theoretical_speed(m, tau, tauv, input_std))

    # 2. Load or Compute Correlations, Widths, Amplitudes, Errors, and Speeds
    cache_path = base_data_dir / "alpha_sweep_analysis.npz"
    if cache_path.exists() and not args.recompute:
        cached = np.load(cache_path, allow_pickle=True)
        corrs_J = cached["corrs_J"]
        corrs_W = cached["corrs_W"]
        speeds = cached["speeds"]
        print(f"[*] Loaded cached analysis from: {cache_path}")
    else:
        if args.template_beta is not None:
            std_J_tpl = float(learned_weight_std(args.template_beta, input_std))
            std_W_tpl = std_J_tpl
        else:
            std_U = float(sim_params.get("std_U", learned_U_std(sim_params["betaJ"], input_std)))
            std_J_tpl = float(sim_params.get("std_J", learned_weight_std(sim_params["betaJ"], input_std)))
            std_W_tpl = float(sim_params.get("std_W", learned_weight_std(sim_params["betaW"], std_U / np.sqrt(2.0))))

        pos = jnp.linspace(0.0, L, N, endpoint=False).reshape((N, 1))
        J_eq = np.array(gaussian(pos, pos.T, std_J_tpl, L))
        J_eq = J_eq / J_eq.sum(axis=1, keepdims=True)
        W_eq = np.array(gaussian(pos, pos.T, std_W_tpl, L))
        W_eq = (W_eq / W_eq.sum(axis=1, keepdims=True)) * (1.0 - np.eye(N))

        shape = (len(alphaJ_vals), len(alphaW_vals))
        corrs_J = np.full(shape, np.nan)
        corrs_W = np.full(shape, np.nan)
        errors_J = np.full(shape, np.nan)
        errors_W = np.full(shape, np.nan)
        amps_J = np.full(shape, np.nan)
        amps_W = np.full(shape, np.nan)
        std_W_mat = np.full(shape, np.nan)
        speeds = np.zeros(shape)

        input_params_eval = (args.eval_dt, input_std, L, 0.0, A_R, 0.0)
        neural_warm = (1.0 / tau, 1.0 / tauv, m, k, g, gain)
        neural_eval = (1.0 / tau, 1.0 / tauv, m, args.k_eval, g, gain)
        v_warm = 2.0 * v * jnp.ones(args.warmup_steps)
        v_eval = v * jnp.ones(args.eval_steps)

        max_i = len(alphaJ_vals) - args.exclude_last_alphaJ if args.exclude_last_alphaJ > 0 else len(alphaJ_vals)
        pairs = [(i, j) for i in range(max_i) for j in range(len(alphaW_vals))]

        for i, j in tqdm(pairs, desc="Analyzing (alpha_J, alpha_W) grid"):
            snap_files = sorted(glob.glob(str(base_data_dir / f"alphaJ-{i}_alphaW-{j}" / "snapshot_*.npz")))
            if not snap_files:
                continue

            final_data = np.load(snap_files[-1], allow_pickle=True)
            J_s, idx_sort = sort_connectivity_matrix(jnp.array(final_data["W_ff"]))
            Jor = np.array(J_s)
            Wor = np.array(jnp.array(final_data["W_rec"])[idx_sort][:, idx_sort])

            j_sum = Jor.sum(axis=1, keepdims=True)
            w_sum = Wor.sum(axis=1, keepdims=True)
            J = np.divide(Jor, j_sum, out=np.zeros_like(Jor), where=j_sum != 0)
            W = np.divide(Wor, w_sum, out=np.zeros_like(Wor), where=w_sum != 0)

            corrs_J[i, j] = np.corrcoef(J_eq.ravel(), J.ravel())[0, 1]
            corrs_W[i, j] = np.corrcoef(W_eq.ravel(), W.ravel())[0, 1]
            errors_J[i, j] = np.sum((J_eq - J) ** 2) / (np.sum(J_eq) ** 2)
            errors_W[i, j] = np.sum((W_eq - W) ** 2) / (np.sum(W_eq) ** 2)

            std_j = float(np.mean(circ_std(J, L)))
            std_w = float(np.mean(circ_std(W, L)))
            std_W_mat[i, j] = std_w
            amps_J[i, j] = J.max() / std_j if std_j > 0 else np.nan
            amps_W[i, j] = W.max() / std_w if std_w > 0 else np.nan

            if np.isfinite(corrs_W[i, j]) and corrs_W[i, j] > args.corr_threshold_speed:
                J_jax = jnp.array(Jor)
                W_jax = jnp.array(Wor)
                res, _ = run_basic_simulation(
                    J_jax[:, :1], J_jax[:, -10:-9] * m, J_jax, W_jax,
                    0.0, v_warm, neural_warm, input_params_eval, pos, args.warmup_steps,
                )
                _, hist = run_basic_simulation(
                    res[0] * args.u_boost, res[1] * args.u_boost, J_jax * 0.0, W_jax,
                    0.0, v_eval, neural_eval, input_params_eval, pos, args.eval_steps,
                )
                unwrapped = np.array(rectify_positions(hist[1], L))
                s = float(np.abs(np.diff(unwrapped)[-args.tail_steps:].mean()) / args.eval_dt)
                if not np.isnan(s):
                    speeds[i, j] = s

        np.savez_compressed(
            cache_path,
            alphaJ_vals=alphaJ_vals,
            alphaW_vals=alphaW_vals,
            corrs_J=corrs_J,
            corrs_W=corrs_W,
            errors_J=errors_J,
            errors_W=errors_W,
            amps_J=amps_J,
            amps_W=amps_W,
            std_W_mat=std_W_mat,
            speeds=speeds,
            params=sim_params,
        )
        print(f"[*] Saved analysis cache to: {cache_path}")

    # 3. Plot 3-Panel Phase Heatmaps
    alpha_W_ticks = np.logspace(-4, 4, 9) if len(alphaW_vals) > 10 else [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    alpha_J_ticks = [1e-1, 1e0, 1e1, 1e2]

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 4.5))
    panel_specs = [
        (axes1[0], corrs_J, "jet", None, None, "FF matrix corr", np.arange(0, 1.2, 0.2), True),
        (axes1[1], corrs_W, "jet", None, None, "Rec matrix corr", np.arange(0, 1.2, 0.2), True),
        (axes1[2], speeds / v_th, "magma", 0.0, 2.0, r"$v / v_0(m)$", [0, 0.5, 1, 1.5, 2], False),
    ]
    for ax, arr, cmap, vmin, vmax, lbl, ticks, diag in panel_specs:
        mesh = ax.pcolormesh(
            alphaW_vals, alphaJ_vals, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        cbar = fig1.colorbar(mesh, ax=ax, label=lbl)
        cbar.set_ticks(ticks)
        ax.set_xlabel(r"$\alpha_W$", fontsize=20)
        ax.set_ylabel(r"$\alpha_J$", fontsize=20)
        ax.set_xticks(alpha_W_ticks)
        ax.set_yticks(alpha_J_ticks)
        ax.tick_params(axis="both", labelsize=16)
        if diag:
            ax.plot([1e-4, 1e4], [1e-4, 1e4], "k--")
        ax.set_xlim([alphaW_vals.min() * 0.5, alphaW_vals.max() * 1.4])
        ax.set_ylim([alphaJ_vals.min() * 0.5, alphaJ_vals.max() * 1.4])

    fig1.tight_layout()
    # fig1.savefig(base_fig_dir / "figure6-phase_heatmaps.svg", format="svg")
    fig1.savefig(base_fig_dir / "figure6-phase_heatmaps.png", format="png", dpi=300)

    # 4. Smoothed REC and FF Stability Contours
    example_pairs = [
        (i_idx, j_idx)
        for j_idx in (0, 3, 6)
        for i_idx in (1, 2, 4)
        if i_idx < len(alphaJ_vals) and j_idx < len(alphaW_vals)
    ]
    label_map = {pair: idx + 1 for idx, pair in enumerate(example_pairs)}

    clean_rec = np.copy(speeds)
    clean_rec[clean_rec < args.speed_threshold] = 0.0
    clean_rec[clean_rec > 0] = np.log(1.0 / (clean_rec[clean_rec > 0] / v_th) * 30.0) * 9.0

    clean_ff = np.zeros_like(corrs_J)
    clean_ff += 1.5 * corrs_J * (corrs_J > args.ff_corr_threshold)
    clean_ff[1] = 0.0

    contour_specs = [
        ("figure6-stability_contour_rec", clean_rec, args.speed_threshold, 10, 2.0),
        ("figure6-stability_contour_ff", clean_ff, args.ff_corr_threshold, 5, 1.0),
    ]
    for stem, grid, thresh, upscale, sigma in contour_specs:
        smoothed = gaussian_filter(zoom(grid, upscale, order=1), sigma=sigma)
        W_fine = np.logspace(np.log10(alphaW_vals.min()), np.log10(alphaW_vals.max()), smoothed.shape[1])
        J_fine = np.logspace(np.log10(alphaJ_vals.min()), np.log10(alphaJ_vals.max()), smoothed.shape[0])

        fig_c, ax_c = plt.subplots(figsize=(6, 5))
        # if smoothed.max() > thresh:
        #     ax_c.contourf(W_fine, J_fine, smoothed, levels=[thresh, smoothed.max()], colors=["#2ca02c"], alpha=0.3)
        #     ax_c.contour(W_fine, J_fine, smoothed, levels=[thresh], colors=["darkgreen"], linewidths=2.5)
        s_max = np.nanmax(smoothed)
        if np.isfinite(s_max) and s_max > thresh:
            ax_c.contourf(W_fine, J_fine, smoothed, levels=[thresh, s_max + 1e-6], colors=["#2ca02c"], alpha=0.3)
            ax_c.contour(W_fine, J_fine, smoothed, levels=[thresh], colors=["darkgreen"], linewidths=2.5)
        for j_idx, w in enumerate(alphaW_vals[:-1]):
            for i_idx, j_val in enumerate(alphaJ_vals):
                color = "forestgreen" if grid[i_idx, j_idx] > 0 else "firebrick"
                ax_c.scatter(w, j_val, color=color, s=120, edgecolors="white", zorder=10)
                if (i_idx, j_idx) in label_map:
                    ax_c.text(w * 1.2, j_val * 1.2, str(label_map[(i_idx, j_idx)]), fontsize=11, fontweight="bold", zorder=11)

        ax_c.plot([0.01, 100.0], [0.01, 100.0], "k--")
        ax_c.set_xscale("log")
        ax_c.set_yscale("log")
        ax_c.set_xlim([0.02, 300.0])
        ax_c.set_ylim([0.1, 13.0])
        ax_c.set_xlabel(r"$\alpha_W$", fontsize=15)
        ax_c.set_ylabel(r"$\alpha_J$", fontsize=15)
        ax_c.set_xticks([0.1, 1.0, 10.0, 100.0])
        ax_c.set_xticklabels([r"$10^{-1}$", r"$10^0$", r"$10^1$", r"$10^2$"], fontsize=15)
        ax_c.set_yticks([0.1, 1.0, 10.0])
        ax_c.set_yticklabels([r"$10^{-1}$", r"$10^0$", r"$10^1$"], fontsize=15)
        ax_c.grid(True, which="both", ls="-", alpha=0.1)
        fig_c.tight_layout()
        # fig_c.savefig(base_fig_dir / f"{stem}.svg", format="svg")
        fig_c.savefig(base_fig_dir / f"{stem}.png", format="png", dpi=300)

    # 5. 3x3 Example Sorted Matrices (FF and REC)
    if len(example_pairs) == 9:
        for key, stem, show_axes in [("W_ff", "figure6-examples_ff", False), ("W_rec", "figure6-examples_rec", True)]:
            fig_m, axes_m = plt.subplots(3, 3, figsize=(15, 15))
            for idx, (i_idx, j_idx) in enumerate(example_pairs):
                ax_m = axes_m[2 - (idx % 3), idx // 3]
                snap_files = sorted(glob.glob(str(base_data_dir / f"alphaJ-{i_idx}_alphaW-{j_idx}" / "snapshot_*.npz")))
                if not snap_files:
                    ax_m.axis("off")
                    continue
                d = np.load(snap_files[-1], allow_pickle=True)
                J_s, idx_s = sort_connectivity_matrix(jnp.array(d["W_ff"]))
                mat = np.array(J_s) if key == "W_ff" else np.array(jnp.array(d["W_rec"])[idx_s][:, idx_s])
                ax_m.imshow(mat, cmap="jet", rasterized=True)
                ax_m.set_title(f"{idx + 1}", fontsize=28 if not show_axes else 20)
                if show_axes:
                    ax_m.set_xticks([0, N])
                    ax_m.set_yticks([0, N])
                    ax_m.tick_params(axis="both", labelsize=16)
                    ax_m.set_xlabel("Input neurons", fontsize=18)
                    ax_m.set_ylabel("Competitive neurons", fontsize=18)
                else:
                    ax_m.axis("off")
            fig_m.tight_layout()
            # fig_m.savefig(base_fig_dir / f"{stem}.svg", format="svg")
            fig_m.savefig(base_fig_dir / f"{stem}.png", format="png", dpi=300)

    print(f"[*] Saved Figure 6 outputs to: {base_fig_dir}")
    plt.show()


if __name__ == "__main__":
    main()