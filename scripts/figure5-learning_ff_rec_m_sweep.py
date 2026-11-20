#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads Figure 5 FF+REC learning m-sweep data, plots FF and REC connectivity
correlations vs. time for a selected adaptation value, evaluates autonomous
recurrent bump speed across m, and saves the figure to figures/.
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
from src.engine import (
    create_gaussian_connectivity_matrix,
    run_basic_simulation,
    set_initial_conditions,
)
from src.helpers import circ_std, rectify_positions, sort_connectivity_matrix
from src.theory import gaussian, theoretical_speed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 5 (FF+REC learning correlations and intrinsic bump speed)."
    )
    parser.add_argument("--corr_m_idx", type=int, default=2, help="m index for correlation curve")
    parser.add_argument("--from_snapshots", action="store_true", help="Recompute correlations from snapshots")
    parser.add_argument("--recompute_speeds", action="store_true", help="Recompute autonomous speed stats")
    parser.add_argument("--use_analytical_weights", action="store_true", help="Debug: use analytical Gaussian weights")
    parser.add_argument("--eval_dt", type=float, default=None, help="Evaluation dt (default: sim dt/2 if >=0.01)")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm-up steps with FF+REC")
    parser.add_argument("--eval_steps", type=int, default=10000, help="Autonomous evaluation steps (FF off)")
    parser.add_argument("--tail_steps", type=int, default=1000, help="Trailing steps for speed stats")
    parser.add_argument("--k_scale", type=float, default=0.1, help="k multiplier during autonomous speed test")
    parser.add_argument("--u_boost", type=float, default=100.0, help="U multiplier at onset of autonomous test")
    parser.add_argument("--exclude_last", type=int, default=2, help="Highest-m points excluded in right panel")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Locate the Data Directory
    repo_root = Path(__file__).resolve().parent.parent
    # Uncomment if data is outside the repo
    # base_data_dir = repo_root.parent / "figure5-learning-ff-rec-data"
    # Comment if data is outside the repo
    base_data_dir = repo_root / "data" / "figure5-learning-ff-rec-data"
    base_fig_dir = repo_root / "figures"
    base_fig_dir.mkdir(parents=True, exist_ok=True)

    if not (base_data_dir / "sim_params.npy").exists():
        print(f"[!] Error: {base_data_dir / 'sim_params.npy'} not found.")
        print("    Make sure you have run 'scripts/script-figure5-learning_ff_rec_m_sweep.py' first.")
        return

    sim_params = np.load(base_data_dir / "sim_params.npy", allow_pickle=True).item()
    N = int(sim_params["N"])
    L = float(sim_params["L"])
    tau = float(sim_params["tau"])
    tauv = float(sim_params["tauv"])
    k = float(sim_params["k"])
    g = float(sim_params["g"])
    gain = float(sim_params["gain"])
    input_std = float(sim_params["input_std"])
    A_R = float(sim_params["A_R"])
    m_vals = np.array(sim_params["m_vals"])
    sim_dt = float(sim_params["dt"])
    eval_dt = args.eval_dt if args.eval_dt is not None else (sim_dt / 2.0 if sim_dt >= 0.01 else sim_dt)

    # 2. Load or Compute Correlations for Selected m Index
    corr_dir = base_data_dir / f"m-{args.corr_m_idx}"
    hist_path = corr_dir / "learning_history.npz"

    if not args.from_snapshots and hist_path.exists():
        hist = np.load(hist_path, allow_pickle=True)
        time_hr = hist["time_s"] / 3600.0
        corr_FF = hist["corr_ff"]
        corr_Rec = hist["corr_rec"]
    else:
        snap_files = sorted(glob.glob(str(corr_dir / "snapshot_*.npz")))
        if not snap_files:
            raise FileNotFoundError(f"[!] No snapshots found in {corr_dir}")
        Jeq_flat = np.array(create_gaussian_connectivity_matrix(N, input_std, L)).ravel()
        corr_FF = np.zeros(len(snap_files))
        corr_Rec = np.zeros(len(snap_files))
        time_hr = np.zeros(len(snap_files))
        for idx_s, fpath in enumerate(tqdm(snap_files, desc=f"Correlations m-{args.corr_m_idx}")):
            snap = np.load(fpath, allow_pickle=True)
            J = snap["W_ff"] / snap["W_ff"].sum(axis=1, keepdims=True)
            W = snap["W_rec"] / snap["W_rec"].sum(axis=1, keepdims=True)
            J_s, idx_sort = sort_connectivity_matrix(jnp.array(J))
            idx_np = np.array(idx_sort)
            corr_FF[idx_s] = np.corrcoef(np.array(J_s).ravel(), Jeq_flat)[0, 1]
            corr_Rec[idx_s] = np.corrcoef(W[idx_np][:, idx_np].ravel(), Jeq_flat)[0, 1]
            chunk_num = int(Path(fpath).stem.split("_")[1])
            time_hr[idx_s] = chunk_num * sim_params["steps_per_chunk"] * sim_dt / 3600.0

    # 3. Load or Evaluate Autonomous Recurrent Bump Speeds across m
    speed_cache = base_data_dir / "speed_analysis.npz"
    if speed_cache.exists() and not args.recompute_speeds and not args.use_analytical_weights:
        cached = np.load(speed_cache, allow_pickle=True)
        empirical_speeds_stats = cached["empirical_speeds_stats"]
        connectivity_std = cached["connectivity_std"]
        print(f"[*] Loaded cached speed statistics from: {speed_cache}")
    else:
        empirical_speeds_stats = np.full((len(m_vals), 5), np.nan)
        connectivity_std = np.full(len(m_vals), np.nan)
        pos = jnp.linspace(0.0, L, N, endpoint=False).reshape((N, 1))
        input_params_eval = (eval_dt, input_std, L, 0.0, A_R, 0.0)
        v_warm = jnp.zeros(args.warmup_steps)
        v_eval = jnp.zeros(args.eval_steps)

        for im, m in enumerate(tqdm(m_vals, desc="Evaluating intrinsic speeds")):
            run_dir = base_data_dir / f"m-{im}"
            snap_files = sorted(glob.glob(str(run_dir / "snapshot_*.npz")))
            if not snap_files:
                print(f"[!] Warning: no snapshots in {run_dir}, skipping.")
                continue

            final_data = np.load(snap_files[-1], allow_pickle=True)
            if args.use_analytical_weights:
                J = gaussian(pos, pos.T, input_std, L)
                W = J * (1.0 - jnp.eye(N))
            else:
                J = jnp.array(final_data["W_ff"])
                W = jnp.array(final_data["W_rec"])

            J, idx_sort = sort_connectivity_matrix(J)
            W = W[idx_sort][:, idx_sort]

            init_dict = {
                "tau": tau, "tauv": tauv, "m": float(m), "k": k,
                "L": L, "std": input_std, "beta": 0.5,
            }
            J0_init = float(W.max()) / np.sqrt(2 * np.pi * input_std) * gain * 1e5
            U0, V0, _ = set_initial_conditions(N, J0_init, init_dict, init_dict)

            # Warm-up with FF + REC active
            res, _ = run_basic_simulation(
                U0, V0, J, W, 0.0, v_warm,
                (1.0 / tau, 1.0 / tauv, float(m), k, g, gain),
                input_params_eval, pos, args.warmup_steps,
            )

            # Autonomous recurrent test (FF off)
            _, history = run_basic_simulation(
                res[0] * args.u_boost, res[1], J * 0.0, W, 0.0, v_eval,
                (1.0 / tau, 1.0 / tauv, float(m), k * args.k_scale, g, gain),
                input_params_eval, pos, args.eval_steps,
            )

            unwrapped = np.array(rectify_positions(history[1], L))
            inst_speeds = np.abs(np.diff(unwrapped)[-args.tail_steps:]) / eval_dt

            empirical_speeds_stats[im, 0] = np.mean(inst_speeds)
            empirical_speeds_stats[im, 1] = np.std(inst_speeds)
            empirical_speeds_stats[im, 2] = np.quantile(inst_speeds, 0.25)
            empirical_speeds_stats[im, 3] = np.quantile(inst_speeds, 0.50)
            empirical_speeds_stats[im, 4] = np.quantile(inst_speeds, 0.75)
            connectivity_std[im] = np.mean(circ_std(np.array(W), L))

        if not args.use_analytical_weights:
            np.savez_compressed(
                speed_cache,
                m_vals=m_vals,
                empirical_speeds_stats=empirical_speeds_stats,
                connectivity_std=connectivity_std,
                params=sim_params,
            )
            print(f"[*] Saved speed statistics to: {speed_cache}")

    # 4. Plot Figure 5
    FONTSIZE = 15
    LINEWIDTH = 2
    MARKERSIZE = 10

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121)
    ax1.plot(time_hr, corr_FF, color="r", label="Feedforward connectivity", linewidth=LINEWIDTH)
    ax1.plot(time_hr, corr_Rec, color="b", label="Recurrent connectivity", linewidth=LINEWIDTH)
    ax1.set_xlabel("Time (hr)", fontsize=FONTSIZE)
    ax1.set_ylabel("corr", fontsize=FONTSIZE)
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax1.tick_params(axis="both", labelsize=FONTSIZE)
    ax1.legend(fontsize=FONTSIZE)

    ax2 = fig.add_subplot(122)
    end_idx = len(m_vals) - args.exclude_last if args.exclude_last > 0 else len(m_vals)
    median = empirical_speeds_stats[:end_idx, 3]
    q1 = empirical_speeds_stats[:end_idx, 2]
    q3 = empirical_speeds_stats[:end_idx, 4]
    iqr = q3 - q1
    asymmetric_error = np.vstack((median - q1 + 1.5 * iqr, q3 - median + 1.5 * iqr))

    m_vals_th = np.linspace(tau / tauv, m_vals[-1], 1000)
    ax2.plot(
        m_vals_th,
        theoretical_speed(m_vals_th, tau, tauv, input_std),
        "k--",
        label="Theory",
        linewidth=LINEWIDTH,
    )
    ax2.errorbar(
        m_vals[:end_idx],
        median,
        yerr=asymmetric_error,
        fmt="*",
        color="r",
        capsize=5,
        label="Simulation",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
    )
    ax2.set_ylabel("Speed (cm/s)", fontsize=FONTSIZE)
    ax2.set_xlabel("m", fontsize=FONTSIZE)
    ax2.set_xticks(np.arange(0.0, 0.6, 0.1))
    ax2.set_yticks(np.arange(0.0, 50.0, 10.0))
    ax2.tick_params(axis="both", labelsize=FONTSIZE)
    ax2.legend(fontsize=FONTSIZE)

    fig.tight_layout()
    fig_svg = base_fig_dir / "figure5-learning_ff_rec.svg"
    fig_png = base_fig_dir / "figure5-learning_ff_rec.png"
    fig.savefig(fig_svg, format="svg")
    fig.savefig(fig_png, format="png", dpi=300)
    print(f"[*] Saved Figure 5 to:\n    {fig_svg}\n    {fig_png}")
    plt.show()


if __name__ == "__main__":
    main()