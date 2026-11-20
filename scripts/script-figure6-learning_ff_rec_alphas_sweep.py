#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Figure 6: 2D parameter sweep over feedforward (alpha_J) and recurrent
(alpha_W) synaptic decay strengths during simultaneous FF + REC Hebbian learning.
Saves compressed snapshots and sim_params to ../figure6-learning-ff-rec-alphas-data/.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import run_learning_ff_and_rec
from src.helpers import sort_connectivity_matrix
from src.theory import gaussian, learned_U_std, learned_weight_std, theoretical_speed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 2D (alpha_J, alpha_W) FF+REC learning sweep for Figure 6."
    )
    parser.add_argument("--N", type=int, default=512, help="Number of neurons")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration time step (s)")
    parser.add_argument("--tau", type=float, default=0.015, help="U time constant (s)")
    parser.add_argument("--tauv", type=float, default=0.6, help="V time constant (s)")
    parser.add_argument("--m", type=float, default=0.2, help="Adaptation strength")
    parser.add_argument("--k", type=float, default=0.1, help="Global inhibition")
    parser.add_argument("--g", type=float, default=2.0, help="Firing rate nonlinearity exponent")
    parser.add_argument("--gain", type=float, default=1.0, help="Firing rate gain")
    parser.add_argument("--input_std", type=float, default=5.0, help="Input width sigma_R (cm)")
    parser.add_argument("--A_R", type=float, default=30.0, help="Input amplitude (Hz)")
    parser.add_argument("--J0", type=float, default=0.001, help="Reference FF scale J0")
    parser.add_argument("--W0", type=float, default=0.0001, help="Reference REC scale W0")
    parser.add_argument("--etaJ", type=float, default=1e-6, help="FF learning rate")
    parser.add_argument("--etaW", type=float, default=1e-6, help="REC learning rate")
    parser.add_argument("--betaJ", type=float, default=1.0, help="FF synaptic decay exponent")
    parser.add_argument("--betaW", type=float, default=1.0, help="REC synaptic decay exponent")
    parser.add_argument("--alphaJ_log_min", type=float, default=-1.5, help="log10(min alpha_J)")
    parser.add_argument("--alphaJ_log_max", type=float, default=2.1, help="log10(max alpha_J)")
    parser.add_argument("--num_alphaJ", type=int, default=7, help="Number of alpha_J points")
    parser.add_argument("--alphaW_log_min", type=float, default=-1.5, help="log10(min alpha_W)")
    parser.add_argument("--alphaW_log_max", type=float, default=4.1, help="log10(max alpha_W)")
    parser.add_argument("--num_alphaW", type=int, default=10, help="Number of alpha_W points")
    parser.add_argument("--alphaJ_indices", type=int, nargs="+", default=None, help="Optional subset of alpha_J indices")
    parser.add_argument("--alphaW_indices", type=int, nargs="+", default=None, help="Optional subset of alpha_W indices")
    parser.add_argument("--init_mode", type=str, choices=("stable", "random"), default="stable", help="Weight init")
    parser.add_argument("--chunks", type=int, default=100, help="Number of simulation chunks")
    parser.add_argument("--steps_per_chunk", type=int, default=int(1.5e6), help="Euler steps per chunk")
    parser.add_argument("--save_every", type=int, default=0, help="Save snapshot every N chunks (0=init+final)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip pairs with existing final snapshot")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    repo_root = Path(__file__).resolve().parent.parent
    base_save_dir = repo_root.parent / "figure6-learning-ff-rec-alphas-data"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Base directory for Figure 6 data: {base_save_dir}")

    # 2. Derived Scales & 2D Sweep Grid
    std_U = float(learned_U_std(args.betaJ, args.input_std))
    std_J = float(learned_weight_std(args.betaJ, args.input_std))
    std_W = float(learned_weight_std(args.betaW, std_U / np.sqrt(2.0)))
    L = float(np.round(np.sqrt(2.0) * 10.0 * std_U))
    v_eq = float(theoretical_speed(args.m, args.tau, args.tauv, args.input_std))
    from_stable = args.init_mode == "stable"

    alphaJ_vals = np.logspace(args.alphaJ_log_min, args.alphaJ_log_max, args.num_alphaJ)
    alphaW_vals = np.logspace(args.alphaW_log_min, args.alphaW_log_max, args.num_alphaW)
    alphaJ_indices = args.alphaJ_indices if args.alphaJ_indices is not None else list(range(len(alphaJ_vals)))
    alphaW_indices = args.alphaW_indices if args.alphaW_indices is not None else list(range(len(alphaW_vals)))

    sim_params = {
        "N": args.N,
        "L": L,
        "dt": args.dt,
        "tau": args.tau,
        "tauv": args.tauv,
        "m": args.m,
        "k": args.k,
        "g": args.g,
        "gain": args.gain,
        "input_std": args.input_std,
        "A_R": args.A_R,
        "J0": args.J0,
        "W0": args.W0,
        "etaJ": args.etaJ,
        "etaW": args.etaW,
        "betaJ": args.betaJ,
        "betaW": args.betaW,
        "std_U": std_U,
        "std_J": std_J,
        "std_W": std_W,
        "v_eq": v_eq,
        "speed": v_eq,
        "av": 0.0,
        "alphaJ_vals": alphaJ_vals,
        "alphaW_vals": alphaW_vals,
        "chunks": args.chunks,
        "steps_per_chunk": args.steps_per_chunk,
        "total_steps": args.chunks * args.steps_per_chunk,
        "init_mode": args.init_mode,
        "from_stable": from_stable,
        "save_every": args.save_every,
        "seed": args.seed,
    }
    np.save(base_save_dir / "sim_params.npy", sim_params)

    pos = jnp.linspace(0.0, L, args.N, endpoint=False).reshape((args.N, 1))
    Jeq = gaussian(pos, pos.T, std_J, L)
    Weq = gaussian(pos, pos.T, std_W, L) * (1.0 - jnp.eye(args.N))
    Jeq_np = np.array(Jeq)
    Weq_np = np.array(Weq)

    neural_params = (1.0 / args.tau, 1.0 / args.tauv, args.m, args.k, args.g, args.gain)
    v_array = jnp.ones(args.steps_per_chunk) * v_eq

    # 3. 2D Sweep over (alpha_J, alpha_W)
    for i in alphaJ_indices:
        alphaJ = float(alphaJ_vals[i])
        for j in alphaW_indices:
            alphaW = float(alphaW_vals[j])
            save_dir = base_save_dir / f"alphaJ-{i}_alphaW-{j}"
            save_dir.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and (save_dir / "learning_history.npz").exists():
                print(f"[*] Skipping existing alphaJ-{i}_alphaW-{j}")
                continue

            print(f"\n[*] Running alphaJ[{i}]={alphaJ:.2e}, alphaW[{j}]={alphaW:.2e}")
            rng = np.random.default_rng(args.seed + i * len(alphaW_vals) + j)

            run_params = dict(sim_params)
            run_params.update({"alphaJ_idx": i, "alphaW_idx": j, "alphaJ": alphaJ, "alphaW": alphaW})
            np.save(save_dir / "sim_params.npy", run_params)

            learning_params = (
                args.dt, args.betaJ, args.betaW, args.etaJ, args.etaW,
                args.input_std, L, v_eq, 0.0, args.A_R, alphaJ, alphaW,
            )

            init_scale = max(0.1, args.J0)
            if from_stable:
                W_ff = jnp.array(Jeq)
                W_rec = jnp.array(Weq)
                U = W_ff[:, :1]
                V = W_ff[:, -6:-5] * args.m
            else:
                U = jnp.array(rng.random((args.N, 1)))
                V = jnp.array(rng.random((args.N, 1)))
                W_ff = jnp.array(np.maximum(rng.normal(size=(args.N, args.N)), 0.0) * init_scale)
                W_rec = jnp.array(np.maximum(rng.normal(size=(args.N, args.N)), 0.0) * 0.01 * init_scale)
                W_rec = W_rec * (1.0 - jnp.eye(args.N))
            x0 = 0.0

            if from_stable:
                stab_params = (
                    args.dt, args.betaJ, args.betaW, 0.0, 0.0,
                    args.input_std, L, v_eq, 0.0, args.A_R, alphaJ, alphaW,
                )
                U, V, W_ff, W_rec, x0 = run_learning_ff_and_rec(
                    U, V, W_ff, W_rec, x0, jnp.ones(5000) * v_eq, neural_params, stab_params, pos, T=5000
                )

            corr_ff = np.full(args.chunks + 1, np.nan)
            corr_rec = np.full(args.chunks + 1, np.nan)
            W_ff_s, idx_s = sort_connectivity_matrix(W_ff)
            corr_ff[0] = np.corrcoef(np.array(W_ff_s).ravel(), Jeq_np.ravel())[0, 1]
            corr_rec[0] = np.corrcoef(np.array(W_rec[idx_s][:, idx_s]).ravel(), Weq_np.ravel())[0, 1]

            np.savez_compressed(
                save_dir / "snapshot_000.npz",
                U=np.array(U),
                V=np.array(V),
                W_ff=np.array(W_ff),
                W_rec=np.array(W_rec),
                x_stimulus=np.array(x0),
                params=run_params,
            )

            chunks_completed = 0
            for chunk in tqdm(range(1, args.chunks + 1), desc=f"aJ-{i}_aW-{j}"):
                U, V, W_ff, W_rec, x0 = run_learning_ff_and_rec(
                    U, V, W_ff, W_rec, x0, v_array, neural_params, learning_params, pos, T=args.steps_per_chunk
                )

                W_ff_s, idx_s = sort_connectivity_matrix(W_ff)
                cff = float(np.corrcoef(np.array(W_ff_s).ravel(), Jeq_np.ravel())[0, 1])
                crec = float(np.corrcoef(np.array(W_rec[idx_s][:, idx_s]).ravel(), Weq_np.ravel())[0, 1])
                corr_ff[chunk] = cff
                corr_rec[chunk] = crec
                chunks_completed = chunk

                early_stop = from_stable and (cff < 0.2 and crec < 0.2)
                if (args.save_every > 0 and chunk % args.save_every == 0) or (chunk == args.chunks) or early_stop:
                    np.savez_compressed(
                        save_dir / f"snapshot_{chunk:03d}.npz",
                        U=np.array(U),
                        V=np.array(V),
                        W_ff=np.array(W_ff),
                        W_rec=np.array(W_rec),
                        x_stimulus=np.array(x0),
                        params=run_params,
                    )

                if early_stop:
                    break

            time_s = np.arange(chunks_completed + 1) * args.steps_per_chunk * args.dt
            np.savez_compressed(
                save_dir / "learning_history.npz",
                chunks=np.arange(chunks_completed + 1),
                time_s=time_s,
                corr_ff=corr_ff[: chunks_completed + 1],
                corr_rec=corr_rec[: chunks_completed + 1],
                params=run_params,
            )
            jax.clear_caches()

    print("\n[*] Figure 6 (alpha_J, alpha_W) sweep complete. All snapshots saved successfully.")


if __name__ == "__main__":
    main()