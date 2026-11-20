#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Figure 5: Simultaneous Hebbian learning of feedforward and recurrent
connectivity across an adaptation-strength (m) sweep.
Saves compressed snapshots and sim_params to ../figure5-learning-ff-rec-data/.
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
from src.helpers import moving_average, sort_connectivity_matrix
from src.theory import gaussian, learned_U_std, learned_weight_std, theoretical_speed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FF + REC CANN learning sweep over adaptation strength m for Figure 5."
    )
    parser.add_argument("--N", type=int, default=512, help="Number of neurons")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration time step (s)")
    parser.add_argument("--tau", type=float, default=0.015, help="U time constant (s)")
    parser.add_argument("--tauv", type=float, default=0.6, help="V time constant (s)")
    parser.add_argument("--k", type=float, default=0.1, help="Global inhibition")
    parser.add_argument("--g", type=float, default=2.0, help="Firing rate nonlinearity exponent")
    parser.add_argument("--gain", type=float, default=1.0, help="Firing rate gain")
    parser.add_argument("--input_std", type=float, default=5.0, help="Input width sigma_R (cm)")
    parser.add_argument("--A_R", type=float, default=30.0, help="Input amplitude (Hz)")
    parser.add_argument("--J0", type=float, default=0.001, help="Reference FF scale J0")
    parser.add_argument("--W0", type=float, default=0.0001, help="Reference REC scale W0")
    parser.add_argument("--etaJ", type=float, default=5e-4, help="FF learning rate")
    parser.add_argument("--etaW", type=float, default=5e-4, help="REC learning rate")
    parser.add_argument("--betaJ", type=float, default=1.0, help="FF synaptic decay exponent")
    parser.add_argument("--betaW", type=float, default=1.0, help="REC synaptic decay exponent")
    parser.add_argument("--alphaJ", type=float, default=1.0, help="FF synaptic decay strength (0=auto)")
    parser.add_argument("--alphaW", type=float, default=8.0, help="REC synaptic decay strength (0=auto)")
    parser.add_argument("--m_min", type=float, default=None, help="Min m (default: 2*tau/tauv)")
    parser.add_argument("--m_max", type=float, default=0.5, help="Max m")
    parser.add_argument("--num_m", type=int, default=10, help="Number of m values")
    parser.add_argument("--m_indices", type=int, nargs="+", default=None, help="Optional subset of m indices")
    parser.add_argument("--chunks", type=int, default=200, help="Number of simulation chunks")
    parser.add_argument("--steps_per_chunk", type=int, default=int(1.5e6), help="Euler steps per chunk")
    parser.add_argument("--varying_speed", action="store_true", help="Enable smooth time-varying speed")
    parser.add_argument("--av", type=float, default=None, help="Speed modulation (0.15 if varying else 0.0)")
    parser.add_argument("--from_stable", action="store_true", help="Initialize from Jeq, Weq")
    parser.add_argument("--save_every", type=int, default=1, help="Save snapshot every N chunks (0=init+final)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    repo_root = Path(__file__).resolve().parent.parent
    base_save_dir = repo_root.parent / "figure5-learning-ff-rec-data"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Base directory for Figure 5 data: {base_save_dir}")

    # 2. Derived Scales & Sweep Grid
    std_U = float(learned_U_std(args.betaJ, args.input_std))
    std_J = float(learned_weight_std(args.betaJ, args.input_std))
    std_W = float(learned_weight_std(args.betaW, std_U / np.sqrt(2.0)))
    L = float(np.round(np.sqrt(2.0) * 10.0 * std_U))

    alphaJ = args.alphaJ
    if alphaJ == 0.0:
        alphaJ = float(args.A_R / (args.J0 ** args.betaJ) / np.sqrt((2 * np.pi * std_J ** 2) ** (1 - args.betaJ) / args.betaJ))
    alphaW = args.alphaW
    if alphaW == 0.0:
        alphaW = float(args.A_R / (args.J0 ** args.betaW) / np.sqrt((2 * np.pi * std_W ** 2) ** (1 - args.betaW) / args.betaW))

    av = args.av if args.av is not None else (0.15 if args.varying_speed else 0.0)
    m_min = args.m_min if args.m_min is not None else (2.0 * args.tau / args.tauv)
    m_vals = np.linspace(m_min, args.m_max, args.num_m)
    v_vals = np.array([float(theoretical_speed(m_i, args.tau, args.tauv, args.input_std)) for m_i in m_vals])
    m_indices = args.m_indices if args.m_indices is not None else list(range(len(m_vals)))

    sim_params = {
        "N": args.N,
        "L": L,
        "dt": args.dt,
        "tau": args.tau,
        "tauv": args.tauv,
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
        "alphaJ": alphaJ,
        "alphaW": alphaW,
        "std_U": std_U,
        "std_J": std_J,
        "std_W": std_W,
        "m_vals": m_vals,
        "v_vals": v_vals,
        "chunks": args.chunks,
        "steps_per_chunk": args.steps_per_chunk,
        "total_steps": args.chunks * args.steps_per_chunk,
        "varying_speed": args.varying_speed,
        "av": av,
        "from_stable": args.from_stable,
        "save_every": args.save_every,
        "seed": args.seed,
    }
    np.save(base_save_dir / "sim_params.npy", sim_params)

    pos = jnp.linspace(0.0, L, args.N, endpoint=False).reshape((args.N, 1))
    Jeq = gaussian(pos, pos.T, std_J, L)
    Weq = gaussian(pos, pos.T, std_W, L) * (1.0 - jnp.eye(args.N))
    Jeq_np = np.array(Jeq)
    Weq_np = np.array(Weq)

    # 3. Sweep over Adaptation Strength m
    for i in m_indices:
        m = float(m_vals[i])
        v_eq = float(v_vals[i])
        save_dir = base_save_dir / f"m-{i}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Running m-{i} (m={m:.4f}, v={v_eq:.2f}) | Saving to: {save_dir}")

        rng = np.random.default_rng(args.seed + i)

        # Speed profile
        if args.varying_speed:
            win = 150
            v_eq = float(theoretical_speed(m, args.tau, args.tauv, std_U))
            vel_std = v_eq / 3.0
            n_raw = args.steps_per_chunk * args.chunks + win - 1
            vels = np.zeros(n_raw)
            vels[0] = v_eq
            noise = rng.normal(size=n_raw) * vel_std
            tau_speed = args.dt / 5.0
            for t in range(1, n_raw):
                vels[t] = vels[t - 1] + tau_speed * (v_eq - vels[t - 1] + noise[t])
            min_vel, max_vel = v_eq * 0.8, v_eq * 1.2
            vels = (vels - vels.min()) / (vels.max() - vels.min()) * (max_vel - min_vel) + min_vel
            vels_host = moving_average(vels, win).astype(np.float32)
            v_const = None
        else:
            vels_host = None
            v_const = jnp.ones(args.steps_per_chunk) * v_eq

        run_params = dict(sim_params)
        run_params.update({"m_idx": i, "m": m, "v_eq": v_eq, "speed": v_eq, "run_seed": args.seed + i})
        np.save(save_dir / "sim_params.npy", run_params)

        neural_params = (1.0 / args.tau, 1.0 / args.tauv, m, args.k, args.g, args.gain)
        learning_params = (
            args.dt, args.betaJ, args.betaW, args.etaJ, args.etaW,
            args.input_std, L, v_eq, av, args.A_R, alphaJ, alphaW,
        )

        # Initialize U, V, W_ff, W_rec
        init_scale = max(0.1, args.J0)
        if args.from_stable:
            W_ff = jnp.array(Jeq)
            W_rec = jnp.array(Weq)
            U = W_ff[:, :1]
            V = W_ff[:, -6:-5] * m
        else:
            U = jnp.array(rng.random((args.N, 1)))
            V = jnp.array(rng.random((args.N, 1)))
            W_ff = jnp.array(np.maximum(rng.normal(size=(args.N, args.N)), 0.0) * init_scale)
            W_rec = jnp.array(np.maximum(rng.normal(size=(args.N, args.N)), 0.0) * 0.01 * init_scale)
            W_rec = W_rec * (1.0 - jnp.eye(args.N))
        x0 = 0.0

        # Stabilization warm-up if starting from equilibrium
        if args.from_stable:
            stab_params = (
                args.dt, args.betaJ, args.betaW, 0.0, 0.0,
                args.input_std, L, v_eq, 0.0, args.A_R, alphaJ, alphaW,
            )
            v_stab = jnp.ones(5000) * (float(vels_host[0]) if vels_host is not None else v_eq)
            U, V, W_ff, W_rec, x0 = run_learning_ff_and_rec(
                U, V, W_ff, W_rec, x0, v_stab, neural_params, stab_params, pos, T=5000
            )

        # Track correlations across chunks
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

        # 4. Simulation and Snapshot Loop
        chunks_completed = 0
        for chunk in tqdm(range(1, args.chunks + 1), desc=f"m-{i} (m={m:.3f})"):
            v_chunk = (
                jnp.array(vels_host[(chunk - 1) * args.steps_per_chunk : chunk * args.steps_per_chunk])
                if vels_host is not None
                else v_const
            )
            U, V, W_ff, W_rec, x0 = run_learning_ff_and_rec(
                U, V, W_ff, W_rec, x0, v_chunk, neural_params, learning_params, pos, T=args.steps_per_chunk
            )

            W_ff_s, idx_s = sort_connectivity_matrix(W_ff)
            cff = float(np.corrcoef(np.array(W_ff_s).ravel(), Jeq_np.ravel())[0, 1])
            crec = float(np.corrcoef(np.array(W_rec[idx_s][:, idx_s]).ravel(), Weq_np.ravel())[0, 1])
            corr_ff[chunk] = cff
            corr_rec[chunk] = crec
            chunks_completed = chunk

            if (args.save_every > 0 and chunk % args.save_every == 0) or (chunk == args.chunks):
                np.savez_compressed(
                    save_dir / f"snapshot_{chunk:03d}.npz",
                    U=np.array(U),
                    V=np.array(V),
                    W_ff=np.array(W_ff),
                    W_rec=np.array(W_rec),
                    x_stimulus=np.array(x0),
                    params=run_params,
                )

            if args.from_stable and cff < 0.2 and crec < 0.2:
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

    print("\n[*] Figure 5 m-sweep complete. All snapshots and parameters saved successfully.")


if __name__ == "__main__":
    main()