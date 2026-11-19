#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Figure 1: Self-organized feedforward connectivity parameter sweep over beta.
Initializes a Feedforward-only CANN with random weights, drives it with a moving 
Gaussian input across a sweep of beta_J values, and periodically saves compressed 
snapshots of the network's state and a parameter dictionary to ../figure1-learning-ff-data/.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import run_learning_ff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FF CANN learning parameter sweep over beta for Figure 1."
    )
    parser.add_argument("--N", type=int, default=512, help="Number of neurons (N_in = N_c)")
    parser.add_argument("--L", type=float, default=100.0, help="Length of the periodic domain (cm)")
    parser.add_argument("--total_steps", type=int, default=int(1e6), help="Total simulation steps (100 * 10^5)")
    parser.add_argument("--snapshots", type=int, default=50, help="Number of equally distributed snapshots to save")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration time step (s)")
    parser.add_argument("--speed", type=float, default=26.8, help="Constant speed of the Gaussian input (cm/s)")
    parser.add_argument("--tau", type=float, default=0.015, help="U time constant (s)")
    parser.add_argument("--tauv", type=float, default=0.6, help="V time constant (s)")
    parser.add_argument("--k", type=float, default=0.0512, help="Global inhibition (cm^4 s^2 / syn)")
    parser.add_argument("--m", type=float, default=0.2, help="Adaptation strength")
    parser.add_argument("--g", type=float, default=2.0, help="Firing rate nonlinearity exponent")
    parser.add_argument("--gain", type=float, default=1.0, help="Firing rate gain")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        help="List of beta_J values to sweep over",
    )
    parser.add_argument("--etaJ", type=float, default=5e-4, help="FF learning rate (Hz)")
    parser.add_argument("--alphaJ", type=float, default=1.0, help="Strength of FF synaptic decay")
    parser.add_argument("--A_R", type=float, default=30.0, help="Input amplitude (Hz)")
    parser.add_argument("--input_std", type=float, default=5.0, help="Input width sigma_R (cm)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight initialization")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    # Resolves from scripts/ -> repo_root -> parent directory
    repo_root = Path(__file__).resolve().parent.parent
    base_save_dir = repo_root.parent / "figure1-learning-ff-data"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Base directory for Figure 1 data: {base_save_dir}")

    # 2. Fixed Parameter Setup
    # Neural parameters: (1/tau, 1/tauv, m, k, g, gain)
    neural_params = (1.0 / args.tau, 1.0 / args.tauv, args.m, args.k, args.g, args.gain)
    pos = jnp.linspace(0, args.L, args.N, endpoint=False).reshape((args.N, 1))

    steps_per_chunk = args.total_steps // args.snapshots
    v_array = jnp.ones(steps_per_chunk) * args.speed

    # velocity-modulation parameters disabled for feedforward-only setup
    av = 0.0

    # 3. Parameter Sweep over beta_J
    for betaJ in args.betas:
        save_dir = base_save_dir / f"beta_{betaJ:.2f}_N{args.N}_steps{args.total_steps}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Running beta_J = {betaJ} | Saving to: {save_dir}")

        # Dictionary containing all simulation parameters
        sim_params = {
            "N": args.N,
            "L": args.L,
            "total_steps": args.total_steps,
            "snapshots": args.snapshots,
            "steps_per_chunk": steps_per_chunk,
            "dt": args.dt,
            "speed": args.speed,
            "v_eq": args.speed,
            "av": av,
            "tau": args.tau,
            "tauv": args.tauv,
            "k": args.k,
            "m": args.m,
            "g": args.g,
            "gain": args.gain,
            "betaJ": betaJ,
            "etaJ": args.etaJ,
            "alphaJ": args.alphaJ,
            "A_R": args.A_R,
            "input_std": args.input_std,
            "seed": args.seed,
        }

        # Save standalone parameter dictionary in the run directory
        np.save(save_dir / "sim_params.npy", sim_params)

        # Learning parameters tuple expected by src.engine.run_learning_ff_and_rec:
        # (dt, betaJ, etaJ, sR, L, v_eq, av, A_R, ain)
        learning_params = (
            sim_params["dt"],
            sim_params["betaJ"],
            sim_params["etaJ"],
            sim_params["input_std"],
            sim_params["L"],
            sim_params["v_eq"],
            sim_params["av"],
            sim_params["A_R"],
            sim_params["alphaJ"],
        )

        # Purely Random Initialization
        key = jax.random.PRNGKey(args.seed)
        key_ff, key_u, key_v = jax.random.split(key, 3)

        W_ff = jax.random.uniform(key_ff, (args.N, args.N), minval=0.0, maxval=0.01)

        U = jax.random.uniform(key_u, (args.N, 1), minval=0.0, maxval=0.1)
        V = jax.random.uniform(key_v, (args.N, 1), minval=0.0, maxval=0.1)
        x0 = 0.0

        # Save Snapshot 0 (Initial State + Parameter Dictionary)
        np.savez_compressed(
            save_dir / "snapshot_000.npz",
            U=np.array(U),
            V=np.array(V),
            W_ff=np.array(W_ff),
            x_stimulus=np.array(x0),
            params=sim_params,
        )

        # 4. Simulation and Snapshot Loop
        for chunk in tqdm(range(1, args.snapshots + 1), desc=f"beta={betaJ:.2f}"):
            U, V, W_ff, x0 = run_learning_ff(
                U, V, W_ff, x0, v_array,
                neural_params, learning_params, pos,
                T=steps_per_chunk
            )

            np.savez_compressed(
                save_dir / f"snapshot_{chunk:03d}.npz",
                U=np.array(U),
                V=np.array(V),
                W_ff=np.array(W_ff),
                x_stimulus=np.array(x0),
                params=sim_params,
            )

    print("\n[*] Figure 1 parameter sweep complete. All snapshots and parameters saved successfully.")


if __name__ == "__main__":
    main()

