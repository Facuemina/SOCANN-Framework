#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Figure 3: Feedforward bump asymmetry sweep over adaptation (m) and speed (v).
Simulates the fixed-weight Feedforward CANN using _cann_dynamics_ff across a grid of
(m, v) values, saving final network states and parameter dictionary to
../figure3-ff-asymmetry-data/.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from tqdm import tqdm

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import (
    _cann_dynamics_FF,
    create_gaussian_connectivity_matrix,
    gaussian,
)
from src.theory import learned_U_std, learned_weight_std


def run_ff_simulation(U0, V0, W_ff, x0, v, neural_params, input_params, pos, T):
    """Fixed-weight feedforward simulation loop using _cann_dynamics_ff."""

    @jit
    def step(i, carry):
        U, V, x_input = carry
        dt, sR, L, v_eq, R0, a_vel = input_params

        v0 = lax.dynamic_index_in_dim(v, i, axis=0, keepdims=False)
        x_input_next = jnp.mod(x_input + dt * v0, L)
        I0 = R0 * gaussian(pos, x_input_next, sR, L)

        dU, dV, _ = _cann_dynamics_FF(
            U, V, W_ff, I0, v0, neural_params, input_params
        )

        U_next = U + dt * dU
        V_next = V + dt * dV

        return U_next, V_next, x_input_next

    Uf, Vf, xf = lax.fori_loop(0, T, step, (U0, V0, x0))
    return Uf, Vf, xf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FF CANN asymmetry sweep over m and v for Figure 3."
    )
    parser.add_argument("--N", type=int, default=1000, help="Number of neurons (N_in = N_c)")
    parser.add_argument("--L", type=float, default=100.0, help="Length of periodic domain (cm)")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration time step (s)")
    parser.add_argument("--T", type=int, default=5000, help="Number of Euler steps per (m, v) run")
    parser.add_argument("--tau", type=float, default=0.010, help="U time constant (s, 10 ms)")
    parser.add_argument("--tauv", type=float, default=0.800, help="V time constant (s, 800 ms)")
    parser.add_argument("--k", type=float, default=0.1, help="Global inhibition (cm^4 s^2 / syn)")
    parser.add_argument("--g", type=float, default=2.0, help="Firing rate nonlinearity exponent")
    parser.add_argument("--gain", type=float, default=1.0, help="Firing rate gain")
    parser.add_argument("--A_R", type=float, default=30.0, help="Input amplitude (Hz)")
    parser.add_argument("--input_std", type=float, default=5.0, help="Input width sigma_R (cm)")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value defining learned W_ff width")
    parser.add_argument("--W0", type=float, default=1.0, help="Feedforward Gaussian connectivity amplitude")
    parser.add_argument(
        "--m_vals",
        type=float,
        nargs="+",
        default=list(np.linspace(0.05, 0.5, 20)),
        help="List of adaptation strengths m to sweep over",
    )
    parser.add_argument(
        "--v_vals",
        type=float,
        nargs="+",
        default=[2.0, 14.0, 26.0],
        help="List of constant stimulus velocities v (cm/s) to sweep over",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    repo_root = Path(__file__).resolve().parent.parent
    save_dir = repo_root.parent / "figure3-ff-asymmetry-data"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving Figure 3 simulation data to: {save_dir}")

    # 2. Fixed Topology & Grids
    m_vals = np.array(args.m_vals, dtype=float)
    v_vals = np.array(args.v_vals, dtype=float)

    w_std = float(learned_weight_std(args.beta, args.input_std))
    u_std = float(learned_U_std(args.beta, args.input_std))

    pos = jnp.linspace(0, args.L, args.N, endpoint=False).reshape((args.N, 1))
    W_ff = create_gaussian_connectivity_matrix(
        args.N, w_std, args.L, W0=args.W0, periodic=True
    )

    U_all = np.zeros((len(m_vals), len(v_vals), args.N, 1))
    V_all = np.zeros((len(m_vals), len(v_vals), args.N, 1))
    x_all = np.zeros((len(m_vals), len(v_vals)))

    U0 = jnp.zeros((args.N, 1))
    V0 = jnp.zeros((args.N, 1))
    x0 = 0.0

    # 3. Sweep over (m, v)
    total_runs = len(m_vals) * len(v_vals)
    with tqdm(total=total_runs, desc="Simulating FF Asymmetry") as pbar:
        for i, m in enumerate(m_vals):
            # Neural parameters: (1/tau, 1/tauv, m, k, g, gain)
            neural_params = (
                1.0 / args.tau,
                1.0 / args.tauv,
                float(m),
                args.k,
                args.g,
                args.gain,
            )
            for j, v in enumerate(v_vals):
                # Input parameters: (dt, sR, L, v_eq, R0, a_vel)
                input_params = (
                    args.dt,
                    args.input_std,
                    args.L,
                    float(v),
                    args.A_R,
                    0.0,
                )
                v_array = jnp.ones(args.T) * float(v)

                Uf, Vf, xf = run_ff_simulation(
                    U0, V0, W_ff, x0, v_array,
                    neural_params, input_params, pos, T=args.T
                )

                U_all[i, j] = np.array(Uf)
                V_all[i, j] = np.array(Vf)
                x_all[i, j] = float(xf)
                pbar.update(1)

    # 4. Parameter Dictionary & Save
    sim_params = {
        "N": args.N,
        "L": args.L,
        "dt": args.dt,
        "T": args.T,
        "tau": args.tau,
        "tauv": args.tauv,
        "k": args.k,
        "g": args.g,
        "gain": args.gain,
        "A_R": args.A_R,
        "input_std": args.input_std,
        "beta": args.beta,
        "W0": args.W0,
        "w_std": w_std,
        "u_std": u_std,
        "m_vals": m_vals,
        "v_vals": v_vals,
    }

    np.save(save_dir / "sim_params.npy", sim_params)
    np.savez_compressed(
        save_dir / "ff_asymmetry_results.npz",
        U_all=U_all,
        V_all=V_all,
        x_all=x_all,
        params=sim_params,
    )
    print(f"[*] Simulation sweep complete. Saved to {save_dir / 'ff_asymmetry_results.npz'}")


if __name__ == "__main__":
    main()