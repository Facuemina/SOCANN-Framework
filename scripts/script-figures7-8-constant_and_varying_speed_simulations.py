#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined simulation script for Figures 7 and 8:
  1. Constant external speed-input sweep over adaptation strength (m) and
     velocity input (v) in a recurrent CANN (used in Figure 7 and top panels of Figure 8).
  2. Autonomous equilibrium speed calibration + smooth time-varying external
     speed tracking (used in bottom panel of Figure 8).

Saves all arrays and sim_params to ../figure7_8-path-integration-data/.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import (
    create_gaussian_connectivity_matrix,
    run_basic_simulation,
    set_initial_conditions,
)
from src.helpers import moving_average, rectify_positions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run constant-speed sweep and time-varying path integration for Figures 7 & 8."
    )
    # Shared network parameters
    parser.add_argument("--N", type=int, default=1000, help="Number of neurons")
    parser.add_argument("--L", type=float, default=100.0, help="Periodic domain length (cm)")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration time step (s)")
    parser.add_argument("--tau", type=float, default=0.01, help="U time constant (s)")
    parser.add_argument("--tauv", type=float, default=0.08, help="V time constant (s)")
    parser.add_argument("--k", type=float, default=0.01, help="Global inhibition")
    parser.add_argument("--g", type=float, default=2.0, help="Firing rate nonlinearity exponent")
    parser.add_argument("--gain", type=float, default=1.0, help="Firing rate gain")
    parser.add_argument("--W0", type=float, default=1.0, help="Recurrent Gaussian weight amplitude")
    parser.add_argument("--beta", type=float, default=0.5, help="Exponent for initial condition width")
    parser.add_argument("--input_std", type=float, default=5.0, help="Connectivity Gaussian std (cm)")

    # Part 1: Constant-speed sweep (Figure 7 & Figure 8 top panels)
    parser.add_argument("--sweep_steps", type=int, default=int(1e5), help="Euler steps per (m, v) run")
    parser.add_argument("--a_vel", type=float, default=5.0, help="Velocity input gain (I_speed = a_vel * v)")
    parser.add_argument(
        "--m_vals",
        type=float,
        nargs="+",
        default=[0.126, 0.13, 0.135, 0.14, 0.145],
        help="Adaptation strengths m to sweep over",
    )
    parser.add_argument("--v_min", type=float, default=-1.0, help="Minimum external speed v")
    parser.add_argument("--v_max", type=float, default=1.0, help="Maximum external speed v")
    parser.add_argument("--num_v", type=int, default=21, help="Number of external speed points")

    # Part 2: Time-varying speed tracking (Figure 8 bottom panel)
    parser.add_argument("--track_m_idx", type=int, default=1, help="Index in m_vals for tracking run (default: 1 -> 0.13)")
    parser.add_argument("--track_steps", type=int, default=59500, help="Euler steps per tracking phase (5950 * 10)")
    parser.add_argument("--track_av", type=float, default=1.0, help="Velocity modulation coupling a_vel for tracking")
    parser.add_argument("--win", type=int, default=750, help="Moving average window (1500 // 2)")
    parser.add_argument("--speed_variation_pctg", type=float, default=0.8, help="Peak-to-peak fractional speed swing")
    parser.add_argument("--seed", type=int, default=237, help="Random seed for external speed trajectory")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    repo_root = Path(__file__).resolve().parent.parent
    save_dir = repo_root.parent / "figure7_8-path-integration-data"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving Figures 7 & 8 data to: {save_dir}")

    # 2. Shared Topology & Sweep Grids
    m_vals = np.array(args.m_vals, dtype=float)
    v_vals = np.linspace(args.v_min, args.v_max, args.num_v)
    pos = jnp.linspace(0.0, args.L, args.N, endpoint=False).reshape((args.N, 1))
    angles = np.linspace(0.0, 2.0 * np.pi, args.N, endpoint=False)

    W_rec = create_gaussian_connectivity_matrix(args.N, args.input_std, args.L, W0=args.W0, periodic=True)
    W_ff = jnp.zeros((args.N, args.N))

    # 3. Part 1: Constant External Speed-Input Sweep over (m, v)
    U_all = np.zeros((len(m_vals), len(v_vals), args.N))
    V_all = np.zeros((len(m_vals), len(v_vals), args.N))
    Iu = np.zeros((len(m_vals), len(v_vals)))
    Iv = np.zeros((len(m_vals), len(v_vals)))
    I_vel = np.zeros((len(m_vals), len(v_vals)))
    speed_u = np.zeros((len(m_vals), len(v_vals)))
    circ_std_all = np.zeros((len(m_vals), len(v_vals)))

    total_runs = len(m_vals) * len(v_vals)
    with tqdm(total=total_runs, desc="Part 1: Constant-Speed Sweep") as pbar:
        for im, m in enumerate(m_vals):
            init_params = {"tau": args.tau, "tauv": args.tauv, "m": float(m), "k": args.k}
            init_input = {"beta": args.beta, "std": args.input_std, "L": args.L}
            U0, V0, _ = set_initial_conditions(args.N, args.W0, init_params, init_input)
            U0 = U0 * 100.0
            V0 = V0 * 100.0

            neural_params = (1.0 / args.tau, 1.0 / args.tauv, float(m), args.k, args.g, args.gain)
            input_params = (args.dt, args.input_std, args.L, 0.0, 0.0, args.a_vel)

            for iv, v in enumerate(v_vals):
                v_array = jnp.ones(args.sweep_steps) * float(v)
                (Uf, Vf, _), history = run_basic_simulation(
                    U0, V0, W_ff, W_rec, 0.0, v_array,
                    neural_params, input_params, pos, T=args.sweep_steps,
                )

                U_np = np.array(Uf)[:, 0]
                V_np = np.array(Vf)[:, 0]
                u_pos = np.array(rectify_positions(history[1], args.L))

                U_all[im, iv] = U_np
                V_all[im, iv] = V_np
                I_vel[im, iv] = args.a_vel * float(v)
                Iu[im, iv] = U_np[(np.argmax(U_np) + args.N // 2) % args.N]
                Iv[im, iv] = V_np[(np.argmax(V_np) + args.N // 2) % args.N]
                speed_u[im, iv] = float(np.mean(np.diff(u_pos)[-1000:]) / args.dt)

                u_pos_only = np.maximum(U_np - Iu[im, iv], 0.0)
                C = np.sum(np.cos(angles) * u_pos_only) / np.sum(u_pos_only)
                S = np.sum(np.sin(angles) * u_pos_only) / np.sum(u_pos_only)
                R = np.sqrt(C**2 + S**2)
                circ_std_all[im, iv] = np.sqrt(-2.0 * np.log(R)) / (2.0 * np.pi) * args.L

                pbar.update(1)

    # 4. Part 2: Time-Varying Speed Path Integration
    m_track = float(m_vals[args.track_m_idx])
    W_rec_track = W_rec * (1.0 - jnp.eye(args.N))
    U0_tr = jnp.sqrt(W_rec_track[:, 1:2]) * 100.0
    V0_tr = jnp.sqrt(W_rec_track[:, -10:-9]) * m_track * 100.0
    neural_params_tr = (1.0 / args.tau, 1.0 / args.tauv, m_track, args.k, args.g, args.gain)

    print(f"\n[*] Part 2a: Equilibrating autonomous bump for m={m_track:.3f} ({args.track_steps} steps)...")
    input_params_eq = (args.dt, args.input_std, args.L, 0.0, 30.0, 0.0)
    (U_eq, V_eq, _), hist_eq = run_basic_simulation(
        U0_tr, V0_tr, W_ff, W_rec_track, 0.0, jnp.zeros(args.track_steps),
        neural_params_tr, input_params_eq, pos, T=args.track_steps,
    )
    eq_pos = np.array(rectify_positions(hist_eq[1], args.L))
    v_eq = float(np.abs(np.mean(np.diff(eq_pos)[-1000:]) / args.dt))
    print(f"[*] Measured equilibrium bump speed: v_eq = {v_eq:.3f} cm/s")

    np.random.seed(args.seed)
    vel_std = v_eq / 1.5
    vels = np.zeros(args.track_steps + args.win - 1)
    vels[0] = v_eq
    stds = np.random.normal(size=(vels.shape[0],)) * vel_std
    for i in range(1, len(vels)):
        vels[i] = vels[i - 1] + 1.0 * (v_eq - vels[i - 1] + stds[i])

    min_vel = v_eq * (1.0 - args.speed_variation_pctg / 2.0)
    max_vel = v_eq * (1.0 + args.speed_variation_pctg / 2.0)
    vels = moving_average(vels, args.win)
    vels = (vels - vels.min()) / (vels.max() - vels.min()) * (max_vel - min_vel) + min_vel

    print(f"[*] Part 2b: Running time-varying speed tracking ({args.track_steps} steps)...")
    input_params_tr = (args.dt, args.input_std, args.L, v_eq, 30.0, args.track_av)
    (U_final, V_final, _), hist_tr = run_basic_simulation(
        U_eq, V_eq, W_ff, W_rec_track, 0.0, jnp.array(vels),
        neural_params_tr, input_params_tr, pos, T=args.track_steps,
    )
    bump_pos = np.array(rectify_positions(hist_tr[1], args.L))
    bump_speed = np.diff(bump_pos) / args.dt

    # 5. Save Parameter Dictionary and Combined Compressed Results
    sim_params = {
        "N": args.N,
        "L": args.L,
        "dt": args.dt,
        "tau": args.tau,
        "tauv": args.tauv,
        "k": args.k,
        "g": args.g,
        "gain": args.gain,
        "W0": args.W0,
        "beta": args.beta,
        "input_std": args.input_std,
        "sweep_steps": args.sweep_steps,
        "a_vel": args.a_vel,
        "m_vals": m_vals,
        "v_vals": v_vals,
        "track_m_idx": args.track_m_idx,
        "m_track": m_track,
        "track_steps": args.track_steps,
        "track_av": args.track_av,
        "v_eq": v_eq,
        "win": args.win,
        "speed_variation_pctg": args.speed_variation_pctg,
        "seed": args.seed,
    }

    np.save(save_dir / "sim_params.npy", sim_params)
    np.savez_compressed(
        save_dir / "figure7_8_results.npz",
        U_all=U_all,
        V_all=V_all,
        Iu=Iu,
        Iv=Iv,
        I_vel=I_vel,
        speed_u=speed_u,
        circ_std=circ_std_all,
        vels=vels,
        bump_pos=bump_pos,
        bump_speed=bump_speed,
        U_eq=np.array(U_eq)[:, 0],
        V_eq=np.array(V_eq)[:, 0],
        U_final=np.array(U_final)[:, 0],
        V_final=np.array(V_final)[:, 0],
        params=sim_params,
    )
    print(f"[*] Complete. Saved all results to {save_dir / 'figure7_8_results.npz'}")


if __name__ == "__main__":
    main()