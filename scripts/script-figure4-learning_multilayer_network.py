#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Figure 4: Self-organized multilayer feedforward architecture.
Initializes a cascade of feedforward CANN layers with random weights, drives
Layer 1 with a moving Gaussian input, saves compressed snapshots across chunks
to ../figure4-learning-multilayer-ff-data/, and runs a post-learning fixed-weight
evaluation over 100 uniformly sampled positions in [0, L) for Panels (c) and (d).
"""

import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import run_learning_multilayer_ff, run_multilayer_ff_evaluation
from src.theory import learned_U_std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multilayer FF CANN learning simulation for Figure 4."
    )
    parser.add_argument("--num_layers", type=int, default=4, help="Number of competitive layers")
    parser.add_argument("--N", type=int, default=512, help="Number of neurons per layer (N_in = N_c)")
    parser.add_argument("--L", type=float, default=70.0, help="Length of the periodic domain (cm)")
    parser.add_argument("--chunks", type=int, default=1500, help="Number of snapshots/chunks to simulate")
    parser.add_argument("--steps_per_chunk", type=int, default=500000, help="Euler steps per chunk (5e5)")
    parser.add_argument("--dt", type=float, default=0.005, help="Integration time step (s)")
    parser.add_argument("--tau", type=float, default=0.015, help="U time constant (s, 15 ms)")
    parser.add_argument("--tauv", type=float, default=0.6, help="V time constant (s, 600 ms)")
    parser.add_argument("--k", type=float, default=3, help="Global inhibition (cm^4 s^2 / syn)")
    parser.add_argument("--g", type=float, default=2.0, help="Firing rate nonlinearity exponent")
    parser.add_argument("--gain", type=float, default=1.0, help="Firing rate gain")
    parser.add_argument("--etaJ", type=float, default=5e-4, help="FF learning rate (Hz)")
    parser.add_argument("--alphaJ", type=float, default=0.5, help="Strength of FF synaptic decay")
    parser.add_argument("--A_R", type=float, default=30.0, help="Input amplitude (Hz)")
    parser.add_argument("--input_std", type=float, default=5.0, help="Input width sigma_R (cm)")
    parser.add_argument("--w_init_max", type=float, default=1., help="Upper bound for uniform random W_ff init")
    parser.add_argument("--eval_positions", type=int, default=100, help="Uniform positions in [0, L) for Panels (c-d)")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of beta_J values to sweep over",
    )
    parser.add_argument(
        "--v_vals",
        type=float,
        nargs="+",
        default=[20.0],
        help="List of stimulus speeds v (cm/s) to sweep over",
    )
    parser.add_argument(
        "--m_vals",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of adaptation strengths m to sweep over",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume from the latest saved snapshot in each run folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    repo_root = Path(__file__).resolve().parent.parent
    base_save_dir = repo_root.parent / "figure4-learning-multilayer-ff-data"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Base directory for Figure 4 data: {base_save_dir}")

    pos = jnp.linspace(0, args.L, args.N, endpoint=False).reshape((args.N, 1))
    av = 0.0

    # 2. Parameter Sweep Loop (matches original beta x v x m loop structure)
    for ibeta, beta in enumerate(args.betas):
        su = float(learned_U_std(beta, args.input_std))
        for iv, v in enumerate(args.v_vals):
            v_array = jnp.ones(args.steps_per_chunk) * float(v)
            for im, m in enumerate(args.m_vals):
                save_dir = (
                    base_save_dir
                    / f"N{args.N}_L{args.L:g}_beta{beta:g}_v{v:g}_m{m:g}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"\n[*] Running beta={beta}, v={v}, m={m} "
                    f"({args.num_layers} layers) | Saving to: {save_dir}"
                )

                # Dictionary containing all simulation parameters
                sim_params = {
                    "num_layers": args.num_layers,
                    "N": args.N,
                    "L": args.L,
                    "chunks": args.chunks,
                    "steps_per_chunk": args.steps_per_chunk,
                    "total_steps": args.chunks * args.steps_per_chunk,
                    "dt": args.dt,
                    "speed": float(v),
                    "v_eq": float(v),
                    "av": av,
                    "tau": args.tau,
                    "tauv": args.tauv,
                    "k": args.k,
                    "m": float(m),
                    "g": args.g,
                    "gain": args.gain,
                    "betaJ": float(beta),
                    "etaJ": args.etaJ,
                    "alphaJ": args.alphaJ,
                    "A_R": args.A_R,
                    "input_std": args.input_std,
                    "u_std": su,
                    "w_init_max": args.w_init_max,
                    "eval_positions": args.eval_positions,
                    "seed": args.seed,
                }
                np.save(save_dir / "sim_params.npy", sim_params)

                # Parameter tuples expected by src.engine
                neural_params = (
                    1.0 / args.tau,
                    1.0 / args.tauv,
                    float(m),
                    args.k,
                    args.g,
                    args.gain,
                )
                learning_params = (
                    args.dt,
                    float(beta),
                    args.etaJ,
                    args.input_std,
                    args.L,
                    float(v),
                    av,
                    args.A_R,
                    args.alphaJ,
                )
                input_params = (
                    args.dt,
                    args.input_std,
                    args.L,
                    float(v),
                    args.A_R,
                    av,
                )

                # 3. Initialization or Resume
                start_chunk = 0
                existing_snaps = sorted(glob.glob(str(save_dir / "snapshot_*.npz")))

                if args.resume and existing_snaps:
                    last_snap = existing_snaps[-1]
                    data = np.load(last_snap, allow_pickle=True)
                    U_tuple = tuple(jnp.array(u) for u in data["U"])
                    V_tuple = tuple(jnp.array(vv) for vv in data["V"])
                    W_tuple = tuple(jnp.array(w) for w in data["W_ff"])
                    x0 = float(data["x_stimulus"])
                    start_chunk = int(Path(last_snap).stem.split("_")[1])
                    print(f"[*] Resuming from {Path(last_snap).name} (chunk {start_chunk})")
                else:
                    key = jax.random.PRNGKey(args.seed)
                    keys = jax.random.split(key, 3 * args.num_layers + 1)
                    W_tuple = tuple(
                        jax.random.uniform(
                            keys[l], (args.N, args.N), minval=0.0, maxval=args.w_init_max
                        )
                        for l in range(args.num_layers)
                    )
                    U_tuple = tuple(
                        jax.random.uniform(
                            keys[args.num_layers + l], (args.N, 1), minval=0.0, maxval=1.0
                        )
                        for l in range(args.num_layers)
                    )
                    V_tuple = tuple(
                        jax.random.uniform(
                            keys[2 * args.num_layers + l], (args.N, 1), minval=0.0, maxval=1.0
                        )
                        for l in range(args.num_layers)
                    )
                    x0 = float(jax.random.uniform(keys[-1], (), minval=0.0, maxval=1.0))

                    # Save Snapshot 0000 (Initial State)
                    np.savez_compressed(
                        save_dir / "snapshot_0000.npz",
                        U=np.stack([np.array(u) for u in U_tuple], axis=0),
                        V=np.stack([np.array(vv) for vv in V_tuple], axis=0),
                        W_ff=np.stack([np.array(w) for w in W_tuple], axis=0),
                        x_stimulus=np.array(x0),
                        step=np.array(0),
                        time_hours=np.array(0.0),
                        params=sim_params,
                    )

                # 4. Learning Loop Across Chunks
                for chunk in tqdm(
                    range(start_chunk + 1, args.chunks + 1),
                    desc=f"beta={beta:g}-v={v:g}-m={m:g}",
                ):
                    U_tuple, V_tuple, W_tuple, x0 = run_learning_multilayer_ff(
                        U_tuple,
                        V_tuple,
                        W_tuple,
                        x0,
                        v_array,
                        neural_params,
                        learning_params,
                        pos,
                        T=args.steps_per_chunk,
                    )

                    total_step = chunk * args.steps_per_chunk
                    time_hours = total_step * args.dt / 3600.0

                    np.savez_compressed(
                        save_dir / f"snapshot_{chunk:04d}.npz",
                        U=np.stack([np.array(u) for u in U_tuple], axis=0),
                        V=np.stack([np.array(vv) for vv in V_tuple], axis=0),
                        W_ff=np.stack([np.array(w) for w in W_tuple], axis=0),
                        x_stimulus=np.array(x0),
                        step=np.array(total_step),
                        time_hours=np.array(time_hours),
                        params=sim_params,
                    )

                # 5. Post-Learning Fixed-Weight Evaluation for Panels (c) and (d)
                # Run one full lap of length L and uniformly sample 100 positions in [0, L)
                T_lap = int(np.round(args.L / (float(v) * args.dt)))
                v_eval = jnp.ones(T_lap) * float(v)
                _, (x_hist, I0_hist, fU_hist) = run_multilayer_ff_evaluation(
                    U_tuple,
                    V_tuple,
                    W_tuple,
                    x0,
                    v_eval,
                    neural_params,
                    input_params,
                    pos,
                    T=T_lap,
                )

                sample_idx = np.linspace(
                    0, T_lap - 1, args.eval_positions, endpoint=False, dtype=int
                )
                np.savez_compressed(
                    save_dir / "eval_results.npz",
                    x_eval=np.array(x_hist)[sample_idx],
                    I0_eval=np.array(I0_hist)[sample_idx, :, 0],
                    fU_eval=np.array(fU_hist)[sample_idx, :, :, 0],
                    params=sim_params,
                )
                print(
                    f"[*] Saved 100-position evaluation data to {save_dir / 'eval_results.npz'}"
                )

    print("\n[*] Figure 4 multilayer simulation complete.")


if __name__ == "__main__":
    main()