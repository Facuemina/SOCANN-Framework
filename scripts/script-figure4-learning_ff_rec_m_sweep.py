#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 simulation script: Simultaneous Hebbian learning of feedforward and
recurrent connectivity in a 1D CANN across an adaptation-strength (m) sweep.

Saves top-level sweep parameters and per-m subfolders (m-0, m-1, ...) with
compressed weight snapshots, learning correlation histories, and final states
outside the repository at:
    <repo_root>/../figure5-learning-ff-rec-data/
"""

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.engine import gaussian, run_learning_ff_and_rec
from src.helpers import moving_average, sort_connectivity_matrix
from src.theory import learned_U_std, learned_weight_std, theoretical_speed


@partial(jax.jit, static_argnames=("g", "T", "T0"))
def _run_learning_ff_and_rec_jit(
    U0,
    V0,
    W_ff0,
    W_rec0,
    x0,
    v,
    inv_tau,
    inv_tauv,
    m,
    k,
    g,
    gain,
    learning_params,
    pos,
    T,
    T0=0,
):
    """Module-level JIT wrapper to prevent per-chunk recompilation."""
    neural_params = (inv_tau, inv_tauv, m, k, g, gain)
    return run_learning_ff_and_rec(
        U0,
        V0,
        W_ff0,
        W_rec0,
        x0,
        v,
        neural_params,
        learning_params,
        pos,
        T,
        T0=T0,
    )


def _safe_corrcoef(a, b):
    """Compute Pearson correlation between flattened arrays, returning NaN if constant."""
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _generate_varying_speed_profile(
    total_steps,
    dt,
    m,
    tau,
    tauv,
    std_u,
    rng,
    win=150,
    speed_variation_pctg=0.4,
):
    """Generate smooth OU-filtered varying speed trajectory (host NumPy array)."""
    vel_eq = float(theoretical_speed(m, tau, tauv, std_u))
    vel_std = vel_eq / 3.0
    n_raw = total_steps + win - 1
    vels = np.empty(n_raw, dtype=np.float64)
    vels[0] = vel_eq
    noise = rng.normal(size=n_raw) * vel_std
    tau_speed = dt / 5.0

    for t in range(1, n_raw):
        vels[t] = vels[t - 1] + tau_speed * (vel_eq - vels[t - 1] + noise[t])

    min_vel = vel_eq * (1.0 - speed_variation_pctg / 2.0)
    max_vel = vel_eq * (1.0 + speed_variation_pctg / 2.0)
    v_min, v_max = vels.min(), vels.max()
    if v_max > v_min:
        vels = (vels - v_min) / (v_max - v_min) * (max_vel - min_vel) + min_vel
    else:
        vels.fill(vel_eq)

    vels_smooth = np.asarray(moving_average(vels, win), dtype=np.float32)
    return vels_smooth, vel_eq, vel_std, win


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep adaptation strength m while learning FF and REC weights."
    )

    # Network & neural dynamics (launcher + fun_learning defaults)
    parser.add_argument("--n", type=int, default=512, help="Number of neurons.")
    parser.add_argument("--dt", type=float, default=0.01, help="Euler time step (s).")
    parser.add_argument("--tau", type=float, default=0.015, help="Membrane time constant (s).")
    parser.add_argument("--tauv", type=float, default=0.6, help="Adaptation time constant (s).")
    parser.add_argument("--k", type=float, default=0.1, help="Divisive inhibition constant.")
    parser.add_argument("--g", type=float, default=2.0, help="Activation exponent / type.")
    parser.add_argument("--gain", type=float, default=1.0, help="Activation gain.")

    # Input & plasticity parameters
    parser.add_argument("--input_std", type=float, default=5.0, help="Input bump std sigma_R.")
    parser.add_argument("--R0", type=float, default=30.0, help="Input amplitude A_R.")
    parser.add_argument("--J0", type=float, default=0.001, help="Reference FF scale J0.")
    parser.add_argument("--W0", type=float, default=0.0001, help="Reference REC scale W0.")
    parser.add_argument("--etaJ", type=float, default=5e-4, help="FF learning rate.")
    parser.add_argument("--etaW", type=float, default=5e-4, help="REC learning rate.")
    parser.add_argument("--betaJ", type=float, default=1.0, help="FF weight decay exponent.")
    parser.add_argument("--betaW", type=float, default=1.0, help="REC weight decay exponent.")
    parser.add_argument("--alphaJ", type=float, default=1.0, help="FF decay coefficient (0 = auto).")
    parser.add_argument("--alphaW", type=float, default=8.0, help="REC decay coefficient (0 = auto).")

    # Adaptation sweep parameters
    parser.add_argument(
        "--m_min",
        type=float,
        default=None,
        help="Minimum adaptation m (default: 2 * tau / tauv).",
    )
    parser.add_argument("--m_max", type=float, default=0.5, help="Maximum adaptation m.")
    parser.add_argument("--num_m", type=int, default=10, help="Number of m values in sweep.")
    parser.add_argument(
        "--m_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of m indices (0-based) to run.",
    )

    # Simulation duration & speed mode
    parser.add_argument("--chunks", type=int, default=200, help="Number of simulation chunks.")
    parser.add_argument(
        "--steps_per_chunk",
        type=int,
        default=1_500_000,
        help="Euler steps per chunk (legacy fun_learning default: 1.5e6).",
    )
    parser.add_argument(
        "--varying_speed",
        action="store_true",
        help="Use smooth time-varying speed profile instead of constant speed.",
    )
    parser.add_argument(
        "--av",
        type=float,
        default=None,
        help="Speed modulation parameter (default: 0.15 if --varying_speed else 0.0).",
    )
    parser.add_argument(
        "--from_stable",
        action="store_true",
        help="Initialize weights from analytical equilibrium (Jeq, Weq).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")

    # Output & snapshot controls
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT.parent / "figure5-learning-ff-rec-data",
        help="Directory outside repository where data will be saved.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save weight snapshot every N chunks (0 = only initial and final).",
    )
    parser.add_argument(
        "--snapshot_m_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of m indices for which intermediate snapshots are saved "
        "(default: all simulated m indices).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    n = args.n
    dt = args.dt
    tau = args.tau
    tauv = args.tauv
    k = args.k
    g = args.g
    gain = args.gain
    input_std = args.input_std
    R0 = args.R0
    J0 = args.J0
    W0 = args.W0
    etaJ = args.etaJ
    etaW = args.etaW
    betaJ = args.betaJ
    betaW = args.betaW
    chunks = args.chunks
    steps = args.steps_per_chunk
    av = args.av if args.av is not None else (0.15 if args.varying_speed else 0.0)

    # Derived spatial scales (matching legacy fun_learning_FeedForward_and_recurrent_CANN.py)
    std_U = float(learned_U_std(betaJ, input_std))
    std_J = float(learned_weight_std(betaJ, input_std))
    std_W = float(learned_weight_std(betaW, std_U / np.sqrt(2.0)))
    L = float(np.round(np.sqrt(2.0) * 10.0 * std_U))

    alphaJ = args.alphaJ
    if alphaJ == 0.0:
        alphaJ = float(
            R0 / (J0**betaJ) / np.sqrt((2.0 * np.pi * std_J**2) ** (1.0 - betaJ) / betaJ)
        )

    alphaW = args.alphaW
    if alphaW == 0.0:
        alphaW = float(
            R0 / (J0**betaW) / np.sqrt((2.0 * np.pi * std_W**2) ** (1.0 - betaW) / betaW)
        )

    # Adaptation sweep grid and corresponding theoretical speeds
    m_min = args.m_min if args.m_min is not None else (2.0 * tau / tauv)
    m_vals = np.linspace(m_min, args.m_max, args.num_m, dtype=np.float64)
    v_vals = np.array(
        [float(theoretical_speed(m_i, tau, tauv, input_std)) for m_i in m_vals],
        dtype=np.float64,
    )

    m_indices = args.m_indices if args.m_indices is not None else list(range(len(m_vals)))
    snapshot_m_set = (
        set(args.snapshot_m_indices) if args.snapshot_m_indices is not None else set(m_indices)
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top-level parameter dictionary
    sim_params = {
        "n": n,
        "dt": dt,
        "tau": tau,
        "tauv": tauv,
        "k": k,
        "g": g,
        "gain": gain,
        "input_std": input_std,
        "R0": R0,
        "J0": J0,
        "W0": W0,
        "etaJ": etaJ,
        "etaW": etaW,
        "betaJ": betaJ,
        "betaW": betaW,
        "alphaJ": alphaJ,
        "alphaW": alphaW,
        "std_U": std_U,
        "std_J": std_J,
        "std_W": std_W,
        "L": L,
        "m_min": m_min,
        "m_max": args.m_max,
        "num_m": args.num_m,
        "m_vals": m_vals,
        "v_vals": v_vals,
        "chunks": chunks,
        "steps_per_chunk": steps,
        "total_steps": chunks * steps,
        "varying_speed": args.varying_speed,
        "av": av,
        "from_stable": args.from_stable,
        "seed": args.seed,
        "save_every": args.save_every,
    }
    np.save(output_dir / "sim_params.npy", sim_params)

    # Spatial grid and analytical equilibrium templates
    pos = jnp.linspace(0.0, L, n, endpoint=False).reshape((n, 1))
    Jeq = gaussian(pos, pos.T, std_J, L)
    Weq = gaussian(pos, pos.T, std_W, L) * (1.0 - jnp.eye(n))
    Jeq_np = np.asarray(Jeq, dtype=np.float32)
    Weq_np = np.asarray(Weq, dtype=np.float32)

    inv_tau = 1.0 / tau
    inv_tauv = 1.0 / tauv
    eye_mask = 1.0 - jnp.eye(n)

    print(f"Saving Figure 5 FF+REC sweep outputs to: {output_dir}")
    print(f"L = {L:.2f}, std_U = {std_U:.3f}, std_J = {std_J:.3f}, std_W = {std_W:.3f}")
    print(f"Running m indices: {m_indices} | m_vals = {np.round(m_vals[m_indices], 4)}")

    for i in m_indices:
        m = float(m_vals[i])
        v_eq = float(v_vals[i])
        run_dir = output_dir / f"m-{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(args.seed + i)
        save_intermediate = (i in snapshot_m_set) and (args.save_every > 0)

        # Build speed profile
        vel_std = 0.0
        avg_win = 0
        if args.varying_speed:
            vels_host, v_eq, vel_std, avg_win = _generate_varying_speed_profile(
                total_steps=steps * chunks,
                dt=dt,
                m=m,
                tau=tau,
                tauv=tauv,
                std_u=std_U,
                rng=rng,
            )
            vels_const = None
        else:
            vels_host = None
            vels_const = jnp.full((steps,), v_eq, dtype=jnp.float32)

        run_params = dict(sim_params)
        run_params.update(
            {
                "m_idx": i,
                "m": m,
                "v": v_eq,
                "vel_std": vel_std,
                "avg_win": avg_win,
                "run_seed": args.seed + i,
            }
        )
        np.save(run_dir / "sim_params.npy", run_params)

        # Initialize state and weights
        init_scale = max(0.1, J0)
        if args.from_stable:
            J = jnp.array(Jeq, dtype=jnp.float32)
            W = jnp.array(Weq, dtype=jnp.float32)
            U = J[:, :1]
            V = J[:, -6:-5] * m
        else:
            U = jnp.asarray(rng.random((n, 1)), dtype=jnp.float32)
            V = jnp.asarray(rng.random((n, 1)), dtype=jnp.float32)
            J = jnp.asarray(
                np.maximum(rng.standard_normal((n, n)), 0.0) * init_scale,
                dtype=jnp.float32,
            )
            W = (
                jnp.asarray(
                    np.maximum(rng.standard_normal((n, n)), 0.0) * 0.01 * init_scale,
                    dtype=jnp.float32,
                )
                * eye_mask
            )

        x = jnp.asarray(0.0, dtype=jnp.float32)

        # Optional stabilization warm-up if starting from equilibrium
        if args.from_stable:
            stab_steps = 5000
            stab_learning_params = (
                dt,
                betaJ,
                betaW,
                0.0,
                0.0,
                input_std,
                L,
                v_eq,
                0.0,
                R0,
                alphaJ,
                alphaW,
            )
            v_stab = jnp.full(
                (stab_steps,),
                float(vels_host[0]) if vels_host is not None else v_eq,
                dtype=jnp.float32,
            )
            U, V, J, W, x = _run_learning_ff_and_rec_jit(
                U, V, J, W, x, v_stab, inv_tau, inv_tauv, m, k, g, gain, stab_learning_params, pos, stab_steps
            )

        learning_params = (
            dt,
            betaJ,
            betaW,
            etaJ,
            etaW,
            input_std,
            L,
            v_eq,
            av,
            R0,
            alphaJ,
            alphaW,
        )

        # Track correlation & weight maxima across chunks (including chunk 0)
        chunk_ids = np.arange(chunks + 1, dtype=np.int32)
        time_s = chunk_ids.astype(np.float64) * steps * dt
        corr_ff_hist = np.full(chunks + 1, np.nan, dtype=np.float64)
        corr_rec_hist = np.full(chunks + 1, np.nan, dtype=np.float64)
        j_max_hist = np.full(chunks + 1, np.nan, dtype=np.float64)
        w_max_hist = np.full(chunks + 1, np.nan, dtype=np.float64)

        J_np = np.asarray(J, dtype=np.float32)
        W_np = np.asarray(W, dtype=np.float32)
        sorted_J0, idx_sort0 = sort_connectivity_matrix(J_np)
        corr_ff_hist[0] = _safe_corrcoef(sorted_J0, Jeq_np)
        corr_rec_hist[0] = _safe_corrcoef(W_np[np.ix_(idx_sort0, idx_sort0)], Weq_np)
        j_max_hist[0] = float(J_np.max())
        w_max_hist[0] = float(W_np.max())

        # Save initial snapshot (chunk 0)
        np.savez_compressed(
            run_dir / "snapshot_0000.npz",
            J_ff=J_np,
            W_rec=W_np,
            chunk=0,
            time_s=0.0,
            corr_ff=corr_ff_hist[0],
            corr_rec=corr_rec_hist[0],
            params=run_params,
        )

        t_start = time.perf_counter()
        chunks_completed = 0
        idx_sort = idx_sort0

        pbar = tqdm(range(1, chunks + 1), desc=f"m-{i} (m={m:.3f}, v={v_eq:.1f})")
        for c in pbar:
            if vels_host is not None:
                v_chunk = jnp.asarray(vels_host[(c - 1) * steps : c * steps], dtype=jnp.float32)
            else:
                v_chunk = vels_const

            U, V, J, W, x = _run_learning_ff_and_rec_jit(
                U,
                V,
                J,
                W,
                x,
                v_chunk,
                inv_tau,
                inv_tauv,
                m,
                k,
                g,
                gain,
                learning_params,
                pos,
                steps,
            )

            J_np = np.asarray(J, dtype=np.float32)
            W_np = np.asarray(W, dtype=np.float32)
            sorted_J, idx_sort = sort_connectivity_matrix(J_np)
            sorted_W = W_np[np.ix_(idx_sort, idx_sort)]

            cff = _safe_corrcoef(sorted_J, Jeq_np)
            crec = _safe_corrcoef(sorted_W, Weq_np)

            corr_ff_hist[c] = cff
            corr_rec_hist[c] = crec
            j_max_hist[c] = float(J_np.max())
            w_max_hist[c] = float(W_np.max())
            chunks_completed = c

            pbar.set_postfix(cff=f"{cff:.3f}", crec=f"{crec:.3f}", Jmax=f"{j_max_hist[c]:.2e}")

            if (save_intermediate and (c % args.save_every == 0)) or (c == chunks):
                np.savez_compressed(
                    run_dir / f"snapshot_{c:04d}.npz",
                    J_ff=J_np,
                    W_rec=W_np,
                    chunk=c,
                    time_s=float(time_s[c]),
                    corr_ff=cff,
                    corr_rec=crec,
                    params=run_params,
                )

            if args.from_stable and (cff < 0.2 and crec < 0.2):
                pbar.write(f"Early stop at chunk {c} (from_stable diverged: cff={cff:.3f}, crec={crec:.3f}).")
                break

        elapsed = time.perf_counter() - t_start

        # Save final state and full chunk-wise learning history
        np.savez_compressed(
            run_dir / "final_state.npz",
            J_ff=np.asarray(J, dtype=np.float32),
            W_rec=np.asarray(W, dtype=np.float32),
            U=np.asarray(U, dtype=np.float32),
            V=np.asarray(V, dtype=np.float32),
            x=float(x),
            idx_sort=np.asarray(idx_sort, dtype=np.int32),
            chunks_completed=chunks_completed,
            time_s=float(time_s[chunks_completed]),
            elapsed_wall_s=elapsed,
            params=run_params,
        )

        np.savez_compressed(
            run_dir / "learning_history.npz",
            chunks=chunk_ids[: chunks_completed + 1],
            time_s=time_s[: chunks_completed + 1],
            corr_ff=corr_ff_hist[: chunks_completed + 1],
            corr_rec=corr_rec_hist[: chunks_completed + 1],
            j_max=j_max_hist[: chunks_completed + 1],
            w_max=w_max_hist[: chunks_completed + 1],
            params=run_params,
        )

        print(
            f"[m-{i}] Finished {chunks_completed}/{chunks} chunks in {elapsed:.1f}s | "
            f"final corr_FF={corr_ff_hist[chunks_completed]:.4f}, "
            f"corr_REC={corr_rec_hist[chunks_completed]:.4f}"
        )


if __name__ == "__main__":
    main()