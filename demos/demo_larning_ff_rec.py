#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the SelfOrganizedCANN-Framework.
Initializes a Feedforward and Recurrent CANN with completely random weights,
drives it with a moving Gaussian input, and periodically saves compressed
snapshots of the network's state to an external directory.
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from tqdm import tqdm
import sys
from pathlib import Path

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import run_learning_ff_and_rec

def parse_args():
    parser = argparse.ArgumentParser(description="Run CANN learning demo with snapshots.")
    parser.add_argument("--N", type=int, default=512, help="Number of neurons")
    parser.add_argument("--L", type=float, default=100.0, help="Length of the periodic domain")
    parser.add_argument("--total_steps", type=int, default=500000, help="Total simulation steps")
    parser.add_argument("--snapshots", type=int, default=10, help="Number of equally distributed snapshots to save")
    parser.add_argument("--dt", type=float, default=0.001, help="Integration time step")
    parser.add_argument("--speed", type=float, default=15.0, help="Constant speed of the Gaussian input")
    parser.add_argument("--tau", type=float, default=0.01, help="U time constant")
    parser.add_argument("--tauv", type=float, default=0.1, help="V time constant")
    parser.add_argument("--k", type=float, default=0.1, help="Global inhibition")
    parser.add_argument("--m", type=float, default=0.4, help="Adaptation strength")
    parser.add_argument("--betaJ", type=float, default=0.5, help="Extent of FF synaptic decay")
    parser.add_argument("--betaW", type=float, default=0.5, help="Extent of REC synaptic decay")
    parser.add_argument("--etaJ", type=float, default=0.01, help="FF learning rate")
    parser.add_argument("--etaW", type=float, default=0.01, help="REC learning rate")    
    parser.add_argument("--alphaJ", type=float, default=.1, help="Strength of FF synaptic decay")    
    parser.add_argument("--alphaW", type=float, default=5, help="Strength of REC synaptic decay")   
    parser.add_argument("--A_R", type=float, default=30, help="Input amplitude") 
    parser.add_argument("--input_std", type=float, default=5, help="Input width") 
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight initialization")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Directory Setup (Strictly outside the repository)
    # This resolves to the folder containing this script, then moves up two levels (out of demos/ and out of repo/)
    repo_root = Path(__file__).resolve().parent.parent
    save_dir = repo_root.parent / 'ff-rec-data' / f'demo_N{args.N}_steps{args.total_steps}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Snapshots will be saved to: {save_dir}")

    # 2. Parameter Setup
    # Neural parameters (1/tau, 1/tauv, m, k, g, gain)
    neural_params = (1/args.tau, 1/args.tauv, args.m, args.k, 2.0, 1.0)
    
    # Learning parameters (dt, betaJ, betaW, etaJ, etaW, sR, L, v_eq, av, R0, ain, arec)
    
    learning_params = (args.dt, args.betaJ, args.betaW, args.etaJ, args.etaW, args.input_std, args.L, args.speed, 0.0, args.A_R, args.alphaJ, args.alphaW)
    
    pos = jnp.linspace(0, args.L, args.N, endpoint=False).reshape((args.N, 1))

    # 3. Purely Random Initialization
    key = jax.random.PRNGKey(args.seed)
    key_ff, key_rec, key_u, key_v = jax.random.split(key, 4)

    # Random uniform weights [0, 0.01]
    W_ff = jax.random.uniform(key_ff, (args.N, args.N), minval=0.0, maxval=0.01)
    
    # Recurrent weights must have a hollow diagonal (no self-connections)
    W_rec = jax.random.uniform(key_rec, (args.N, args.N), minval=0.0, maxval=0.01)
    diag_idx = jnp.diag_indices(args.N)
    W_rec = W_rec.at[diag_idx].set(0.0)

    # Small random initial activity
    U = jax.random.uniform(key_u, (args.N, 1), minval=0.0, maxval=0.1)
    V = jax.random.uniform(key_v, (args.N, 1), minval=0.0, maxval=0.1)
    
    x0 = 0.0 # Initial position of the stimulus
    
    # Constant speed array for the chunk
    steps_per_chunk = args.total_steps // args.snapshots
    v_array = jnp.ones(steps_per_chunk) * args.speed

    # 4. Simulation and Snapshot Loop
    print(f"[*] Starting simulation: {args.snapshots} chunks of {steps_per_chunk} steps.")
    
    # Save the initial state (Snapshot 0)
    np.savez_compressed(
        save_dir / "snapshot_000.npz", 
        U=np.array(U), 
        V=np.array(V), 
        W_ff=np.array(W_ff), 
        W_rec=np.array(W_rec)
    )

    for chunk in tqdm(range(1, args.snapshots + 1), desc="Simulating"):
        # Run JAX compiled loop for the chunk
        U, V, W_ff, W_rec, x0 = run_learning_ff_and_rec(
            U, V, W_ff, W_rec, x0, v_array, 
            neural_params, learning_params, pos, 
            T=steps_per_chunk
        )
        
        # Pull data to CPU and compress into .npz to save space
        np.savez_compressed(
            save_dir / f"snapshot_{chunk:03d}.npz", 
            U=np.array(U), 
            V=np.array(V), 
            W_ff=np.array(W_ff), 
            W_rec=np.array(W_rec),
            x_stimulus=np.array(x0)
        )

    print("[*] Demo complete. All snapshots saved successfully.")

if __name__ == "__main__":
    main()