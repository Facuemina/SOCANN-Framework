#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the compressed snapshots from the CANN demo and plots the 
initial (random) and final (learned) states of the network.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import jax.numpy as jnp

# Import the sorting function from your framework
from src.helpers import sort_connectivity_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Plot initial and final states from CANN demo.")
    parser.add_argument("--N", type=int, default=512, help="Number of neurons to match the demo data")
    parser.add_argument("--total_steps", type=int, default=500000, help="Total simulation steps to match the demo data")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Locate the Data Directory
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root.parent / 'ff-rec-data' / f'demo_N{args.N}_steps{args.total_steps}'
    
    if not data_dir.exists():
        print(f"[!] Error: Data directory {data_dir} not found.")
        print("    Make sure you have run 'demo_learning_ff_rec.py' first.")
        return

    # 2. Find Initial and Final Snapshots
    snapshot_files = sorted(glob.glob(str(data_dir / "snapshot_*.npz")))
    if len(snapshot_files) < 2:
        print("[!] Error: Not enough snapshots found to compare initial and final states.")
        return

    init_file = snapshot_files[0]
    final_file = snapshot_files[-1]

    print(f"[*] Loading Initial State: {Path(init_file).name}")
    print(f"[*] Loading Final State:   {Path(final_file).name}")

    init_data = np.load(init_file)
    final_data = np.load(final_file)

    # 3. Sort the Final Matrices to reveal the learned structure
    # Feedforward sorting (sorts post-synaptic target rows)
    W_ff_final_sorted, idx_ff = sort_connectivity_matrix(jnp.array(final_data['W_ff']))
    
    # Recurrent sorting (sorts both rows and columns to keep the diagonal structure)
    # W_rec_final, idx_rec = sort_connectivity_matrix(jnp.array(final_data['W_rec']))
    W_rec_final_sorted = final_data['W_rec'][:, idx_ff][idx_ff] # Apply to columns as well

    # 4. Plotting Setup
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('Self-Organizing CANNs: Initial vs. Final State', fontsize=20, y=1.05)

    cmap = 'jet'  # Using your preferred colormap
    x_axis = np.arange(args.N)

    # --- ROW 1: INITIAL CONDITIONS ---
    axes[0, 0].plot(x_axis, init_data['U'], color='b', lw=2)
    axes[0, 0].set_title('Initial U Activity')
    axes[0, 0].set_ylabel('Initial', fontsize=16, fontweight='bold')

    axes[0, 1].plot(x_axis, init_data['V'], color='r', lw=2)
    axes[0, 1].set_title('Initial V (Adaptation)')

    im_ff_init = axes[0, 2].imshow(init_data['W_ff'], aspect='auto', cmap=cmap)
    axes[0, 2].set_title('Initial W_ff (Random)')
    fig.colorbar(im_ff_init, ax=axes[0, 2])

    im_rec_init = axes[0, 3].imshow(init_data['W_rec'], aspect='auto', cmap=cmap)
    axes[0, 3].set_title('Initial W_rec (Random)')
    fig.colorbar(im_rec_init, ax=axes[0, 3])

    # --- ROW 2: FINAL CONDITIONS ---
    axes[1, 0].plot(x_axis, final_data['U'][idx_ff], color='b', lw=2)
    axes[1, 0].set_title('Final U Activity')
    axes[1, 0].set_ylabel('Final', fontsize=16, fontweight='bold')
    axes[1, 0].set_xlabel('Neuron Index')

    axes[1, 1].plot(x_axis, final_data['V'][idx_ff], color='r', lw=2)
    axes[1, 1].set_title('Final V (Adaptation)')
    axes[1, 1].set_xlabel('Neuron Index')

    # Plot sorted feedforward matrix
    im_ff_final = axes[1, 2].imshow(W_ff_final_sorted, aspect='auto', cmap=cmap)
    axes[1, 2].set_title('Final W_ff (Sorted)')
    axes[1, 2].set_xlabel('Input Index')
    axes[1, 2].set_ylabel('Sorted Neuron Index')
    fig.colorbar(im_ff_final, ax=axes[1, 2])

    # Plot fully sorted recurrent matrix
    im_rec_final = axes[1, 3].imshow(W_rec_final_sorted, aspect='auto', cmap=cmap)
    axes[1, 3].set_title('Final W_rec (Sorted)')
    axes[1, 3].set_xlabel('Sorted Neuron Index')
    axes[1, 3].set_ylabel('Sorted Neuron Index')
    fig.colorbar(im_rec_final, ax=axes[1, 3])

    # Clean up layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()