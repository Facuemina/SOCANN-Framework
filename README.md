# SOCANN-Framework

[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue?logo=google-colab)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A high-performance JAX-based framework for simulating **Continuous Attractor Neural Networks (CANNs)** undergoing **Hebbian plasticity**. 

This repository provides the computational tools to simulate self-organizing neural dynamics. It is optimized for GPU acceleration, memory-efficient integration, and large-scale parameter grid searches.

## Key Features
* **JAX-Accelerated Physics:** Custom ODE integrators (`lax.scan`, `lax.fori_loop`) for lightning-fast, compiled neural simulations.
* **Synaptic Plasticity Engines:** Highly optimized, in-place update loops for both feedforward and recurrent Hebbian weight learning.
* **Modular Architecture:** A clean separation of analytical theory, core physics kernels, and data post-processing.
* **Reproducible Research:** Dedicated scripts to exactly reproduce the figures and grid searches from the associated publication.

---

## Repository Structure

```text
SelfOrganizedCANN-Framework/
│
├── src/                    # Core Library
│   ├── engine.py           # JAX-accelerated physics and simulation loops
│   ├── theory.py           # Analytical formulations and equilibrium solutions
│   └── helpers.py          # Matrix manipulation, sorting, and smoothing tools
│
├── demos/                  # Out-of-the-box Interactive Examples
│   ├── demo_learning_ff_rec.py  # Runs a learning simulation and saves compressed snapshots
│   └── plot_demo_results.py     # Visualizes the learned weight structures
│
├── scripts/                # Execution scripts for heavy compute jobs
│   ├── gridsearch_m.py     # Multiprocessing launcher for adaptation parameters
│   └── gridsearch_alphas.py# Multiprocessing launcher for weight learning alphas
│
├── requirements.txt        # Python dependencies
└── README.md
