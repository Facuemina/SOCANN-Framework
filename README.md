[README.md](https://github.com/user-attachments/files/35179566/README.md)
# SOCANN-Framework

[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue?logo=google-colab)](https://jax.readthedocs.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2606.14649-b31b1b.svg)](https://arxiv.org/abs/2606.14649)
[![Phys. Rev. E](https://img.shields.io/badge/Phys.%20Rev.%20E-114%2C%20054405-informational)](https://doi.org/10.1103/syk9-5c24)

A high-performance, JAX-based framework for simulating **Self-Organizing Continuous Attractor Neural Networks (SOCANNs)**: networks whose attractor connectivity is *learned* through Hebbian plasticity rather than pre-wired.

This is the official code for:

> **Prospective coding and path integration emerge as equilibrium solutions of self-organizing neural networks with firing-rate adaptation**
> Facundo Emina and Emilio Kropff
> *Phys. Rev. E* **114**, 054405 (2026) · [DOI: 10.1103/syk9-5c24](https://doi.org/10.1103/syk9-5c24) · [arXiv:2606.14649](https://arxiv.org/abs/2606.14649)

---

## Table of Contents

- [Background](#background)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing the Paper](#reproducing-the-paper)
- [Citation](#citation)
- [License](#license)

---

## Background

Continuous Attractor Neural Networks (CANNs) are a standard model for how the brain represents continuous variables such as position or head direction. They usually rely on recurrent connectivity that is assumed to be pre-wired. This work asks how that structure could arise through learning.

The model has a two-layer architecture: an input "tutor" layer feeding a competitive layer. Three ingredients drive the self-organization:

- **Hebbian plasticity** shapes the synaptic weights.
- **Firing-rate adaptation** acts on the neurons' dynamics.
- **Global inhibition** forces neurons to compete, so only a localized bump of activity persists.

The main results reproduced by this framework:

1. Translationally invariant inputs lead to stable, **Gaussian-profiled feedforward weights**.
2. **Prospective coding** emerges spontaneously: the activity bump shifts forward without recurrent excitatory collaterals. The shift is amplified linearly across layers in multilayer networks, consistent with anticipatory activity reported in the superficial layers of the entorhinal cortex.
3. With recurrent learning, the network can sustain a moving bump of activity.
4. A time-varying baseline current encoding speed lets the network act as a **unidirectional path integrator**.

These properties arise as equilibrium solutions of a single self-organizing system, which `theory.py` provides in analytical form to compare against simulation.

## Key Features

- **JAX-accelerated simulation:** custom ODE integrators built on `lax.scan` and `lax.fori_loop`, JIT-compiled for CPU, GPU, or TPU.
- **Plasticity engines:** optimized update loops for both feedforward and recurrent Hebbian weight learning.
- **Memory-efficient integration:** designed for long simulations and compressed snapshot saving.
- **Large-scale parameter sweeps:** multiprocessing launchers for grid searches over adaptation and learning-rate parameters.
- **Analytical theory module:** closed-form formulations and equilibrium solutions to validate numerical results.
- **Modular design:** analytical theory, core physics kernels, and post-processing are kept separate.
- **Reproducible research:** scripts to regenerate the figures and grid searches of the publication.

## Repository Structure

```
SOCANN-Framework/
│
├── src/                          # Core library
│   ├── engine.py                 # JAX-accelerated physics and simulation loops
│   ├── theory.py                 # Analytical formulations and equilibrium solutions
│   └── helpers.py                # Matrix manipulation, sorting, and smoothing tools
│
├── demos/                        # Ready-to-run examples
│   ├── demo_learning_ff_rec.py   # Learning simulation (feedforward + recurrent); saves compressed snapshots
│   └── plot_demo_results.py      # Visualizes the learned weight structures
│
├── scripts/                      # Everything needed to reproduce the paper
│   ├── scripts-figure<N>/        # Simulation scripts behind each figure (heavy compute)
│   ├── figure<N>/                # Plotting scripts that generate each figure
│
├── data/                         # Folder containing data coming from the scripts to reproduce results
│
├── figures/                      # PNG example figures (most of them shown in the paper)
│
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT
└── README.md
```

## Installation

```bash
git clone https://github.com/Facuemina/SOCANN-Framework.git
cd SOCANN-Framework

# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**GPU support.** The default install runs on CPU. For NVIDIA GPUs, install the CUDA build of JAX following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html), for example:

```bash
pip install -U "jax[cuda12]"
```

Verify which devices JAX sees:

```bash
python -c "import jax; print(jax.devices())"
```

## Quick Start

Run the demo to simulate learning of feedforward and recurrent weights and save snapshots of the learned structure:

```bash
python demos/demo_learning_ff_rec.py
```

Then visualize the learned weights:

```bash
python demos/plot_demo_results.py
```

The simulation engine is in `src/engine.py`. To build your own experiments, start from `demos/demo_learning_ff_rec.py` and adapt the network and plasticity parameters.

## Reproducing the Paper

Every figure in the paper is reproduced in two stages, and the `scripts/` folder is organized accordingly:

| Folder | Purpose |
| --- | --- |
| `scripts/scripts-figure<N>/` | **Simulation:** runs the simulations behind Figure `<N>` and saves the results. This is the compute-intensive step. |
| `scripts/figure<N>/` | **Plotting:** loads the saved results and generates Figure `<N>`. |

**Workflow for a given figure:**

```bash
# 1. Run the simulations for Figure N (may take a while; GPU recommended)
python scripts/scripts-figureN/<simulation_script>.py

# 2. Generate the plot from the saved results
python scripts/figureN/<plotting_script>.py
```

Replace `N` and the script names with the figure number and files you want to reproduce. Separating the two stages lets you re-style or tweak a figure without re-running the simulations.

Large parameter sweeps are launched with multiprocessing:

```bash
# Sweep over firing-rate adaptation parameters
python scripts/gridsearch_m.py

# Sweep over feedforward/recurrent learning rates (alphas)
python scripts/gridsearch_alphas.py
```

These jobs can be compute-intensive. A GPU and enough RAM for parallel workers are recommended. Analytical predictions from `src/theory.py` can be overlaid on simulation outputs to compare numerical and theoretical equilibria.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{emina2026prospective,
  title   = {Prospective coding and path integration emerge as equilibrium solutions of self-organizing neural networks with firing-rate adaptation},
  author  = {Emina, Facundo and Kropff, Emilio},
  journal = {Physical Review E},
  volume  = {114},
  pages   = {054405},
  year    = {2026},
  doi     = {10.1103/syk9-5c24},
  eprint  = {2606.14649},
  archivePrefix = {arXiv}
}
```

## Authors

**Facundo Emina** and **Emilio Kropff**
Universidad de Buenos Aires (FCEyN, Departamento de Física) and Leloir Institute (IIBBA/CONICET), Buenos Aires, Argentina.

## License

Released under the [MIT License](LICENSE).
