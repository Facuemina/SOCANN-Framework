# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 14:08:23 2025

@author: Facundo
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects

# Add repo root to Python's module path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))
from src.theory import learned_U_std, gamma
from src.engine import gaussian

plt.rcParams['axes.titlesize'] = 25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 15


# def estimate_g_periodic(U, x0, sigma, L):
#     N = len(U)
#     pos = np.linspace(0, L, N, endpoint=False)
#     d = (pos - x0 + L / 2.0) % L - L / 2.0
#     g0 = np.exp(-0.5 * (d / sigma) ** 2)
#     g1 = (d / sigma) * g0
#     A = np.column_stack([g0, g1])
#     (c0, c1), _, _, _ = np.linalg.lstsq(A, U, rcond=None)
#     return c0, c1

def estimate_g_periodic(y, mu, std, L):
    x_ = np.linspace(0,L,len(y),False)
    dx = x_[1] - x_[0]
    z_ = (x_ - mu + L/2) % L - L/2
    def CANN_eigen_func0(z, std, L):
        #norm = jnp.sqrt(jnp.sqrt(jnp.pi)*2*std)
        norm = 1
        return norm * gaussian(z, 0, std, L)
    def CANN_eigen_func1(z, std, L):
        #norm = jnp.sqrt((jnp.sqrt(jnp.pi) * 2 * std) / (2))
        norm = 1
        return norm * gaussian(z, 0, std, L) * 2 * z / std
    
    a0 = np.sum(CANN_eigen_func0(z_, std, L) * y) * dx
    a1 = np.sum(CANN_eigen_func1(z_, std, L) * y) * dx
    
    return a0, a1


def compute_gamma_contours(Gamma_val, m_lims, v_lims, nfig=1,
                           figsize=(10, 8), dpi=400):
    # 1. Parameters & Levels
    gamma_levels = np.array([-.025, 0, .025, .05, 0.075, .1])
    gamma_levels[np.abs(gamma_levels) < 1e-8] = 0

    # 2. Range definitions & Meshgrid
    m_range = np.linspace(0., m_lims[1], 1000)
    v_range = np.linspace(v_lims[0], v_lims[1], 1000)
    M, V = np.meshgrid(m_range, v_range)

    GAMMA_FIELD = gamma(M, V, Gamma_val)

    # 3. Construct Styling Lists
    colors = []
    linestyles = []
    for val in gamma_levels:
        if val < 0:
            colors.append('royalblue')
            linestyles.append('solid')
        elif val == 0:
            colors.append('black')
            linestyles.append('dashed')
        else:
            colors.append('crimson')
            linestyles.append('solid')

    plt.figure(nfig, figsize=figsize, dpi=dpi)

    # 1. Background
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=0.2)
    mesh = plt.pcolormesh(v_range, m_range, GAMMA_FIELD.T,
                          cmap='RdBu_r', norm=norm, shading='auto',
                          alpha=0.4, rasterized=True)

    # 2. Contour Lines
    contours = plt.contour(V, M, GAMMA_FIELD, levels=gamma_levels,
                           colors=colors, linestyles=linestyles, linewidths=2.5)

    # 3. LABEL PLACEMENT
    x_min, x_max = v_lims[0], v_lims[1]
    target_x_columna = x_min + 0.55 * (x_max - x_min)

    label_positions = []
    for i, segs in enumerate(contours.allsegs):
        if len(segs) > 0 and len(segs[0]) > 0:
            path = segs[0]
            if i == 0:
                mid_idx = len(path) // 2
                label_positions.append((path[mid_idx, 0], path[mid_idx, 1]))
            else:
                idx_closest = np.argmin(np.abs(path[:, 0] - target_x_columna))
                label_positions.append((path[idx_closest, 0], path[idx_closest, 1]))

    clabels = plt.clabel(contours, inline=True, fontsize=15, fmt='%.3f',
                         colors='black', inline_spacing=10, manual=label_positions)

    plt.setp(clabels, path_effects=[
        patheffects.withStroke(linewidth=2.5, foreground="white")
    ])

    # 4. Colorbar and Labels
    cbar = plt.colorbar(mesh)
    cbar.ax.set_title(r'$\gamma$')
    plt.ylabel('$m$')
    plt.xlabel(r'$\frac{v\tau}{\sqrt{2}\sigma_u}$')

    plt.tight_layout()


#%% Compute empirical gammas
path2load = repo_root.parent / 'figure3-ff-asymmetry-data'
data = np.load(path2load / 'ff_asymmetry_results.npz', allow_pickle=True)
sim_params = data['params'].item()

U_all = data['U_all']
x_all = data['x_all']

n = sim_params['N']
input_std = sim_params['input_std']
tau = sim_params['tau']
tauv = sim_params['tauv']
k = sim_params['k']
R0 = sim_params['A_R']
L = sim_params['L']
u_std = learned_U_std(sim_params['beta'], input_std)

m_vals = sim_params['m_vals']
v_vals = sim_params['v_vals']

gammas = np.zeros((len(m_vals), len(v_vals)))
gammas_theory = gammas.copy()

for i, m in enumerate(m_vals):
    for j, v in enumerate(v_vals):
        U = U_all[i, j]
        x = x_all[i, j]

        c0, c1 = estimate_g_periodic(U.T[0], x, u_std, L)
        gammas[i, j] = c1 / c0
        gammas_theory[i, j] = np.sqrt(2) * gamma(m, v * tau / (np.sqrt(2) * u_std), tauv / tau)


#%% Horizontal Version
ncols = 2
lrows = 8

plt.figure(1, figsize=(lrows * ncols, lrows))
plt.subplot(1, ncols, 1)
norm = tau / (np.sqrt(2) * u_std)
compute_gamma_contours(
    tauv / tau,
    (0.0, 0.5),
    (2.0 * norm, 75.0 * norm),
    nfig=1,
    figsize=(6, 5),
    dpi=400,
)

idx2plot = [0, 1, 2]
for i, idx in enumerate(idx2plot):
    plt.subplot(3, ncols, ncols * i + ncols)
    plt.plot(m_vals, gammas_theory[:, idx], '--', label='Theory', color='k')
    plt.plot(m_vals, gammas[:, idx], '*', label='Simulation', color='r')
    plt.xlabel('m')
    plt.ylabel(r'$\gamma$')
    plt.title(f'v={int(v_vals[idx])}cm/s')
    plt.legend()

plt.tight_layout()
plt.savefig(path2load / 'GammaAsymmetries.png', format='png', dpi=300)
plt.show()