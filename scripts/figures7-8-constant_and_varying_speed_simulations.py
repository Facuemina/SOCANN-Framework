#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the combined simulation results from figure7_8-path-integration-data/
and plots/saves both Figure 7 (baseline shift relations) and Figure 8
(bump width scaling, speed-vs-input curves, and time-varying path integration)
to figures/.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory (the repo root) to Python's module path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.theory import theoretical_speed


def main():
    # 1. Locate the Data Directory
    repo_root = Path(__file__).resolve().parent.parent
    # Uncomment if data is outside the repo
    # data_dir = repo_root.parent / "figure7_8-path-integration-data"
    # Comment if data is outside the repo
    data_dir = repo_root / "data" / "figure7_8-path-integration-data"
    base_fig_dir = repo_root / "figures"
    base_fig_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / "figure7_8_results.npz"
    if not data_file.exists():
        print(f"[!] Error: Data file {data_file} not found.")
        print("    Make sure you have run 'scripts/script-figure7_8-path_integration.py' first.")
        return

    # 2. Load Data and Parameters
    data = np.load(data_file, allow_pickle=True)
    sim_params = data["params"].item()

    N = int(sim_params["N"])
    L = float(sim_params["L"])
    dt = float(sim_params["dt"])
    tau = float(sim_params["tau"])
    tauv = float(sim_params["tauv"])
    k = float(sim_params["k"])
    W0 = float(sim_params["W0"])
    input_std = float(sim_params["input_std"])
    m_vals = np.array(sim_params["m_vals"])
    v_vals = np.array(sim_params["v_vals"])
    track_steps = int(sim_params["track_steps"])

    U_all = data["U_all"]
    Iu = data["Iu"]
    Iv = data["Iv"]
    I_vel = data["I_vel"]
    speed_u = data["speed_u"]
    vels = data["vels"]
    bump_speed = data["bump_speed"]

    # Recompute circular std and normalized amplitude from U - I_u
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    std_u2 = np.zeros_like(Iu)
    u_max_vals = np.zeros_like(Iu)
    for im in range(len(m_vals)):
        for iv in range(len(v_vals)):
            U = U_all[im, iv]
            u0 = np.maximum(U - Iu[im, iv], 0.0)
            C = np.sum(np.cos(angles) * u0) / np.sum(u0)
            S = np.sum(np.sin(angles) * u0) / np.sum(u0)
            R = np.sqrt(C**2 + S**2)
            std_u2[im, iv] = np.sqrt(-2.0 * np.log(R)) / (2.0 * np.pi) * L
            u_max_vals[im, iv] = (U.max() - Iu[im, iv]) * np.sqrt(2.0 * np.pi) * std_u2[im, iv]

    # 3. Plot Figure 7: Baseline Shift Relations
    fig7 = plt.figure(figsize=(10, 5), dpi=400)
    ticks = np.arange(-5.0, 7.5, 2.5)

    ax7_1 = fig7.add_subplot(121)
    ax7_1.plot(
        [I_vel.min() * 1.5, I_vel.max() * 1.5],
        [I_vel.min() * 1.5, I_vel.max() * 1.5],
        "--b",
        label="Id. line",
    )
    for i in range(len(m_vals)):
        ax7_1.plot(
            I_vel[i],
            Iu[i] + Iv[i],
            "*",
            color=[i / len(m_vals), 0, 0],
            label=f"m = {m_vals[i]:.3f}",
        )
    ax7_1.set_ylabel(r"$(\epsilon_u + \epsilon_v)\cdot I_{speed}$", fontsize=18)
    ax7_1.set_xlabel(r"$I_{speed}$", fontsize=18)
    ax7_1.set_xticks(ticks)
    ax7_1.set_yticks(ticks)
    ax7_1.tick_params(axis="both", labelsize=15)
    ax7_1.set_xlim([I_vel.min() * 1.3, I_vel.max() * 1.3])
    ax7_1.set_ylim([I_vel.min() * 1.3, I_vel.max() * 1.3])
    ax7_1.legend(fontsize=14)

    ax7_2 = fig7.add_subplot(122)
    ax7_2.plot(
        [I_vel.min() * 1.12, I_vel.max() * 1.12],
        [I_vel.min() * 1.12, I_vel.max() * 1.12],
        "--b",
        label="Id. line",
    )
    for i in range(len(m_vals)):
        ax7_2.plot(
            Iv[i],
            Iu[i] * m_vals[i],
            marker="*",
            linestyle="",
            color=[i / len(m_vals), 0, 0],
        )
    ax7_2.set_xlabel(r"$\epsilon_v \cdot I_{speed}$", fontsize=18)
    ax7_2.set_ylabel(r"$m\cdot \epsilon_u \cdot I_{speed}$", fontsize=18)
    ax7_2.set_xticks(ticks / 10.0)
    ax7_2.set_yticks(ticks / 10.0)
    ax7_2.tick_params(axis="both", labelsize=15)
    ax7_2.set_xlim([Iv.min() * 1.2, Iv.max() * 1.2])
    ax7_2.set_ylim([Iv.min() * 1.2, Iv.max() * 1.2])

    fig7.tight_layout()
    fig7_svg = base_fig_dir / "figure7-epsilons.svg"
    fig7_png = base_fig_dir / "figure7-epsilons.png"
    fig7.savefig(fig7_svg, format="svg")
    fig7.savefig(fig7_png, format="png", dpi=300)

    # 4. Plot Figure 8: Width Scaling, Speed Curve, and Time-Varying Path Integration
    fontsize = 35
    addfsize = 3
    linewidth = 5
    markersize = 10

    fig8 = plt.figure(figsize=(20, 20), dpi=400)

    # --- Top-Left (221): Bump Width Ratio vs. Analytical Scaling ---
    ax1 = fig8.add_subplot(221)
    ax2 = ax1.twinx()

    mid_idx = int(np.argmin(np.abs(v_vals)))
    VAR = 1.0 + np.sqrt(2.0 * np.pi) * std_u2 * Iu / u_max_vals
    VAR_STD = std_u2 / std_u2[:, mid_idx : mid_idx + 1]

    for i in range(len(m_vals)):
        ax1.plot(
            I_vel[i], VAR[i], "-.",
            color=[i / len(m_vals), 0, 0],
            label="Theory" if i == 0 else None,
            linewidth=linewidth,
        )
        ax2.plot(
            I_vel[i], VAR_STD[i], "o",
            color=[0, 0, (i + 1) / (len(m_vals) + 1)],
            label="Simulation" if i == 0 else None,
            alpha=0.5,
            markersize=markersize,
        )

    ymax = 1.025 * max(np.nanmax(VAR_STD), np.nanmax(VAR))
    mn = min(np.nanmin(VAR_STD), np.nanmin(VAR))
    ymin = 1.025 * mn if mn < 0 else 0.975 * mn
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    yticks_w = np.arange(np.round(ymin * 10.0) / 10.0, np.round(ymax * 10.0) / 10.0 + 0.1, 0.1)
    ax1.set_xlabel(r"$I_{speed}$", fontsize=fontsize + addfsize)
    ax1.set_ylabel(
        r"$1 + \frac{\epsilon_u I_{speed}}{U_{max}(I_{speed})}$",
        fontsize=fontsize + addfsize,
        color=[0.5, 0, 0],
    )
    ax1.tick_params(axis="y", labelcolor=[0.5, 0, 0], labelsize=fontsize)
    ax1.set_yticks(yticks_w)
    ax1.set_xticks([-5, 0, 5])
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.legend(fontsize=fontsize - addfsize, loc="upper left")

    ax2.set_ylabel(
        r"$\frac{\mathrm{std}[U(I_{speed})]_x}{\mathrm{std}[U(0)]_x}$",
        fontsize=fontsize + addfsize,
        color=[0, 0, 0.5],
    )
    ax2.tick_params(axis="y", labelcolor=[0, 0, 0.5], labelsize=fontsize)
    ax2.set_yticks(yticks_w)
    ax2.legend(fontsize=fontsize - addfsize, loc="lower right")

    # --- Top-Right (222): Bump Speed vs. I_speed ---
    ax_sp = fig8.add_subplot(222)
    rho = N / L
    k_cont = k * L / N
    sigma_u_th = np.sqrt(2.0) * input_std
    a0 = sigma_u_th * np.sqrt(2.0 * np.pi)
    norm = np.sqrt(2.0 * np.pi) * input_std

    for i in range(len(m_vals)):
        m_i = m_vals[i]
        a_q = k_cont * rho * a0 * (1.0 + np.sqrt(m_i * tau / tauv))
        b_q = -rho * W0
        c_q = 1.0 + np.sqrt(m_i * tau / tauv)
        u0_th = (-b_q + np.sqrt(b_q**2 - 4.0 * a_q * c_q)) / (2.0 * a_q) * a0

        std_eff = input_std * (1.0 + norm * (I_vel[i] / (m_i + 1.0)) / u0_th)
        v0 = theoretical_speed(m_i, tau, tauv, std_eff)

        ax_sp.plot(I_vel[i], v0, "--", color=[i / len(m_vals), 0, 0], linewidth=linewidth)
        ax_sp.plot(
            I_vel[i], speed_u[i], "*",
            color=[i / len(m_vals), 0, 0], linewidth=linewidth, markersize=markersize,
        )

    ax_sp.set_xlabel(r"$I_{speed}$", fontsize=fontsize + addfsize)
    ax_sp.set_ylabel("bump speed (cm / s)", fontsize=fontsize + addfsize)
    ax_sp.set_yticks(np.arange(0, 35, 10))
    ax_sp.set_xticks([-5, 0, 5])
    ax_sp.tick_params(axis="both", labelsize=fontsize)

    # --- Bottom (212): External Speed vs. Tracked Bump Speed ---
    ax_tr = fig8.add_subplot(212)
    ax_tr.plot(np.arange(track_steps) * dt, vels, label="External Speed", linewidth=4)
    ax_tr.plot(np.arange(track_steps - 1) * dt, bump_speed, "r", label="Bump Speed", linewidth=4)
    ax_tr.set_xlabel("Time (s)", fontsize=fontsize)
    ax_tr.set_ylabel("Speed (cm/s)", fontsize=fontsize)
    ax_tr.tick_params(axis="both", labelsize=fontsize)
    ax_tr.legend(fontsize=fontsize)

    fig8.tight_layout()
    fig8_svg = base_fig_dir / "figure8-path_integration.svg"
    fig8_png = base_fig_dir / "figure8-path_integration.png"
    fig8.savefig(fig8_svg, format="svg")
    fig8.savefig(fig8_png, format="png", dpi=300)

    print(f"[*] Saved Figures 7 & 8 to:\n    {fig7_png}\n    {fig8_png}")
    plt.show()


if __name__ == "__main__":
    main()