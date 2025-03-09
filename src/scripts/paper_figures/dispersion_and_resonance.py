"""
dispersion_and_resonance.py

Generates and saves figures depicting dispersion relations,
resonance conditions, and their intersections.

This script calculates and visualises the dispersion relation and
resonance conditions, also plotting their intersections. It generates
a series of figures, 'disp_and_res_???.png', where '???' is a three-digit
integer padded with zeros, ranging from 0 to X_npoints - 1.
The parameter X=tan(psi) is varied linearly across the range [X_min, X_max],
and X_npoints specifies the number of discrete values of X.

Parameters for the calculations and plotting can be adjusted within the
'Parameters' section of the main function.

Usage:
    python path/to/dispersion_and_resonance.py
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib.ticker import MultipleLocator

from piran.cpdr import Cpdr
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


def get_resonance_condition(cpdr: Cpdr, X, y_list, k_par_sign):
    """
    k_par_sign: 'positive' for theta in [0, pi/2] or
        'negative' for theta in (pi/2, pi].
    """
    psi = np.arctan(X)
    if k_par_sign == "positive":
        theta = psi.to(u.rad)
    elif k_par_sign == "negative":
        theta = (np.pi << u.rad) - psi.to(u.rad)

    n = cpdr.resonance
    electron_gyro = cpdr.plasma.gyro_freq[0]
    gamma = cpdr.gamma
    v_par = cpdr.v_par

    resonance_condition = []
    for y in y_list:
        omega = np.abs(electron_gyro) * y
        res_cond_k = (omega - (n * electron_gyro / gamma)) / (np.cos(theta) * v_par)
        x = res_cond_k * const.c / np.abs(electron_gyro)
        resonance_condition.append((x, y))

    return resonance_condition


def get_dispersion_relation(cpdr: Cpdr, X, y_list):
    dispersion_relation = []
    for y in y_list:
        electron_gyro = cpdr.plasma.gyro_freq[0]
        omega = np.abs(electron_gyro) * y

        try:
            k_root = cpdr.solve_cpdr(omega, X)
            is_desired_wave_mode = [cpdr.filter(X, omega, k) for k in k_root]
            k_filtered = k_root[is_desired_wave_mode]
        except Exception:
            continue

        for k in k_filtered:
            if not np.isnan(k):
                x = k * const.c / np.abs(electron_gyro)
                dispersion_relation.append((x, y))

    return dispersion_relation


def plot_data(
    ax,
    values,
    color="k",
    linestyle="-",
    marker="",
    alpha=1.0,
    label="",
    zorder=0,
    linewidth=1.2,
):
    xx = [val[0].value for val in values]
    yy = [val[1].value for val in values]

    ax.semilogy(
        xx,
        yy,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def format_figure(
    ax,
    cpdr,
    X,
):
    psi = np.arctan(X)
    electron_gyro_abs = np.abs(cpdr.plasma.gyro_freq[0])
    lower_cutoff = cpdr.omega_lc / electron_gyro_abs
    upper_cutoff = cpdr.omega_uc / electron_gyro_abs
    if cpdr.energy.to(u.keV).value < 1000:
        energy_name = rf"{cpdr.energy.to(u.keV).value:.0f}$\,$keV"
    else:
        energy_name = rf"{cpdr.energy.to(u.MeV).value:.1f}$\,$MeV"

    x_lim_min = 0
    x_lim_max = 20

    y_lim_min = 0.1
    y_lim_max = 1.0

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    yticks = [0.1, lower_cutoff, upper_cutoff, 1.0]
    yticks_labels = [str(yticks[0]), r"$\frac{\omega_{\text{LC}}}{| \Omega_e |}$", r"$\frac{\omega_{\text{UC}}}{| \Omega_e |}$", str(yticks[-1])]
    ax.set_yticks(yticks, labels=yticks_labels)
    ax.set_yticks(np.arange(0.2, 1.0, 0.1), labels=[], minor=True)

    ax.tick_params("x", which="both", top=True, labeltop=False)
    ax.tick_params("y", which="both", right=True, labelright=False)

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_xlabel(r"$k \frac{c}{| \Omega_e |}$")
    ax.set_ylabel(r"$\frac{\omega}{| \Omega_e |}$")

    ax.set_title(rf"E={energy_name}, $\alpha={cpdr.alpha.deg}^\circ, \psi={psi.to(u.deg).value:.2f}^\circ$")

    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", prop={"size": 8})


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0.0, u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 << u.MeV
    alpha = Angle(76, u.deg)
    resonances = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    X_min = 1.19175  # 1.19175 for 50deg or 1.73205 for 60deg
    X_max = 5.0
    X_npoints = 1
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    # Scale it by abs(electron gyrofrequency) to get omega.
    y_min = 0.1 << u.dimensionless_unscaled
    y_max = 1.0 << u.dimensionless_unscaled
    y_list = np.linspace(y_min, y_max, num=1000)

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    # =================================================

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

    for i, X in enumerate(X_range):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))

        for j, resonance in enumerate(resonances):
            cpdr = Cpdr(plasma_point, energy, alpha, resonance, freq_cutoff_params)
            electron_gyro_abs = np.abs(cpdr.plasma.gyro_freq[0])

            # The values that we print here are the same for every X,
            # so print them only once per resonance.
            if i == 0:
                print(f"Resonance: {resonance}")
                print(f"Plasma: {particles}")
                print(f"Energy: {energy}")
                print(f"Pitch angle: {alpha}")
                print(f"Electron plasma-to-gyro ratio: {plasma_over_gyro_ratio}")
                print(f"Electron gyrofrequency (modulus): {np.abs(cpdr.plasma.gyro_freq[0]):.1f}")
                print(f"Electron plasma frequency: {cpdr.plasma.plasma_freq[0]:.1f}")
                print(f"Proton gyrofrequency: {cpdr.plasma.gyro_freq[1]:.1f}")
                print(f"Proton plasma frequency: {cpdr.plasma.plasma_freq[1]:.1f}")
                print()

                print("Cutoffs as defined in Glauert 2005")
                print(f"omega_lc: {cpdr.omega_lc:.1f}")
                print(f"omega_uc: {cpdr.omega_uc:.1f}")
                print()

                print("Lower hybrid frequency from Artemyev 2016")
                omega_lh = np.sqrt(
                    np.abs(cpdr.plasma.gyro_freq[0]) * cpdr.plasma.gyro_freq[1]
                ) / np.sqrt(1 + (cpdr.plasma.gyro_freq[0] ** 2 / cpdr.plasma.plasma_freq[0] ** 2))
                print(f"omega_lh: {omega_lh:.1f}")
                print(
                    f"approximate omega_lh: {np.sqrt(np.abs(cpdr.plasma.gyro_freq[0]) * cpdr.plasma.gyro_freq[1]):.1f}"
                )
                print(f"\n{'=' * 80}\n")

            # Plot resonance condition.
            resonance_condition_kpar_pos = get_resonance_condition(
                cpdr, X, y_list, "positive"
            )
            resonance_condition_kpar_neg = get_resonance_condition(
                cpdr, X, y_list, "negative"
            )
            plot_data(ax, resonance_condition_kpar_pos, colors[j], "-", "", 0.85, rf"Resonance condition n={resonance}", 2)
            plot_data(ax, resonance_condition_kpar_neg, colors[j], "--", "", 0.85, "", 2)

            # Plot resonant roots
            for root in cpdr.solve_resonant(X)[0]:
                if np.isnan(root.omega) or np.isnan(root.k):
                    continue
                root_x = root.k * (const.c / electron_gyro_abs)
                root_y = root.omega / electron_gyro_abs
                marker = "o" if root.k_par >= 0 else "h"
                plot_data(ax, [(root_x, root_y)], colors[j], "", marker, 0.9, "", 3)

        # Plot dispersion relation.
        # To compute the dispersion relation, we can reuse a Cpdr object,
        # as it does not depend on the specific resonance.
        dispersion_relation = get_dispersion_relation(cpdr, X, y_list)
        plot_data(ax, dispersion_relation, "k", "-", "", 0.7, "Dispersion relation", 1)

        # Plot upper and lower cutoffs
        lower_upper_x = [0, 20] << u.dimensionless_unscaled
        lower_y = [cpdr.omega_lc / electron_gyro_abs for val in lower_upper_x]
        upper_y = [cpdr.omega_uc / electron_gyro_abs for val in lower_upper_x]
        plot_data(ax, [(x, y) for x, y in zip(lower_upper_x, lower_y)], "k", ":", "", 0.6, "", 0, 1.0)
        plot_data(ax, [(x, y) for x, y in zip(lower_upper_x, upper_y)], "k", ":", "", 0.6, "", 0, 1.0)

        format_figure(ax, cpdr, X)
        fig.tight_layout()
        fig.savefig(f"disp_and_res_{i:03d}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
