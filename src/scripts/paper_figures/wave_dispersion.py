# Copyright (C) 2025 The University of Birmingham, United Kingdom /
#   Dr Oliver Allanson, ORCiD: 0000-0003-2353-8586, School Of Engineering, University of Birmingham /
#   Dr Thomas Kappas, ORCiD: 0009-0003-5888-2093, Advanced Research Computing, University of Birmingham /
#   Dr James Tyrrell, ORCiD: 0000-0002-2344-737X, Advanced Research Computing, University of Birmingham /
#   Dr Adrian Garcia, ORCiD: 0009-0007-4450-324X, Advanced Research Computing, University of Birmingham
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
wave_dispersion.py

Generates and saves three figures depicting wave dispersion relations
for different plasma-to-gyro ratios.

This script calculates and visualises wave dispersion relations for
plasma-to-gyro ratios of 0.75, 1.5, and 7.0.
The resulting figures, 'wave_dispersion_1.png', 'wave_dispersion_2.png',
and 'wave_dispersion_3.png', are saved to the current working directory.

Parameters can be adjusted within the 'Parameters' section of the main
function, but the script's functionality is only guaranteed for the
currently specified values.

Usage:
    python path/to/wave_dispersion.py
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib.ticker import LogFormatterMathtext

from piran.cpdr import Cpdr
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix
from piran.wavefilter import WaveFilter


class NullFilter(WaveFilter):
    """
    A filter that accepts all inputs.
    """

    @u.quantity_input
    def filter(
        self,
        X: u.Quantity[u.dimensionless_unscaled],
        omega: u.Quantity[u.rad / u.s],
        k: u.Quantity[u.rad / u.m],
        plasma: PlasmaPoint,
        stix: Stix,
    ) -> bool:
        return True


def draw_vertical_lines(ax, cpdr, omega_lh, omega_uh):
    # Draw omega_ce and omega_pe lines (electron)
    omega_ce_abs = np.abs(cpdr.plasma.gyro_freq[0]).value
    omega_pe = cpdr.plasma.plasma_freq[0].value
    ax.axvline(x=omega_pe, color="k", linestyle="--", linewidth=0.6)
    ax.axvline(x=omega_ce_abs, color="k", linestyle="--", linewidth=0.6)

    # Draw omega_L=0 and omega_R=0 lines
    omega_L0 = (np.sqrt(omega_ce_abs**2 + 4 * omega_pe**2) - omega_ce_abs) / 2
    omega_R0 = (np.sqrt(omega_ce_abs**2 + 4 * omega_pe**2) + omega_ce_abs) / 2
    ax.axvline(x=omega_L0, color="k", linestyle=":", linewidth=0.6)
    ax.axvline(x=omega_R0, color="k", linestyle=":", linewidth=0.6)

    # Draw proton gyrofrequency
    omega_pp = cpdr.plasma.gyro_freq[1].value
    ax.axvline(x=omega_pp, color="k", linestyle="-.", linewidth=0.6)

    # Draw lower and upper hybrid lines
    ax.axvline(x=omega_lh, color="k", linestyle="-.", linewidth=0.6)
    ax.axvline(x=omega_uh, color="k", linestyle="-.", linewidth=0.6)

    # Annotate
    ax.text(omega_ce_abs, 3.0 * 10**5, r"$\omega_{ce}$")
    ax.text(omega_pe, 1.0 * 10**5, r"$\omega_{pe}$")
    ax.text(omega_L0, 1.0 * 10**4, r"$\omega_{L=0}$")
    ax.text(omega_R0, 3.0 * 10**4, r"$\omega_{R=0}$")
    ax.text(omega_pp, 1.0 * 10**6, r"$\omega_{pp}$")
    ax.text(omega_lh, 1.0 * 10**6, r"$\omega_{LH}$")
    ax.text(omega_uh, 1.0 * 10**6, r"$\omega_{UH}$")


def plot_cpdr_roots(fig, ax, cpdr, X_range, omega_range):
    xaxis_values = []
    yaxis_values = []
    color_values = []
    for omega in omega_range:
        for X in X_range:
            k_list = cpdr.solve_cpdr(omega, X)

            for k in k_list:
                mu = const.c * abs(k) / omega

                xaxis_values.append(omega.value)
                yaxis_values.append((mu**2).value)
                color_values.append(X.value)

    im = ax.scatter(
        xaxis_values,
        yaxis_values,
        marker=".",
        s=4,
        c=color_values,
        cmap="viridis",
        vmin=X_range[0].value,
        vmax=X_range[-1].value,
        alpha=1.0,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel(r"Wave normal angle $X=\tan(\psi)$")


def plot_resonant_roots(ax, cpdr, X_range):
    xaxis_values = []
    yaxis_values = []

    resonant_roots = cpdr.solve_resonant(X_range)
    for row in resonant_roots:
        for root in row:
            mu = const.c * abs(root.k) / root.omega

            xaxis_values.append(root.omega.value)
            yaxis_values.append((mu**2).value)

    ax.scatter(
        xaxis_values,
        yaxis_values,
        marker=".",
        s=8,
        c="red",
    )


def format_figure(ax, energy, alpha, res, ratio):
    if ratio in [0.75, 1.5]:
        xticks = [10**i for i in range(1, 6)]
        yticks = [10**i for i in range(-2, 9)]

        xlim_min = xticks[0]
        xlim_max = 3 * xticks[-1]

        ylim_min = yticks[0]
        ylim_max = yticks[-1]
    elif ratio == 7.0:
        xticks = [10**i for i in range(1, 7)]
        yticks = [10**i for i in range(-2, 9)]

        xlim_min = xticks[0]
        xlim_max = 2 * xticks[-1]

        ylim_min = yticks[0]
        ylim_max = yticks[-1]

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(xticks, labels=[str(v) for v in xticks])
    ax.set_yticks(yticks, labels=[str(v) for v in yticks])

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    ax.tick_params(axis="x", which="both", top=True, labeltop=False)
    ax.tick_params(axis="y", which="both", right=True, labelright=False)

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\mu^2$")

    title = rf"E={energy:.1f}, $\alpha={alpha.value:.2f}^{{\circ}}$, n={res}, $\omega_{{\text{{pe}}}}/\omega_{{\text{{ce}}}}={ratio}$"
    ax.set_title(title)


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0, u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratios = [0.75, 1.5, 7.0]

    energy = 1.0 << u.MeV
    alpha = Angle(45, u.deg)
    resonance = -1
    freq_cutoff_params = (
        0.5,
        0.5,
        -1 + 1e-10,
        1e5,
    )  # force omega_lc to be zero and omega_uc a very large value

    X_min = 0.0
    X_max = 20.0
    X_npoints = 400
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    omega_npoints = 2000
    # =================================================

    for i, ratio in enumerate(plasma_over_gyro_ratios):
        mag_point = MagPoint(mlat_deg, l_shell)
        plasma_point = PlasmaPoint(mag_point, particles, ratio)
        wave_filter = NullFilter()
        cpdr = Cpdr(
            plasma_point, energy, alpha, resonance, freq_cutoff_params, wave_filter
        )

        # Geometric sequence between lower and upper cutoffs
        omega_range = u.Quantity(
            np.geomspace(
                cpdr.omega_lc.value,
                cpdr.omega_uc.value,
                num=omega_npoints,
                endpoint=True,
            ),
            unit=cpdr.omega_lc.unit,
        )

        print(f"Plasma: {particles}")
        print(f"Energy: {energy}")
        print(f"Pitch angle: {alpha}")
        print(f"Resonance: {resonance}")
        print(f"Electron plasma over gyro ratio: {ratio}")
        print(f"Electron gyrofrequency (abs): {np.abs(cpdr.plasma.gyro_freq[0]):.1f}")
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
        ) / np.sqrt(
            1 + (cpdr.plasma.gyro_freq[0] ** 2 / cpdr.plasma.plasma_freq[0] ** 2)
        )
        print(f"omega_lh: {omega_lh:.1f}")
        print(
            f"approximate omega_lh: {np.sqrt(np.abs(cpdr.plasma.gyro_freq[0]) * cpdr.plasma.gyro_freq[1]):.1f}"
        )
        print()

        print("Upper hybrid frequency from Kurth 2015")
        omega_uh = np.sqrt(
            cpdr.plasma.gyro_freq[0] ** 2 + cpdr.plasma.plasma_freq[0] ** 2
        )
        print(f"omega_uh: {omega_uh:.1f}")
        print(f"\n{'=' * 80}\n")

        # Init figure and axes
        fig, ax = plt.subplots()

        draw_vertical_lines(ax, cpdr, omega_lh.value, omega_uh.value)
        plot_cpdr_roots(fig, ax, cpdr, X_range, omega_range)
        plot_resonant_roots(ax, cpdr, X_range)
        format_figure(ax, energy, alpha, resonance, ratio)

        fig.tight_layout()
        fig.savefig(f"wave_dispersion_{i + 1}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
