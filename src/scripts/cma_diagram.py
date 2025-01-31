import math

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib.ticker import LogFormatterMathtext

from piran.cpdr import Cpdr
from piran.helpers import get_real_and_positive_roots
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


class Cpdr2(Cpdr):
    def solve_resonant2(
        self,
        X,
    ):
        # Solve modified CPDR to obtain omega roots for given X
        omega_l = self._resonant_roots_in_omega(X.value)

        # Categorise roots
        # Keep only real, positive and within bounds
        valid_omega_l = get_real_and_positive_roots(omega_l)
        valid_omega_l = [
            x for x in valid_omega_l if self.omega_lc.value <= x <= self.omega_uc.value
        ]

        return valid_omega_l

    def solve_cpdr2(
        self,
        omega,
        X,
    ):
        """
        Similar to solve_cpdr() but we return all real and
        positive roots, even if we have more than one.
        """
        # Solve unmodified CPDR to obtain k roots for given X, omega
        k_l = self._roots_in_k(X, omega)

        # Keep only real and positive roots
        valid_k_l = get_real_and_positive_roots(k_l)

        return valid_k_l

    def find_resonant_parallel_wavenumber2(
        self,
        X,
        omega,
        k,
        rel_tol=1e-05,
    ):
        """
        Similar to find_resonant_parallel_wavenumber() but we do not
        throw an error if neither k_par, nor -k_par are roots.
        """
        if np.isnan(k):
            return np.nan << u.rad / u.m

        psi = np.arctan(X)
        k_par = k * np.cos(psi)
        gyrofreq = self.plasma.gyro_freq[0]
        reson = self.resonance
        v_par = self.v_par
        gamma = self.gamma

        # Rearrange resonance condition to produce scaled `1 = ...` equation
        result1 = ((reson * gyrofreq / gamma) + (k_par * v_par)) / omega  # [0, pi/2)
        result2 = ((reson * gyrofreq / gamma) - (k_par * v_par)) / omega  # (pi/2, pi]

        # Compare to unity to determine signedness of k_par
        k_par_is_pos = math.isclose(result1.value, 1.0, rel_tol=rel_tol)
        k_par_is_neg = math.isclose(result2.value, 1.0, rel_tol=rel_tol)

        if k_par_is_pos and not k_par_is_neg:
            # only positive k_par is root
            return k_par, True
        elif k_par_is_neg and not k_par_is_pos:
            # only negative k_par is root
            return -k_par, True
        elif k_par_is_pos and k_par_is_neg:
            raise ValueError("Both are roots")
        else:
            return k_par, False


def format_figure(fig, ax, energy, alpha, res, ratio):
    xticks = [10**i for i in range(1, 7)]
    yticks = [10**i for i in range(-2, 9)]

    xlim_min = xticks[0]
    xlim_max = xticks[-1]

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

    title = "CMA diagram\n"
    title += rf"{energy:.1f}, $\alpha={alpha:.2f}$, n={res}, ratio={ratio}"
    ax.set_title(title)

    fig.tight_layout()


def draw_vertical_lines(ax, cpdr, omega_lh, omega_uh):
    # Draw omega_c and omega_p lines (electron)
    omega_c_abs = np.abs(cpdr.plasma.gyro_freq[0]).value
    omega_p = cpdr.plasma.plasma_freq[0].value
    ax.axvline(x=omega_p, color="k", linestyle="--")
    ax.axvline(x=omega_c_abs, color="k", linestyle="--")

    # Draw omega_L=0 and omega_R=0 lines
    omega_L0 = (np.sqrt(omega_c_abs**2 + 4 * omega_p**2) - omega_c_abs) / 2
    omega_R0 = (np.sqrt(omega_c_abs**2 + 4 * omega_p**2) + omega_c_abs) / 2
    ax.axvline(x=omega_L0, color="k", linestyle=":")
    ax.axvline(x=omega_R0, color="k", linestyle=":")

    # Draw proton gyrofrequency
    omega_pp = cpdr.plasma.gyro_freq[1].value
    ax.axvline(x=omega_pp, color="k", linestyle="-.")

    # Draw lower and upper hybrid lines
    ax.axvline(x=omega_lh, color="k", linestyle="-.")
    ax.axvline(x=omega_uh, color="k", linestyle="-.")

    # Annotate
    ax.text(omega_c_abs, 2 * 10**3, r"$\omega_{ce}$")
    ax.text(omega_p, 8 * 10**2, r"$\omega_{pe}$")
    ax.text(omega_L0, 4 * 10**2, r"$\omega_{L=0}$")
    ax.text(omega_R0, 5 * 10**3, r"$\omega_{R=0}$")
    ax.text(omega_pp, 2.5 * 10**3, r"$\omega_{pp}$")
    ax.text(omega_lh, 2.5 * 10**3, r"$\omega_{LH}$")
    ax.text(omega_uh, 2.5 * 10**3, r"$\omega_{UH}$")


def plot_resonant_roots(ax, cpdr, X_range):
    xaxis_values = []
    yaxis_values = []

    # Find resonant omegas and then k
    for X in X_range:
        valid_omegas = cpdr.solve_resonant2(X)
        if len(valid_omegas) == 0:
            continue

        for omega in valid_omegas:
            k_list = cpdr.solve_cpdr2(omega, X.value)

            for k in k_list:
                _, is_resonant = cpdr.find_resonant_parallel_wavenumber2(
                    X, omega << (u.rad / u.s), k << (u.rad / u.m)
                )
                if is_resonant:
                    mu = const.c * abs(k) / omega

                    xaxis_values.append(omega)
                    yaxis_values.append((mu**2).value)

    ax.scatter(
        xaxis_values,
        yaxis_values,
        marker=".",
        s=14,
        c="red",
    )


def plot_cpdr_roots(fig, ax, cpdr, X_range, omega_range):
    xaxis_values = []
    yaxis_values = []
    color_values = []
    for omega in omega_range:
        for X in X_range:
            k_list = cpdr.solve_cpdr2(omega, X.value)

            for k in k_list:
                mu = const.c * abs(k) / omega

                xaxis_values.append(omega)
                yaxis_values.append((mu**2).value)
                color_values.append(X.value)

    im = ax.scatter(
        xaxis_values,
        yaxis_values,
        marker=".",
        s=8,
        c=color_values,
        cmap="viridis",
        vmin=X_range[0].value,
        vmax=X_range[-1].value,
        # alpha=0.4,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel(r"Wave normal angle $X=\tan(\psi)$")


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0, u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 0.75

    energy = 1.0 << u.MeV
    alpha = Angle(50, u.deg)
    resonance = -5
    freq_cutoff_params = (0.5, 0.5, -0.9999, 3.999)

    X_min = 0.0
    X_max = 60.0
    X_npoints = 60
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    omega_npoints = 150
    # =================================================

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr = Cpdr2(plasma_point, energy, alpha, resonance, freq_cutoff_params)

    # Geometric sequence between lower and upper cutoffs
    omega_range = np.geomspace(
        cpdr.omega_lc.value, cpdr.omega_uc.value, num=omega_npoints, endpoint=True
    )

    print(f"Plasma: {particles}")
    print(f"Energy: {energy}")
    print(f"Pitch angle: {alpha}")
    print(f"Resonance: {resonance}")
    print(f"Electron plasma over gyro ratio: {plasma_over_gyro_ratio}")
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
    ) / np.sqrt(1 + (cpdr.plasma.gyro_freq[0] ** 2 / cpdr.plasma.plasma_freq[0] ** 2))
    print(f"omega_lh: {omega_lh:.1f}")
    print(
        f"approximate omega_lh: {np.sqrt(np.abs(cpdr.plasma.gyro_freq[0]) * cpdr.plasma.gyro_freq[1]):.1f}"
    )
    print()

    print("Upper hybrid frequency from Kurth 2015")
    omega_uh = np.sqrt(cpdr.plasma.gyro_freq[0] ** 2 + cpdr.plasma.plasma_freq[0] ** 2)
    print(f"omega_uh: {omega_uh:.1f}")
    print()

    # Init figure and axes
    fig, ax = plt.subplots()

    draw_vertical_lines(ax, cpdr, omega_lh.value, omega_uh.value)
    plot_cpdr_roots(fig, ax, cpdr, X_range, omega_range)
    plot_resonant_roots(ax, cpdr, X_range)
    format_figure(fig, ax, energy, alpha, resonance, plasma_over_gyro_ratio)

    plt.show()
    # plt.savefig("cma_diagram.png", dpi=300)


if __name__ == "__main__":
    main()
