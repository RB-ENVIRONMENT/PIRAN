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
    Keep all solutions.
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
    # Draw omega_c and omega_p lines (electron)
    omega_c_abs = np.abs(cpdr.plasma.gyro_freq[0]).value
    omega_p = cpdr.plasma.plasma_freq[0].value
    ax.axvline(x=omega_p, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(x=omega_c_abs, color="k", linestyle="--", linewidth=0.8)

    # Draw omega_L=0 and omega_R=0 lines
    omega_L0 = (np.sqrt(omega_c_abs**2 + 4 * omega_p**2) - omega_c_abs) / 2
    omega_R0 = (np.sqrt(omega_c_abs**2 + 4 * omega_p**2) + omega_c_abs) / 2
    ax.axvline(x=omega_L0, color="k", linestyle=":", linewidth=0.8)
    ax.axvline(x=omega_R0, color="k", linestyle=":", linewidth=0.8)

    # Draw proton gyrofrequency
    omega_pp = cpdr.plasma.gyro_freq[1].value
    ax.axvline(x=omega_pp, color="k", linestyle="-.", linewidth=0.8)

    # Draw lower and upper hybrid lines
    ax.axvline(x=omega_lh, color="k", linestyle="-.", linewidth=0.8)
    ax.axvline(x=omega_uh, color="k", linestyle="-.", linewidth=0.8)

    # Annotate
    ax.text(omega_c_abs, 6.0 * 10**4, r"$\omega_{ce}$")
    ax.text(omega_p, 1.0 * 10**4, r"$\omega_{pe}$")
    ax.text(omega_L0, 2.5 * 10**3, r"$\omega_{L=0}$")
    ax.text(omega_R0, 2.5 * 10**3, r"$\omega_{R=0}$")
    ax.text(omega_pp, 2.5 * 10**3, r"$\omega_{pp}$")
    ax.text(omega_lh, 2.5 * 10**3, r"$\omega_{LH}$")
    ax.text(omega_uh, 2.0 * 10**5, r"$\omega_{UH}$")


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
        s=6,
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
    xticks = [10**i for i in range(1, 6)]
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

    title = rf"E={energy:.1f}, $\alpha={alpha:.2f}$, n={res}, $\omega_{{\text{{pe}}}}/\omega_{{\text{{ce}}}}={ratio}$"
    ax.set_title(title)


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0, u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratios = [0.75]  # [0.75, 7.0]

    energy = 1.0 << u.MeV
    alpha = Angle(45, u.deg)
    resonance = -1
    freq_cutoff_params = (0.5, 0.5, -1 + 1e-10, 1e5)  # force omega_lc to be zero and omega_uc a very large value

    X_min = 0.0
    X_max = 20.0
    X_npoints = 200
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    omega_npoints = 800
    # =================================================

    for i, ratio in enumerate(plasma_over_gyro_ratios):
        mag_point = MagPoint(mlat_deg, l_shell)
        plasma_point = PlasmaPoint(mag_point, particles, ratio)
        wave_filter = NullFilter()
        cpdr = Cpdr(plasma_point, energy, alpha, resonance, freq_cutoff_params, wave_filter)

        # Geometric sequence between lower and upper cutoffs
        omega_range = u.Quantity(np.geomspace(
            cpdr.omega_lc.value, cpdr.omega_uc.value, num=omega_npoints, endpoint=True
        ), unit=cpdr.omega_lc.unit)

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
        ) / np.sqrt(1 + (cpdr.plasma.gyro_freq[0] ** 2 / cpdr.plasma.plasma_freq[0] ** 2))
        print(f"omega_lh: {omega_lh:.1f}")
        print(
            f"approximate omega_lh: {np.sqrt(np.abs(cpdr.plasma.gyro_freq[0]) * cpdr.plasma.gyro_freq[1]):.1f}"
        )
        print()

        print("Upper hybrid frequency from Kurth 2015")
        omega_uh = np.sqrt(cpdr.plasma.gyro_freq[0] ** 2 + cpdr.plasma.plasma_freq[0] ** 2)
        print(f"omega_uh: {omega_uh:.1f}")
        print(f"\n{'=' * 80}\n")

        # Init figure and axes
        fig, ax = plt.subplots()

        draw_vertical_lines(ax, cpdr, omega_lh.value, omega_uh.value)
        plot_cpdr_roots(fig, ax, cpdr, X_range, omega_range)
        plot_resonant_roots(ax, cpdr, X_range)
        format_figure(ax, energy, alpha, resonance, ratio)

        fig.tight_layout()
        fig.savefig(f"cma_diagram_{i + 1}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
