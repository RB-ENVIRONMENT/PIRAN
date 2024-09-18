# This script reproduces the results from Figure 1b in Cunningham, 2023.
#
# To run use the `--path` option to pass the filepath of the .dat
# file from Cunningham's paper (ex. Figure1b.dat).
#
# If you pass the optional `-s` argument the figure will be saved
# on disk in the current working directory, e.g. as "PIRAN_Figure1b.png".
#
# e.g.
# python src/scripts/cunningham_figure1b.py \
#     --path "PATH/TO/Figure1b.dat"
import argparse
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint

script_version = "1.0.1"


def format_figure(fig, ax):
    ax.annotate(
        f"piran: {metadata.version('piran')}\nscript: {script_version}",
        xy=(0.0, 0.0),
        xycoords="figure fraction",
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=8,
    )

    xticks = [0.01, 0.10, 1.00]
    yticks = [10, 100, 1000, 10000]

    xlim_min = xticks[0]
    xlim_max = xticks[-1]

    ylim_min = yticks[0]
    ylim_max = yticks[-1]

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(xticks, labels=[str(v) for v in xticks])
    ax.set_yticks(yticks, labels=[str(v) for v in yticks])

    ax.tick_params(axis="x", which="both", top=True, labeltop=False)
    ax.tick_params(axis="y", which="both", right=True, labelright=False)

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    ax.set_xlabel(r"Normalized frequency ($\omega / \omega_{\text{ce}}$)")
    ax.set_ylabel("Index of refraction")

    title = r"Index of refraction for $X_{\text{max}}=\min(5.67, 0.9999X_{\text{rc}})$"
    title += "\nCold proton plasma"
    ax.set_title(title)

    fig.tight_layout()


def plot_figure1b(
    ax,
    xx,
    yy,
    color,
    linestyle,
    alpha,
    label=None,
):
    ax.plot(
        xx,
        yy,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )
    ax.legend(loc="upper left")


def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_2023_Figure1b",
        description="Reproduce Figure 1b from Cunningham, 2023",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to Cunningham's dat file for Figure1b.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=False,
        help="Pass this argument to save the figure on disk.",
    )
    args = parser.parse_args()

    # Load data from Cunningham paper
    cunningham_dat_filepath = Path(args.path)
    if not cunningham_dat_filepath.is_file():
        msg = f"Incorrect path for Cunningham's dat file: {cunningham_dat_filepath}"
        raise Exception(msg)

    cunningham_figure_data = np.loadtxt(
        cunningham_dat_filepath,
        dtype=np.float64,
        delimiter=None,
        comments=";",
    )

    ratio = [0.75, 1.5, 10.0]
    color = ["r", "k", "b"]

    # Init figure and axes
    fig, ax = plt.subplots()

    # Plot Cunningham's data
    for ii in range(len(ratio)):
        xx = cunningham_figure_data[:, 0]
        yy = cunningham_figure_data[:, ii + 1]

        # Find the indices where the value is 0.0 and delete them
        zeros_ids = np.where(yy == 0.0)[0]

        xx = np.delete(xx, zeros_ids)
        yy = np.delete(yy, zeros_ids)

        plot_figure1b(
            ax,
            xx,
            yy,
            color[ii],
            "-",
            0.4,
            rf"C2023 $\omega_{{\text{{pe}}}}$/$\omega_{{\text{{ce}}}}$={ratio[ii]}",
        )

    # Plot values calculated using PIRAN
    # Start by defining const parameters
    xx = cunningham_figure_data[:, 0]

    particles = ("e", "p+")
    cpdr_sym = CpdrSymbolic(len(particles))

    mlat_deg = Angle(0, u.deg)
    l_shell = 4.5
    mag_point = MagPoint(mlat_deg, l_shell)

    # Loop over differing ratios
    for ii in range(len(ratio)):
        plasma_point = PlasmaPoint(mag_point, particles, ratio[ii])

        cpdr = Cpdr(cpdr_sym, plasma_point)

        yy = np.zeros(xx.shape)
        for jj in range(xx.shape[0]):
            omega = np.abs(cpdr.plasma.gyro_freq[0]) * xx[jj]
            resonance_cone_angle = -cpdr.stix.P(omega) / cpdr.stix.S(omega)
            X_rc = np.sqrt(resonance_cone_angle).value

            X_max = min(5.67, 0.9999 * X_rc) << u.dimensionless_unscaled
            k = cpdr.solve_cpdr(omega, X_max)
            filtered_k = cpdr.filter(X_max, omega, k)
            mu = const.c * filtered_k / omega
            yy[jj] = mu.value

        plot_figure1b(
            ax,
            xx,
            yy,
            color[ii],
            "-",
            1.0,
            rf"PIRAN $\omega_{{\text{{pe}}}}$/$\omega_{{\text{{ce}}}}$={ratio[ii]}",
        )

    # Format the figure, add legend, labels etc.
    format_figure(fig, ax)

    if args.save:
        plt.savefig("PIRAN_Figure1b.png", dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
