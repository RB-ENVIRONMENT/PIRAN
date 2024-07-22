# This script reproduces the results from Figure 1a in Cunningham, 2023.
#
# To run use the `--path` option to pass the filepath of the .dat
# file from Cunningham's paper (ex. Figure1a.dat).
#
# If you pass the optional `-s` argument the figure will be saved
# on disk in the current working directory, e.g. as "PIRAN_Figure1a.png".
#
# e.g.
# python src/scripts/cunningham_figure1a.py \
#     --path "PATH/TO/Figure1a.dat"
import argparse
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint

script_version = "1.0.0"


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
    yticks = [0.1, 1.0, 10.0, 100.0]

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
    ax.set_ylabel(r"$X_{\text{rc}} = \tan(\theta_{\text{rc}})$")

    ax.set_title(
        "Resonance cone angle for various plasma parameters\nCold proton plasma"
    )

    fig.tight_layout()


def plot_figure1a(
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
    ax.legend(loc="lower left")


def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_2023_Figure1a",
        description="Reproduce Figure 1a from Cunningham, 2023",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to Cunningham's dat file for Figure1a.",
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

        # Find the indices where the value is -1 and delete them
        minus_one_ids = np.where(yy == -1.0)[0]

        xx = np.delete(xx, minus_one_ids)
        yy = np.delete(yy, minus_one_ids)

        plot_figure1a(
            ax,
            xx,
            yy,
            color[ii],
            "-",
            0.4,
            rf"C2023 $\omega_{{\text{{pe}}}}$/$\omega_{{\text{{ce}}}}$={ratio[ii]}",
        )

    # Plot values calculated using PIRAN
    for ii in range(len(ratio)):
        xx = cunningham_figure_data[:, 0]

        particles = ("e", "p+")
        cpdr_sym = CpdrSymbolic(len(particles))

        mlat_deg = Angle(0, u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)
        plasma_point = PlasmaPoint(mag_point, particles, ratio[ii])

        cpdr = Cpdr(cpdr_sym, plasma_point)

        yy = np.zeros(xx.shape)
        for jj in range(xx.shape[0]):
            omega = np.abs(cpdr.plasma.gyro_freq[0]) * xx[jj]
            resonance_cone_angle = -cpdr.stix.P(omega) / cpdr.stix.S(omega)
            X_rc = np.sqrt(resonance_cone_angle)
            yy[jj] = X_rc

        plot_figure1a(
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
        plt.savefig("PIRAN_Figure1a.png", dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
