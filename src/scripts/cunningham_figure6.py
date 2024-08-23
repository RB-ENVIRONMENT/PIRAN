# This script reproduces the results from Figure 6 in Cunningham, 2023.
#
# To run use the `--fig` option to pass the figure id ([a-f])
# and the `--path` option to pass the filepath of the .dat
# file from Cunningham's paper (ex. Figure6b.dat).
# The file should contain results for the methods proposed by:
#
# - Glauert and Horne,
# - Cunningham,
#
# for Figures a, c and e and:
#
# - Glauert and Horne,
# - Cunningham,
# - Kennel and Engelmann,
# - Glauert and Horne, using 300 as the maximum index of refraction,
#
# # for Figures b, d and f.
#
# Use `-c` and `-g` to pass the directories where the
# `results*.json` files produced by the `diffusion_coefficients.py`
# script are located, for Cunningham and Glauert respectively.
#
# If you pass the optional `-s` argument the figure will be saved
# on disk in the current working directory, e.g. as "PIRAN_Figure6b.png".
#
# e.g.
# python src/scripts/cunningham_figure6.py \
#     --fig b
#     --path "PATH/TO/Figure6b.dat" \
#     -c "PATH/TO/fig6b_cunn" \
#     -g "PATH/TO/fig6b_glau"
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext

from piran.diffusion import get_diffusion_coefficients


def calc_Daa_over_p_squared(pathname):
    """
    `pathname` must contain the `results*.json` files produced by the
    `diffusion_coefficients.py` script.
    """
    energy = []
    Daa_over_p_squared = []

    for file in Path(pathname).glob("results*.json"):
        with open(file, "r") as f:
            results = json.load(f)

        X_range = np.array(results["X_range"])
        DnXaa = results["DnXaa"]
        momentum = results["momentum"]

        Daa = 0.0
        # Loop over resonances (n)
        for DnXaa_this_res in DnXaa:
            integral = get_diffusion_coefficients(X_range, np.array(DnXaa_this_res))
            Daa += integral

        energy.append(results["rel_kin_energy_MeV"])
        Daa_over_p_squared.append(Daa / momentum**2)

    # Sort by energy
    sorted_vals = sorted(zip(energy, Daa_over_p_squared), key=lambda z: z[0])
    xx = [z[0] for z in sorted_vals]
    yy = [z[1] for z in sorted_vals]

    return (np.array(xx), np.array(yy))


def format_figure(fig, ax, ratio, Xmax):
    xticks = [0.001, 0.010, 0.100, 1.000]
    yticks = [10 ** (n) for n in range(-8, 1, 2)]

    x_lim_min = xticks[0]
    x_lim_max = xticks[-1] + 1.0

    y_lim_min = yticks[0]
    y_lim_max = yticks[-1]

    ax.set_xticks(xticks, labels=[str(v) for v in xticks])
    ax.set_yticks(yticks, labels=[str(v) for v in yticks])

    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:g}".format(y)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    ax.tick_params("x", which="both", top=True, labeltop=False)
    ax.tick_params("y", which="both", right=True, labelright=False)

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(r"$\text{D}_{\alpha\alpha} / \text{p}^2$")

    title_omega = rf"$\omega_{{\text{{pe}}}}/\omega_{{\text{{ce}}}}={ratio}$"
    title_alpha = r"$\alpha_{\text{eq}}=0.125$"
    title_Xmax = rf"$X_{{\text{{max}}}}={Xmax}$"
    ax.set_title(f"{title_omega}, {title_alpha}, {title_Xmax}")

    ax.legend(loc="lower left")

    fig.tight_layout()


def plot_figure(
    ax,
    xx,
    yy,
    color,
    linestyle,
    marker,
    alpha,
    label,
):
    ax.loglog(
        xx,
        yy,
        color=color,
        linestyle=linestyle,
        marker=marker,
        alpha=alpha,
        label=label,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_2023_Figure6",
        description="Reproduce Figure 6 from Cunningham, 2023",
    )
    parser.add_argument(
        "-f",
        "--fig",
        required=True,
        choices=["a", "b", "c", "d", "e", "f"],
        help="Figure identifier.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        help="Path to Cunningham's dat file for Figure6.",
    )
    parser.add_argument(
        "-c",
        "--cunn",
        default=None,
        help="Path to directory with results*.json files for Cunningham.",
    )
    parser.add_argument(
        "-g",
        "--glau",
        default=None,
        help="Path to directory with results*.json files for Glauert.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=False,
        help="Pass this argument to save the figure on disk.",
    )
    args = parser.parse_args()

    if args.fig in ["a", "b"]:
        ratio = 0.75
    elif args.fig in ["c", "d"]:
        ratio = 1.5
    elif args.fig in ["e", "f"]:
        ratio = 10

    if args.fig in ["a", "c", "e"]:
        Xmax = 1.0
    elif args.fig in ["b", "d", "f"]:
        Xmax = 5.67

    # Load data from Cunningham paper
    if args.path is not None:
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
    else:
        cunningham_figure_data = None

    # Init figure and axes
    fig, ax = plt.subplots()

    # In the following, we remove very small values so that the figures
    # look almost the same as Cunningham's.
    tol = 10 ** (-12)

    # Plot Cunningham's data
    if cunningham_figure_data is not None:
        # Paper Glauert & Horne
        xx = cunningham_figure_data[:, 0]
        yy = cunningham_figure_data[:, 1]
        ids = np.where(yy < tol)[0]
        xx = np.delete(xx, ids)
        yy = np.delete(yy, ids)
        plot_figure(ax, xx, yy, "k", "-", "", 0.4, "Glauert & Horne")

        # Paper Cunningham
        xx = cunningham_figure_data[:, 0]
        yy = cunningham_figure_data[:, 2]
        ids = np.where(yy < tol)[0]
        xx = np.delete(xx, ids)
        yy = np.delete(yy, ids)
        plot_figure(ax, xx, yy, "r", "-", "", 0.4, "Cunningham")

        if args.fig in ["b", "d", "f"]:
            # Paper Kennel & Engelmann
            xx = cunningham_figure_data[:, 0]
            yy = cunningham_figure_data[:, 3]
            ids = np.where(yy < tol)[0]
            xx = np.delete(xx, ids)
            yy = np.delete(yy, ids)
            plot_figure(ax, xx, yy, "b", "-", "", 0.4, "Kennel & Engelmann")

            # Paper Glauert & Horne n_max=300
            xx = cunningham_figure_data[:, 0]
            yy = cunningham_figure_data[:, 4]
            ids = np.where(yy < tol)[0]
            xx = np.delete(xx, ids)
            yy = np.delete(yy, ids)
            plot_figure(
                ax, xx, yy, "k", "", "+", 0.4, r"Glauert & Horne $n_{\text{max}}=300$"
            )

    # Plot our data
    # PIRAN Glauert
    if args.glau is not None:
        xx, yy = calc_Daa_over_p_squared(args.glau)
        ids = np.where(yy < tol)[0]
        xx = np.delete(xx, ids)
        yy = np.delete(yy, ids)
        plot_figure(ax, xx, yy, "k", "-", "", 1.0, "PIRAN Glauert & Horne")

    # PIRAN Cunningham
    if args.cunn is not None:
        xx, yy = calc_Daa_over_p_squared(args.cunn)
        ids = np.where(yy < tol)[0]
        xx = np.delete(xx, ids)
        yy = np.delete(yy, ids)
        plot_figure(ax, xx, yy, "r", "-", "", 1.0, "PIRAN Cunningham")

    # Format the figure, add legend, labels etc.
    format_figure(fig, ax, ratio, Xmax)

    if args.save:
        plt.savefig(f"PIRAN_Figure6{args.fig}.png", dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
