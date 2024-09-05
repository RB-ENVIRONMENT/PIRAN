# This script reproduces the results from Figure 5b in Cunningham, 2023.
#
# To run use the `--path` option to pass the filepath of the .dat
# file from Cunningham's paper (Figure5b.dat).
# The file should contain results for the methods proposed by:
#
# - Glauert and Horne,
# - Cunningham,
#
# for plasma over gyrofrequencies ratios of both 1.5 and 10.
#
# Use `--c1`, `--c2`, `--g1` and `--g2` to pass the directories where
# the `results*.json` files produced by the `diffusion_coefficients.py`
# script are located, for Cunningham with 1.5 ratio, Cunningham with 10 ratio,
# Glauert with 1.5 ratio and Glauert with 10 ratio respectively.
#
# If you pass the optional `-s` argument the figure will be saved
# on disk in the current working directory as "PIRAN_Figure5b.png".
#
# e.g.
# python src/scripts/cunningham_figure5b.py \
#     --path "PATH/TO/Figure5b.dat" \
#     --c1 "PATH/TO/cunningham_1.5" \
#     --c2 "PATH/TO/cunningham_10.0" \
#     --g1 "PATH/TO/glauert_1.5" \
#     --g2 "PATH/TO/glauert_10.0"
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from piran.diffusion import UNIT_DIFF, get_energy_diffusion_coefficient


def calc_DEE_over_E_squared(pathname):
    """
    `pathname` must contain the `results*.json` files produced by the
    `diffusion_coefficients.py` script.
    """
    pitch_angle = []
    Dee_over_e_squared = []

    for file in Path(pathname).glob("results*.json"):
        with open(file, "r") as f:
            results = json.load(f)

        Dpp = results["Dpp"] << UNIT_DIFF
        alpha = results["pitch_angle"] << u.deg
        rest_mass_energy_J = results["rest_mass_energy_Joule"] << u.J
        rel_kin_energy_J = (results["rel_kin_energy_MeV"] << u.MeV).to(u.J)

        Dee = get_energy_diffusion_coefficient(
            rel_kin_energy_J, rest_mass_energy_J, Dpp
        )

        pitch_angle.append(alpha)
        Dee_over_e_squared.append(Dee / rel_kin_energy_J.value**2)

    # Sort by pitch angle
    sorted_vals = sorted(zip(pitch_angle, Dee_over_e_squared), key=lambda z: z[0])
    xx = [z[0].value for z in sorted_vals]
    yy = [z[1].value for z in sorted_vals]

    return (xx, yy)


def plot_figure5b(
    piran_cunn_1,
    piran_cunn_2,
    piran_glau_1,
    piran_glau_2,
    cunningham_figure_data,
    save,
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    xticks = list(range(0, 91, 15))
    yticks = [10 ** (-n) for n in range(6, 1, -1)]

    x_lim_min = xticks[0]
    x_lim_max = xticks[-1]

    y_lim_min = yticks[0]
    y_lim_max = yticks[-1]

    # PIRAN frequency ratio 1.5 (Glauert & Horne)
    if piran_glau_1 is not None:
        xval = piran_glau_1[0]
        yval = piran_glau_1[1]
        plt.semilogy(
            xval,
            yval,
            color="k",
            linestyle="-",
            alpha=1.0,
            label="PIRAN (Glauert & Horne)",
        )

    # PIRAN frequency ratio 10.0 (Glauert & Horne)
    if piran_glau_2 is not None:
        xval = piran_glau_2[0]
        yval = piran_glau_2[1]
        plt.semilogy(
            xval,
            yval,
            color="k",
            linestyle="-",
            alpha=1.0,
        )

    # PIRAN frequency ratio 1.5 (Cunningham)
    if piran_cunn_1 is not None:
        xval = piran_cunn_1[0]
        yval = piran_cunn_1[1]
        plt.semilogy(
            xval,
            yval,
            color="r",
            linestyle="-",
            alpha=1.0,
            label="PIRAN (Cunningham)",
        )

    # PIRAN frequency ratio 10.0 (Cunningham)
    if piran_cunn_2 is not None:
        xval = piran_cunn_2[0]
        yval = piran_cunn_2[1]
        plt.semilogy(
            xval,
            yval,
            color="r",
            linestyle="-",
            alpha=1.0,
        )

    if cunningham_figure_data is not None:
        overlay_x = cunningham_figure_data[:, 0]

        # Glauert and Horne frequency ratio 1.5
        overlay_y_glau1 = cunningham_figure_data[:, 1]
        plt.semilogy(
            overlay_x,
            overlay_y_glau1,
            color="k",
            linestyle="-",
            alpha=0.4,
            label="C2023 v2.0.0 (Glauert & Horne)",
        )

        # Cunningham frequency ratio 1.5
        overlay_y_cunn1 = cunningham_figure_data[:, 2]
        plt.semilogy(
            overlay_x,
            overlay_y_cunn1,
            color="r",
            linestyle="-",
            alpha=0.4,
            label="C2023 v2.0.0 (Cunningham)",
        )

        # Glauert and Horne frequency ratio 10.0
        overlay_y_glau2 = cunningham_figure_data[:, 3]
        plt.semilogy(
            overlay_x,
            overlay_y_glau2,
            color="k",
            linestyle="-",
            alpha=0.4,
        )

        # Cunningham frequency ratio 10.0
        overlay_y_cunn2 = cunningham_figure_data[:, 4]
        plt.semilogy(
            overlay_x,
            overlay_y_cunn2,
            color="r",
            linestyle="-",
            alpha=0.4,
        )

    plt.text(10, 10 ** (-3.5), r"$\omega_{\text{pe}}/\omega_{\text{ce}}=1.5$")
    plt.text(60, 10 ** (-5.5), r"$\omega_{\text{pe}}/\omega_{\text{ce}}=10$")

    plt.minorticks_on()
    plt.xticks(xticks, [str(v) for v in xticks])
    # plt.yticks(yticks, [str(v) for v in yticks])
    plt.tick_params("x", which="both", top=True, labeltop=False)
    plt.tick_params("y", which="both", right=True, labelright=False)
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(y_lim_min, y_lim_max)
    plt.xlabel("Local pitch angle (degrees)")
    plt.ylabel(r"$\text{D}_{\text{EE}} / \text{E}^2$")
    plt.legend(loc="lower left")
    plt.title("KE=10keV Harmonics [-5, 5]")
    plt.tight_layout()

    if save:
        plt.savefig("PIRAN_Figure5b.png", dpi=150)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_2023_Figure5b",
        description="Reproduce Figure 5b from Cunningham, 2023",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        help="Path to Cunningham's dat file for Figure5b.",
    )
    parser.add_argument(
        "--c1",
        default=None,
        help="Path to directory with results*.json files for Cunningham 1.5 ratio",
    )
    parser.add_argument(
        "--c2",
        default=None,
        help="Path to directory with results*.json files for Cunningham 10 ratio",
    )
    parser.add_argument(
        "--g1",
        default=None,
        help="Path to directory with results*.json files for Glauert 1.5 ratio",
    )
    parser.add_argument(
        "--g2",
        default=None,
        help="Path to directory with results*.json files for Glauert 10 ratio",
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

    # Our results
    if args.c1 is not None:
        piran_cunn_1 = calc_DEE_over_E_squared(args.c1)  # Cunningham ratio 1.5
    else:
        piran_cunn_1 = None

    if args.c2 is not None:
        piran_cunn_2 = calc_DEE_over_E_squared(args.c2)  # Cunningham ratio 10
    else:
        piran_cunn_2 = None

    if args.g1 is not None:
        piran_glau_1 = calc_DEE_over_E_squared(args.g1)  # Glauert & Horne ratio 1.5
    else:
        piran_glau_1 = None

    if args.g2 is not None:
        piran_glau_2 = calc_DEE_over_E_squared(args.g2)  # Glauert & Horne ratio 10
    else:
        piran_glau_2 = None

    plot_figure5b(
        piran_cunn_1,
        piran_cunn_2,
        piran_glau_1,
        piran_glau_2,
        cunningham_figure_data,
        args.save,
    )


if __name__ == "__main__":
    main()
