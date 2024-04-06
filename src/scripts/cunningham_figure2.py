# This script reproduces the results from Figure 2 in Cunningham, 2023.
# To run use the `-p` option to pass the directory with the .dat
# file from Cunningham paper (it is expected that the files are name
# as "Figure2?.dat") and the `-f` option to select the figure that
# you want to reproduce (a, b, c or d).
# If you pass the optional `-s` argument the figure will be saved
# on disk in the current working directory as "Figure2[abcd].png",
# instead of displayed on screen and also a file named "Figure2[abcd].txt",
# will also be saved on disk. This contains our results, in a similar
# format as the one in the .dat files.
# Finally passing the `-o` argument will overlay Cunningham's results
# from the .dat file in our plots.
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.cpdr2 import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.gauss import Gaussian
from piran.magpoint import MagPoint
from piran.normalisation import (
    compute_cunningham_norm_factor,
    compute_glauert_norm_factor,
)
from piran.plasmapoint import PlasmaPoint


def plot_figure2(
    norm_ratios,
    plasma_over_gyro_ratio,
    omega_ratios,
    x_ticks,
    y_ticks,
    overlay,
    save,
    filestem,
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    x_lim_min = x_ticks[0]
    x_lim_max = x_ticks[-1]

    y_lim_min = y_ticks[0]
    y_lim_max = y_ticks[-1]

    ratio1x = [v[0] for v in norm_ratios[0]]
    ratio1y = [v[1] for v in norm_ratios[0]]
    plt.loglog(
        ratio1x, ratio1y, "k", label=rf"$\omega$/$\omega_{{ce}}$={omega_ratios[0]}"
    )

    ratio2x = [v[0] for v in norm_ratios[1]]
    ratio2y = [v[1] for v in norm_ratios[1]]
    plt.loglog(
        ratio2x, ratio2y, "r", label=rf"$\omega$/$\omega_{{ce}}$={omega_ratios[1]}"
    )

    if overlay is not None:
        overlay_ratio1x = overlay[:, 0]
        overlay_ratio1y = overlay[:, 1]
        plt.loglog(
            overlay_ratio1x,
            overlay_ratio1y,
            color="k",
            linestyle="--",
            alpha=0.6,
            label=rf"$\omega$/$\omega_{{ce}}$={omega_ratios[0]} from paper",
        )

        overlay_ratio2x = overlay[:, 2]
        overlay_ratio2y = overlay[:, 3]
        plt.loglog(
            overlay_ratio2x,
            overlay_ratio2y,
            color="r",
            linestyle="--",
            alpha=0.6,
            label=rf"$\omega$/$\omega_{{ce}}$={omega_ratios[1]} from paper",
        )

    # plt.minorticks_on()
    plt.xticks(x_ticks, [str(v) for v in x_ticks])
    plt.yticks(y_ticks, [str(v) for v in y_ticks])
    plt.tick_params("x", which="both", top=True, labeltop=False)
    plt.tick_params("y", which="both", right=True, labelright=False)
    # plt.yticks(np.arange(0.2, 1.0, 0.1), [], minor=True)
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(y_lim_min, y_lim_max)
    plt.xlabel(r"X ($\tan{\theta}$)")
    plt.ylabel("Ratio")
    plt.legend(loc="upper right")
    plt.title(
        rf"Ratio of normalizers for "
        rf"$\omega_{{pe}}$/$\omega_{{ce}}$={plasma_over_gyro_ratio}"
    )
    plt.tight_layout()

    if save:
        plt.savefig(f"{filestem}.png", dpi=150)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_2023_Figure2",
        description="Reproduce Figure 2 from Cunningham, 2023",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to directory with Cunningham's dat files.",
    )
    parser.add_argument(
        "-f",
        "--figure",
        choices=["a", "b", "c", "d"],
        required=True,
        help="Select figure to plot (a, b, c or d).",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=False,
        help="Pass this argument to save our results on disk, \
              both figure and dat file.",
    )
    parser.add_argument(
        "-o",
        "--overlay",
        action="store_true",
        default=False,
        help="Pass this argument to overlay Cunningham's results in our figures.",
    )
    args = parser.parse_args()

    if args.figure == "a" or args.figure == "b":
        X_min = 0.00 << u.dimensionless_unscaled
        X_max = 1.00 << u.dimensionless_unscaled
        X_npoints = 1000
        xticks = [0.01, 0.10, 1.00]
        yticks = [0.1, 1.0, 10.0, 100.0]
    elif args.figure == "c" or args.figure == "d":
        X_min = 0.00 << u.dimensionless_unscaled
        X_max = 5.67 << u.dimensionless_unscaled
        X_npoints = 5000
        xticks = [0.01, 0.10, 1.00, 10.00]
        yticks = [0.1, 1.0, 10.0, 100.0, 1000.0]

    if args.figure == "a" or args.figure == "c":
        plasma_over_gyro_ratio = 1.5
    elif args.figure == "b" or args.figure == "d":
        plasma_over_gyro_ratio = 10.0

    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    X_m = 0.0 << u.dimensionless_unscaled  # peak
    X_w = 0.577 << u.dimensionless_unscaled  # angular width
    wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5
    mag_point = MagPoint(mlat_deg, l_shell)

    particles = ("e", "p+")
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

    n_particles = len(particles)
    cpdr_sym = CpdrSymbolic(n_particles)

    cpdr = Cpdr(cpdr_sym, plasma_point)

    # Load data from Cunningham paper
    cunningham_dat_filepath = Path(args.path) / f"Figure2{args.figure}.dat"
    if not cunningham_dat_filepath.is_file():
        msg = f"Incorrect path for Cunningham's dat file: {cunningham_dat_filepath}"
        raise Exception(msg)

    cunningham_figure_data = np.loadtxt(
        cunningham_dat_filepath,
        dtype=np.float64,
        delimiter=None,
        comments=";",
    )
    X_range_cunningham = [
        u.Quantity(cunningham_figure_data[:, 0], unit=u.dimensionless_unscaled),
        u.Quantity(cunningham_figure_data[:, 2], unit=u.dimensionless_unscaled),
    ]

    norm_ratios = []
    omega_ratios = [0.125, 0.575]
    for i, omega_ratio in enumerate(omega_ratios):
        omega = (
            np.abs(cpdr.plasma.gyro_freq[0]) * omega_ratio
        )  # Electron gyrofrequency * ratio

        resonance_cone_angle = -cpdr.stix.P(omega) / cpdr.stix.S(omega)
        epsilon = 0.9999  # Glauert & Horne 2005 paragraph 23
        X_upper = min(X_max, epsilon * np.sqrt(resonance_cone_angle))

        # For Glauert's norm factor use a uniform distribution
        # between X_min and X_upper=min(X_max, epsilon*sqrt(-P/S)).
        X_range_glauert = X_range[X_range <= X_upper]

        # Glauert's norm factor
        norm_factor_glauert = compute_glauert_norm_factor(
            cpdr, omega, X_range_glauert, wave_norm_angle_dist, method="simpson"
        )

        # Cunningham's norm factors (numpy ndarray)
        norm_factors_cunningham = compute_cunningham_norm_factor(
            cpdr,
            omega,
            X_range_cunningham[i],
        )

        # Calculate integral of g(X) to normalise it.
        # Technically we should use left endpoint integration rule,
        # not simpson, as this is what Cunningham used in his paper.
        eval_gx = wave_norm_angle_dist.eval(X_range_glauert)
        integral_gx = simpson(eval_gx, x=X_range_glauert)

        # Calculate the ratio of equation (4) to equation (5)
        ratio = []
        for X, norm_factor_cunningham in zip(
            X_range_cunningham[i], norm_factors_cunningham
        ):
            ratio.append(
                (
                    X,
                    (1.0 / norm_factor_cunningham)
                    / (integral_gx / norm_factor_glauert),
                )
            )

        norm_ratios.append(ratio)

    if args.overlay:
        overlay = cunningham_figure_data
    else:
        overlay = None

    filestem = cunningham_dat_filepath.stem

    plot_figure2(
        norm_ratios,
        plasma_over_gyro_ratio,
        omega_ratios,
        xticks,
        yticks,
        overlay,
        args.save,
        filestem,
    )

    if args.save:
        # Save ratios in similar format as the dat files from the paper
        comment = ""
        comment += "Column 1 is tangent of the wave normal angle used in column 2\n"
        comment += (
            f"Column 2 is the ratio of equation 4 to equation 5 using the X "
            f"in column 1 when Xmax={X_max} and the ratio of the plasma "
            f"frequency to the unsigned electron gyrofrequency is "
            f"{plasma_over_gyro_ratio} and the ratio of the wave frequency "
            f"to the unsigned electron gyrofrequency is {omega_ratios[0]}\n"
        )
        comment += "Column 3 is tangent of the wave normal angle used in column 4\n"
        comment += (
            f"Column 4 is the ratio of equation 4 to equation 5 using the X "
            f"in column 3 when Xmax={X_max} and the ratio of the plasma "
            f"frequency to the unsigned electron gyrofrequency is "
            f"{plasma_over_gyro_ratio} and the ratio of the wave frequency "
            f"to the unsigned electron gyrofrequency is {omega_ratios[1]}"
        )

        ratios = np.empty_like(cunningham_figure_data)
        for i in range(ratios.shape[0]):
            ratios[i, 0] = norm_ratios[0][i][0]
            ratios[i, 1] = norm_ratios[0][i][1]
            ratios[i, 2] = norm_ratios[1][i][0]
            ratios[i, 3] = norm_ratios[1][i][1]

        np.savetxt(
            f"{filestem}.txt",
            ratios,
            fmt=["%.9f", "%.8e", "%.9f", "%.8e"],
            delimiter=",",
            newline="\n",
            header=comment,
            comments="# ",
        )


if __name__ == "__main__":
    main()
