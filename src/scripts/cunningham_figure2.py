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
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magfield import MagField
from piran.normalisation import (
    compute_cunningham_normalisation_factor,
    compute_glauert_normalisation_factor,
    solve_dispersion_relation,
)
from piran.particles import Particles, PiranParticle
from piran.resonance import replace_cpdr_symbols


def calculate_ratio(
    dispersion,
    omega_c,
    omega_p,
    omega_ratio,
    X_range_glauert,
    X_range_cunningham,
):
    omega = abs(omega_c[0]) * omega_ratio  # Electron gyrofrequency * ratio

    glauert_root_pairs = solve_dispersion_relation(
        dispersion,
        omega_c,
        omega_p,
        omega,
        X_range_glauert,
    )

    cunningham_root_pairs = solve_dispersion_relation(
        dispersion,
        omega_c,
        omega_p,
        omega,
        X_range_cunningham,
    )

    values_dict = {
        "Omega": (omega_c[0].value, omega_c[1].value),
        "omega_p": (omega_p[0].value, omega_p[1].value),
    }
    dispersion_poly_k = replace_cpdr_symbols(dispersion._poly_k, values_dict)

    # Calculate normalisation factors from Glauert
    glauert_norm_factor = compute_glauert_normalisation_factor(
        dispersion,
        dispersion_poly_k,
        glauert_root_pairs,
    )

    # Calculate integral of g(X)
    # Technically we should prepend point X=0 and we should use
    # left endpoint integration rule, not simpson, as this is
    # what Cunningham used in his paper.
    eval_gx = dispersion._wave_angles.eval(np.array(X_range_glauert))
    integral_gx = simpson(eval_gx, x=X_range_glauert)

    # Calculate Cunningham's normalisation factors
    cunningham_norm_factors = compute_cunningham_normalisation_factor(
        dispersion_poly_k,
        cunningham_root_pairs,
    )

    # Calculate the ratio of equation (4) to equation (5)
    ratios = []
    for X, cunningham_norm_factor in zip(X_range_cunningham, cunningham_norm_factors):
        ratios.append(
            (X, (1.0 / cunningham_norm_factor) / (integral_gx / glauert_norm_factor))
        )

    return ratios


def plot_figure2(
    norm_ratios,
    frequency_ratio,
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
            overlay_ratio1x, overlay_ratio1y, color="k", linestyle="--", alpha=0.6
        )

        overlay_ratio2x = overlay[:, 2]
        overlay_ratio2y = overlay[:, 3]
        plt.loglog(
            overlay_ratio2x, overlay_ratio2y, color="r", linestyle="--", alpha=0.6
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
        rf"$\omega_{{pe}}$/$\omega_{{ce}}$={frequency_ratio.value}"
    )
    plt.tight_layout()

    if save:
        plt.savefig(f"{filestem}.png", dpi=150)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_Figure2",
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

    cunningham_dat_filepath = Path(args.path) / f"Figure2{args.figure}.dat"
    if not cunningham_dat_filepath.is_file():
        raise Exception(
            f"Incorrect filepath for Cunningham's dat file: {cunningham_dat_filepath}"
        )

    filestem = cunningham_dat_filepath.stem
    if args.figure == "a" or args.figure == "b":
        X_min = 0.00
        X_max = 1.00
        # X_npoints = 1000
        xticks = [0.01, 0.10, 1.00]
        yticks = [0.1, 1.0, 10.0, 100.0]
    elif args.figure == "c" or args.figure == "d":
        X_min = 0.00
        X_max = 5.67
        # X_npoints = 5000
        xticks = [0.01, 0.10, 1.00, 10.00]
        yticks = [0.1, 1.0, 10.0, 100.0, 1000.0]

    if args.figure == "a" or args.figure == "c":
        frequency_ratio = 1.5 * u.dimensionless_unscaled
        resonance_cone_limits = [6.67910235853265, 1.15458277480413]
    elif args.figure == "b" or args.figure == "d":
        frequency_ratio = 10.0 * u.dimensionless_unscaled
        resonance_cone_limits = [8.03773679895362, 1.41695373926307]

    # ============================== START ============================== #
    # Those should be attributes of one of the main classes
    # Magnetic field
    M = 8.033454e15 * (u.tesla * u.m**3)
    mlat = Angle(0, u.deg)
    l_shell = 4.5 * u.dimensionless_unscaled
    B = (M * math.sqrt(1 + 3 * math.sin(mlat.rad) ** 2)) / (
        l_shell**3 * const.R_earth**3 * math.cos(mlat.rad) ** 6
    )

    q_e = -const.e.si  # Signed electron charge
    q_p = const.e.si  # Signed proton charge

    # Gyrofrequency can be calculated directly using electron charge, mass, and
    # the magnetic field.
    Omega_e = (q_e * B) / const.m_e

    # Application of the frequency ratio yields the electron plasma frequency.
    Omega_e_abs = abs(Omega_e)
    omega_pe = Omega_e_abs * frequency_ratio

    # We assume that the number density of electrons and protons are equal.
    # These can be derived from the electron plasma frequency.
    n_ = omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2

    Omega_p = (q_p * B) / const.m_p
    omega_pp = np.sqrt((n_ * q_p**2) / (const.eps0 * const.m_p))
    # =============================== END =============================== #

    # ============================== START ============================== #
    # We need those because they are input arguments to the new Cpdr class.
    # They are not needed for this script.
    RKE = 1.0 * u.MeV  # Relativistic kinetic energy (Mega-electronvolts)
    alpha = Angle(5, u.deg)  # pitch angle

    # Lower and upper cut-off frequencies
    omega_m = 0.35 * Omega_e_abs
    delta_omega = 0.15 * Omega_e_abs
    omega_lc = omega_m - 1.5 * delta_omega
    omega_uc = omega_m + 1.5 * delta_omega

    # Resonances
    n_min = -5
    n_max = 5
    n_range = u.Quantity(
        range(n_min, n_max + 1), unit=u.dimensionless_unscaled, dtype=np.int32
    )
    # =============================== END =============================== #

    piran_particle_list = (PiranParticle("e", n_), PiranParticle("H+", n_))
    cpdr_particles = Particles(piran_particle_list, RKE, alpha)
    cpdr_wave_angles = Gaussian(0, 1e10, 0, 0.577)
    cpdr_wave_freqs = Gaussian(omega_lc, omega_uc, omega_m, delta_omega)
    cpdr_mag_field = MagField(mlat, l_shell)
    cpdr_resonances = n_range

    dispersion = Cpdr(
        cpdr_particles,
        cpdr_wave_angles,
        cpdr_wave_freqs,
        cpdr_mag_field,
        cpdr_resonances,
    )
    dispersion.as_poly_in_k()

    # Load data from Cunningham paper
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

    X_max_limits = [
        min(X_max, 0.9999 * resonance_cone_limits[0]),
        min(X_max, 0.9999 * resonance_cone_limits[1]),
    ]

    # Use points from .dat files for integrating glauert's norm factor
    # restricted to 0, min(X_max, sqrt(-P/S).
    X_range_glauert_integral0 = u.Quantity(np.insert(X_range_cunningham[0], 0, X_min))
    X_range_glauert_integral1 = u.Quantity(np.insert(X_range_cunningham[1], 0, X_min))
    # And restrict them
    X_range_glauert_integral0 = X_range_glauert_integral0[
        X_range_glauert_integral0 < X_max_limits[0]
    ]
    X_range_glauert_integral1 = X_range_glauert_integral1[
        X_range_glauert_integral1 < X_max_limits[1]
    ]

    # Or use a uniform distribution between X_min and X_upper = min(X_max, sqrt(-P/S))
    # X_range_glauert_integral0 = u.Quantity(
    #     np.linspace(X_min, X_max_limits[0], X_npoints), unit=u.dimensionless_unscaled
    # )
    # X_range_glauert_integral1 = u.Quantity(
    #     np.linspace(X_min, X_max_limits[1], X_npoints), unit=u.dimensionless_unscaled
    # )

    X_range_glauert_integral = [X_range_glauert_integral0, X_range_glauert_integral1]

    omega_ratios = [0.125, 0.575]

    ratio1 = calculate_ratio(
        dispersion,
        (Omega_e, Omega_p),
        (omega_pe, omega_pp),
        omega_ratios[0],
        X_range_glauert_integral[0],
        X_range_cunningham[0],
    )

    ratio2 = calculate_ratio(
        dispersion,
        (Omega_e, Omega_p),
        (omega_pe, omega_pp),
        omega_ratios[1],
        X_range_glauert_integral[1],
        X_range_cunningham[1],
    )

    if args.overlay:
        overlay = cunningham_figure_data
    else:
        overlay = None

    plot_figure2(
        (ratio1, ratio2),
        frequency_ratio,
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
            f"{frequency_ratio.value} and the ratio of the wave frequency "
            f"to the unsigned electron gyrofrequency is {omega_ratios[0]}\n"
        )
        comment += "Column 3 is tangent of the wave normal angle used in column 4\n"
        comment += (
            f"Column 4 is the ratio of equation 4 to equation 5 using the X "
            f"in column 3 when Xmax={X_max} and the ratio of the plasma "
            f"frequency to the unsigned electron gyrofrequency is "
            f"{frequency_ratio.value} and the ratio of the wave frequency "
            f"to the unsigned electron gyrofrequency is {omega_ratios[1]}"
        )

        ratios = np.empty_like(cunningham_figure_data)
        for i in range(ratios.shape[0]):
            ratios[i, 0] = ratio1[i][0]
            ratios[i, 1] = ratio1[i][1]
            ratios[i, 2] = ratio2[i][0]
            ratios[i, 3] = ratio2[i][1]

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
