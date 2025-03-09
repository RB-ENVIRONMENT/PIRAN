"""
normalisers_ratio.py

Generates and saves figures and data files depicting the ratio of normalisers
for specific plasma-to-gyro ratios and X_max values.

This script calculates and visualises the ratio of normalisers for
plasma-to-gyro ratios of 1.5 and 10, and X_max values of 1.0 and 5.67.
It generates four figures, 'normalisers_ratio_[abcd].png', and corresponding
data files, 'normalisers_ratio_[abcd].dat', where [abcd] represents a unique
identifier for each combination of parameters.
The .dat files contain the raw numerical values used to create the figures.

Usage:
    python path/to/normalisers_ratio.py
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magpoint import MagPoint
from piran.normalisation import (
    compute_cunningham_norm_factor,
    compute_glauert_norm_factor,
)
from piran.plasmapoint import PlasmaPoint


def format_figure(
    ax,
    xticks,
    yticks,
    plasma_over_gyro_ratio,
    legend_loc,
):
    x_lim_min = xticks[0]
    x_lim_max = xticks[-1]

    y_lim_min = yticks[0]
    y_lim_max = yticks[-1]

    ax.set_xticks(xticks, labels=[str(v) for v in xticks])
    ax.set_yticks(yticks, labels=[str(v) for v in yticks])

    ax.tick_params("x", which="both", top=True, labeltop=False)
    ax.tick_params("y", which="both", right=True, labelright=False)

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_xlabel(r"X ($\tan{\theta}$)")
    ax.set_ylabel("Ratio")

    ax.legend(loc=legend_loc)

    ax.set_title(
        rf"Ratio of normalisers for "
        rf"$\omega_{{pe}}$/$\omega_{{ce}}$={plasma_over_gyro_ratio}"
    )


def main():
    X_min = 0.0 << u.dimensionless_unscaled
    X_max_l = [1.0, 5.67] << u.dimensionless_unscaled
    X_npoints_l = [1000, 2000]
    plasma_over_gyro_ratio_l = [1.5, 10.0]

    omega_ratios_l = [0.125, 0.575]
    colours = ["C4", "C8"]
    linestyles = ["-", (0, (5, 1))]
    xticks = [[0.01, 0.10, 1.00], [0.01, 0.10, 1.00, 10.00]]
    yticks = [[0.1, 1.0, 10.0, 100.0], [0.1, 1.0, 10.0, 100.0, 1000.0]]

    xaxis_min = [0.0006, 0.0001]
    xaxis_npoints = 10000

    particles = ("e", "p+")
    X_m = 0.0 << u.dimensionless_unscaled  # peak
    X_w = 0.577 << u.dimensionless_unscaled  # angular width
    mlat_deg = Angle(0.0, u.deg)
    l_shell = 4.5
    epsilon = 0.9999  # Glauert & Horne 2005 paragraph 23
    suffix = ["a", "b", "c", "d"]

    for ii, X_max in enumerate(X_max_l):
        X_npoints = X_npoints_l[ii]
        X_range = u.Quantity(np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled)
        wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

        for jj, plasma_over_gyro_ratio in enumerate(plasma_over_gyro_ratio_l):
            filename = f"normalisers_ratio_{suffix[ii * len(X_max_l) + jj]}"
            print(filename)

            mag_point = MagPoint(mlat_deg, l_shell)
            plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
            cpdr = Cpdr(plasma_point)

            # Init array for dat file
            ratios_dat_file = np.zeros((xaxis_npoints, 4))
            comments_dat_file = ""

            # Init figure and axes
            fig, ax = plt.subplots()

            for kk, omega_ratio in enumerate(omega_ratios_l):
                omega = (np.abs(cpdr.plasma.gyro_freq[0]) * omega_ratio)  # Electron gyrofrequency * ratio

                # Glauert's norm factor
                # For Glauert's norm factor use a uniform distribution
                # between X_min and X_upper=min(X_max, epsilon*sqrt(-P/S)).
                # We use Simpson's rule for the integration.
                resonance_cone_angle = -cpdr.stix.P(omega) / cpdr.stix.S(omega)
                X_upper = min(X_max, epsilon * np.sqrt(resonance_cone_angle))
                print(f"{plasma_over_gyro_ratio=}, {omega_ratio=}, X_max={X_max.value}, epsilon * sqrt(res_cone_angle)={epsilon * np.sqrt(resonance_cone_angle).value}, X_upper={X_upper.value}")
                X_range_glauert = X_range[X_range <= X_upper]
                norm_factor_glauert = compute_glauert_norm_factor(cpdr, omega, X_range_glauert, wave_norm_angle_dist, method="simpson")

                xaxis_max = epsilon * np.sqrt(resonance_cone_angle)
                xaxis_discretisation = u.Quantity(np.linspace(xaxis_min[kk], xaxis_max, xaxis_npoints), unit=u.dimensionless_unscaled)

                # Cunningham's norm factors
                norm_factors_cunningham = compute_cunningham_norm_factor(cpdr, omega, xaxis_discretisation)

                # Calculate integral of g(X) to normalise it, using Simpson's rule.
                eval_gx = wave_norm_angle_dist.eval(X_range)
                integral_gx = simpson(eval_gx, x=X_range)

                # Calculate the ratio of equation (4) to equation (5) from Cunningham 2023
                ratio = norm_factor_glauert / (norm_factors_cunningham * integral_gx)

                # Write to dat array
                column_id_xaxis = 2 * kk + 0
                column_id_ratio = 2 * kk + 1
                ratios_dat_file[:, column_id_xaxis] = xaxis_discretisation
                ratios_dat_file[:, column_id_ratio] = ratio
                comments_dat_file += f"Column {column_id_xaxis + 1} is tangent of the wave normal angle used in column {column_id_ratio + 1}\n"
                comments_dat_file += f"Column {column_id_ratio + 1} is the ratio of equation 4 to equation 5 from Cunningham 2023 using the X in column {column_id_xaxis + 1} when Xmax={X_max}, the ratio of the plasma frequency to the unsigned electron gyrofrequency is {plasma_over_gyro_ratio} and the ratio of the wave frequency to the unsigned electron gyrofrequency is {omega_ratio}\n"

                # Plot
                ax.loglog(xaxis_discretisation, ratio, color=colours[kk], linestyle=linestyles[kk], label=rf"$\omega$/$\omega_{{ce}}$={omega_ratio}")
                format_figure(ax, xticks[ii], yticks[ii], plasma_over_gyro_ratio, "lower left")
                fig.tight_layout()

                fig.savefig(f"{filename}.png", dpi=300)
                # fig.savefig(f"{filename}.eps")

            # Create the dat file
            format_seq = ["%.7f", "%.7e", "%.7f", "%.7e"]
            delimiter_char = "\t"
            newline_char = "\n"
            comments_char = "#"

            np.savetxt(
                f"{filename}.dat",
                ratios_dat_file,
                header=comments_dat_file,
                fmt=format_seq,
                delimiter=delimiter_char,
                newline=newline_char,
                comments=comments_char,
            )


if __name__ == "__main__":
    main()
