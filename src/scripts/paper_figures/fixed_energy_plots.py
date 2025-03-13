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
fixed_energy_plots.py

Generates and saves fixed energy plots from simulation results for both local
and bounce-averaged diffusion coefficients.

This script produces fixed energy plots of diffusion coefficients based on
simulation results. It supports both local and bounce-averaged diffusion
coefficients.
The user must specify the paths to the simulation result files: path_to_glauert1,
path_to_glauert2, path_to_cunningham1, and path_to_cunningham2.
These paths correspond to results obtained using normalisation factors from
Glauert and Horne 2005 (plasma-to-gyro ratios 1.5 and 10, respectively) and
Cunningham 2023 (same plasma-to-gyro ratios).
The 'bounce' variable must be set to True for bounce-averaged results or False
for local diffusion coefficients.

Three figures are generated and saved in the current working directory:
    - Daa_ENERGY.png (pitch angle diffusion coefficients)
    - DaE_ENERGY.png (mixed pitch angle-energy diffusion coefficients)
    - DEE_ENERGY.png (energy diffusion coefficients)

The 'ba' prefix is added to the filenames if 'bounce' is set to True.
Corresponding .dat files containing the raw numerical data for each figure are
also generated.

Usage:
    python path/to/fixed_energy_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.ticker import LogFormatterMathtext, MultipleLocator
from misc import load_and_post_process


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
    ax.semilogy(
        xx,
        yy,
        color=color,
        linestyle=linestyle,
        marker=marker,
        alpha=alpha,
        label=label,
    )


def format_figure(
    ax,
    energy,
    xlabel,
    ylabel,
    y_exp_limits,
    legend_loc,
    ratio1_loc,
    ratio2_loc,
):
    xticks = list(range(0, 91, 15))
    yticks = [10 ** (n) for n in range(y_exp_limits[0], y_exp_limits[1], 1)]

    x_lim_min = xticks[0]
    x_lim_max = xticks[-1]

    y_lim_min = yticks[0]
    y_lim_max = yticks[-1]

    ax.set_xticks(xticks, labels=[str(v) for v in xticks])
    ax.set_yticks(yticks, labels=[str(v) for v in yticks])

    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    ax.tick_params("x", which="both", top=True, labeltop=False)
    ax.tick_params("y", which="both", right=True, labelright=False)

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.text(
        ratio1_loc[0], ratio1_loc[1], r"$\omega_{\text{pe}}/\omega_{\text{ce}}=1.5$"
    )
    ax.text(ratio2_loc[0], ratio2_loc[1], r"$\omega_{\text{pe}}/\omega_{\text{ce}}=10$")

    ax.legend(loc=legend_loc)

    if energy.to(u.keV).value < 1000:
        energy_title = energy.to(u.keV)
    else:
        energy_title = energy.to(u.MeV)

    ax.set_title(f"KE={energy_title:.0f}")


def main():
    bounce = False  # True or False

    # Paths to results
    path_to_glauert1 = ""  # Glauert norm factor, 1.5 ratio
    path_to_glauert2 = ""  # Glauert norm factor, 10 ratio
    path_to_cunningham1 = ""  # Cunningham norm factor, 1.5 ratio
    path_to_cunningham2 = ""  # Cunningham norm factor, 10 ratio

    # Load all data
    xx_g1, yy_aa_g1, yy_aE_g1, yy_EE_g1, method_g1, ratio_g1, X_max_g1, energy_g1 = (
        load_and_post_process(path_to_glauert1, bounce, True)
    )
    xx_g2, yy_aa_g2, yy_aE_g2, yy_EE_g2, method_g2, ratio_g2, X_max_g2, energy_g2 = (
        load_and_post_process(path_to_glauert2, bounce, True)
    )
    xx_c1, yy_aa_c1, yy_aE_c1, yy_EE_c1, method_c1, ratio_c1, X_max_c1, energy_c1 = (
        load_and_post_process(path_to_cunningham1, bounce, True)
    )
    xx_c2, yy_aa_c2, yy_aE_c2, yy_EE_c2, method_c2, ratio_c2, X_max_c2, energy_c2 = (
        load_and_post_process(path_to_cunningham2, bounce, True)
    )

    # Check that the parameters are correct
    if energy_g1.to(u.keV).value not in [10, 100, 500, 1000]:
        raise ValueError("Only supports 10keV, 100keV, 500keV and 1MeV")
    if not (energy_g1 == energy_g2 == energy_c1 == energy_c2):
        raise ValueError("Check energies. Must be same.")
    if not (ratio_g1 == ratio_c1 == 1.5 and ratio_g2 == ratio_c2 == 10.0):
        raise ValueError("Check plasma-to-gyro ratios")
    if not (method_g1 == method_g2 == 0 and method_c1 == method_c2 == 1):
        raise ValueError("Check normalisation methods")
    if not (X_max_g1 == X_max_g2 == X_max_c1 == X_max_c2):
        raise ValueError("Check X_max. Must be the same")
    if not (xx_g1 == xx_g2 == xx_c1 == xx_c2):
        raise ValueError("Check pitch angle discretisation")

    # At this point all pitch angle discretisations
    # and all energies are the same.
    energy = energy_g1
    xx = xx_g1

    # Init figure and axes
    fig_aa, ax_aa = plt.subplots()
    fig_aE, ax_aE = plt.subplots()
    fig_EE, ax_EE = plt.subplots()

    # Plot data
    color1 = "C0"
    color2 = "C1"
    linestyle1 = "-"  # solid
    linestyle2 = (0, (5, 1))  # densely dashed
    marker = ""
    alpha_val = 0.80
    label1 = "Glauert & Horne (PIRAN)"
    label2 = "Cunningham (PIRAN)"
    empty_label = ""
    plot_figure(ax_aa, xx, yy_aa_g1, color1, linestyle1, marker, alpha_val, label1)
    plot_figure(ax_aa, xx, yy_aa_g2, color1, linestyle2, marker, alpha_val, empty_label)
    plot_figure(ax_aa, xx, yy_aa_c1, color2, linestyle1, marker, alpha_val, label2)
    plot_figure(ax_aa, xx, yy_aa_c2, color2, linestyle2, marker, alpha_val, empty_label)

    plot_figure(ax_aE, xx, yy_aE_g1, color1, linestyle1, marker, alpha_val, label1)
    plot_figure(ax_aE, xx, yy_aE_g2, color1, linestyle2, marker, alpha_val, empty_label)
    plot_figure(ax_aE, xx, yy_aE_c1, color2, linestyle1, marker, alpha_val, label2)
    plot_figure(ax_aE, xx, yy_aE_c2, color2, linestyle2, marker, alpha_val, empty_label)

    plot_figure(ax_EE, xx, yy_EE_g1, color1, linestyle1, marker, alpha_val, label1)
    plot_figure(ax_EE, xx, yy_EE_g2, color1, linestyle2, marker, alpha_val, empty_label)
    plot_figure(ax_EE, xx, yy_EE_c1, color2, linestyle1, marker, alpha_val, label2)
    plot_figure(ax_EE, xx, yy_EE_c2, color2, linestyle2, marker, alpha_val, empty_label)

    # Format figures, add legend, labels etc.
    if bounce:
        langle = r"\langle"
        rangle = r"\rangle"
        xlabel = r"Equatorial pitch angle (degrees)"
    else:
        langle = ""
        rangle = ""
        xlabel = r"Local pitch angle (degrees)"

    ylabel_aa = rf"${langle}\text{{D}}_{{\alpha\alpha}}{rangle} / \text{{p}}^2\ (\text{{s}}^{{-1}})$"
    ylabel_aE = rf"$|{langle}\text{{D}}_{{\alpha\text{{E}}}}{rangle}| / \text{{E}}^2\ (\text{{s}}^{{-1}})$"
    ylabel_EE = rf"${langle}\text{{D}}_{{\text{{EE}}}}{rangle} / \text{{E}}^2\ (\text{{s}}^{{-1}})$"

    if bounce:
        if energy.to(u.keV).value == 10:
            y_exp_limits_aa = (-7, 0)
            y_exp_limits_aE = (-7, 0)
            y_exp_limits_EE = (-7, 0)

            legend_loc_aa = "upper left"
            legend_loc_aE = "upper left"
            legend_loc_EE = "upper left"

            ratio1_loc_aa = (32, 4 * 10**-7)
            ratio1_loc_aE = (32, 4 * 10**-7)
            ratio1_loc_EE = (32, 3 * 10**-7)

            ratio2_loc_aa = (64, 8 * 10**-6)
            ratio2_loc_aE = (64, 1 * 10**-6)
            ratio2_loc_EE = (64, 1 * 10**-6)
    else:
        if energy.to(u.keV).value == 10:
            y_exp_limits_aa = (-7, 0)
            y_exp_limits_aE = (-7, 0)
            y_exp_limits_EE = (-7, 0)

            legend_loc_aa = "upper left"
            legend_loc_aE = "upper left"
            legend_loc_EE = "upper left"

            ratio1_loc_aa = (32, 2 * 10**-6)
            ratio1_loc_aE = (32, 1 * 10**-6)
            ratio1_loc_EE = (32, 3 * 10**-7)

            ratio2_loc_aa = (64, 8 * 10**-6)
            ratio2_loc_aE = (64, 1 * 10**-6)
            ratio2_loc_EE = (64, 3 * 10**-7)
        elif energy.to(u.keV).value == 100:
            y_exp_limits_aa = (-7, 0)
            y_exp_limits_aE = (-7, 0)
            y_exp_limits_EE = (-7, 0)

            legend_loc_aa = "upper left"
            legend_loc_aE = "upper left"
            legend_loc_EE = "upper left"

            ratio1_loc_aa = (13, 4 * 10**-3)
            ratio1_loc_aE = (10, 5 * 10**-4)
            ratio1_loc_EE = (18, 2 * 10**-4)

            ratio2_loc_aa = (35, 5 * 10**-6)
            ratio2_loc_aE = (40, 1 * 10**-6)
            ratio2_loc_EE = (44, 3 * 10**-7)
        elif energy.to(u.keV).value == 500:
            y_exp_limits_aa = (-7, 0)
            y_exp_limits_aE = (-7, 0)
            y_exp_limits_EE = (-7, 0)

            legend_loc_aa = "upper left"
            legend_loc_aE = "upper left"
            legend_loc_EE = "upper left"

            ratio1_loc_aa = (13, 1 * 10**-4)
            ratio1_loc_aE = (13, 2 * 10**-5)
            ratio1_loc_EE = (17, 9 * 10**-6)

            ratio2_loc_aa = (54, 4 * 10**-6)
            ratio2_loc_aE = (46, 1 * 10**-6)
            ratio2_loc_EE = (43, 3 * 10**-7)
        elif energy.to(u.keV).value == 1000:
            y_exp_limits_aa = (-7, 0)
            y_exp_limits_aE = (-7, 0)
            y_exp_limits_EE = (-7, 0)

            legend_loc_aa = "upper left"
            legend_loc_aE = "upper left"
            legend_loc_EE = "upper left"

            ratio1_loc_aa = (20, 3 * 10**-5)
            ratio1_loc_aE = (18, 3 * 10**-6)
            ratio1_loc_EE = (21, 2 * 10**-6)

            ratio2_loc_aa = (47, 4 * 10**-6)
            ratio2_loc_aE = (55, 1 * 10**-6)
            ratio2_loc_EE = (59, 3 * 10**-7)

    format_figure(
        ax_aa,
        energy,
        xlabel,
        ylabel_aa,
        y_exp_limits_aa,
        legend_loc_aa,
        ratio1_loc_aa,
        ratio2_loc_aa,
    )
    format_figure(
        ax_aE,
        energy,
        xlabel,
        ylabel_aE,
        y_exp_limits_aE,
        legend_loc_aE,
        ratio1_loc_aE,
        ratio2_loc_aE,
    )
    format_figure(
        ax_EE,
        energy,
        xlabel,
        ylabel_EE,
        y_exp_limits_EE,
        legend_loc_EE,
        ratio1_loc_EE,
        ratio2_loc_EE,
    )

    fig_aa.tight_layout()
    fig_aE.tight_layout()
    fig_EE.tight_layout()

    if energy.to(u.keV).value < 1000:
        energy_name = f"{energy.to(u.keV).value:.0f}keV"
    else:
        energy_name = f"{energy.to(u.MeV).value:.0f}MeV"

    fig_aa.savefig(f"{'ba' if bounce else ''}Daa_{energy_name}.png", dpi=300)
    fig_aE.savefig(f"{'ba' if bounce else ''}DaE_{energy_name}.png", dpi=300)
    fig_EE.savefig(f"{'ba' if bounce else ''}DEE_{energy_name}.png", dpi=300)

    # fig_aa.savefig(f"{'ba' if bounce else ''}Daa_{energy_name}.eps")
    # fig_aE.savefig(f"{'ba' if bounce else ''}DaE_{energy_name}.eps")
    # fig_EE.savefig(f"{'ba' if bounce else ''}DEE_{energy_name}.eps")

    # ======================================================================= #

    # Create dat files
    yy_aa = [yy_aa_g1, yy_aa_g2, yy_aa_c1, yy_aa_c2]
    yy_aE = [yy_aE_g1, yy_aE_g2, yy_aE_c1, yy_aE_c2]
    yy_EE = [yy_EE_g1, yy_EE_g2, yy_EE_c1, yy_EE_c2]
    ratios = [ratio_g1, ratio_g2, ratio_c1, ratio_c2]
    methods = [method_g1, method_g2, method_c1, method_c2]

    nrows = len(xx)

    res_aa = np.zeros((nrows, 5))
    res_aE = np.zeros((nrows, 5))
    res_EE = np.zeros((nrows, 5))

    comment_aa = f"Column 1 is the {'equatorial' if bounce else 'local'} pitch angle in degrees\n"
    comment_aE = f"Column 1 is the {'equatorial' if bounce else 'local'} pitch angle in degrees\n"
    comment_EE = f"Column 1 is the {'equatorial' if bounce else 'local'} pitch angle in degrees\n"

    res_aa[:, 0] = xx
    res_aE[:, 0] = xx
    res_EE[:, 0] = xx

    for ii in range(len(methods)):
        if methods[ii] == 0:
            m = "Glauert and Horne"
        elif methods[ii] == 1:
            m = "Cunningham"

        comment_aa += f"Column {ii + 2} is the pitch angle {'bounce-averaged ' if bounce else ''}diffusion coefficient using the method of {m} when the ratio of the electron plasma frequency to the unsigned electron gyrofrequency is {ratios[ii]}\n"
        comment_aE += f"Column {ii + 2} is the mixed pitch angle energy {'bounce-averaged ' if bounce else ''}diffusion coefficient using the method of {m} when the ratio of the electron plasma frequency to the unsigned electron gyrofrequency is {ratios[ii]}\n"
        comment_EE += f"Column {ii + 2} is the energy {'bounce-averaged ' if bounce else ''}diffusion coefficient using the method of {m} when the ratio of the electron plasma frequency to the unsigned electron gyrofrequency is {ratios[ii]}\n"

        res_aa[:, ii + 1] = yy_aa[ii]
        res_aE[:, ii + 1] = yy_aE[ii]
        res_EE[:, ii + 1] = yy_EE[ii]

    format_seq = ["%.3f", "%.6e", "%.6e", "%.6e", "%.6e"]
    delimiter_char = "\t"
    newline_char = "\n"
    comments_char = "#"

    np.savetxt(
        f"{'ba' if bounce else ''}Daa_{energy_name}.dat",
        res_aa,
        header=comment_aa,
        fmt=format_seq,
        delimiter=delimiter_char,
        newline=newline_char,
        comments=comments_char,
    )

    np.savetxt(
        f"{'ba' if bounce else ''}DaE_{energy_name}.dat",
        res_aE,
        header=comment_aE,
        fmt=format_seq,
        delimiter=delimiter_char,
        newline=newline_char,
        comments=comments_char,
    )

    np.savetxt(
        f"{'ba' if bounce else ''}DEE_{energy_name}.dat",
        res_EE,
        header=comment_EE,
        fmt=format_seq,
        delimiter=delimiter_char,
        newline=newline_char,
        comments=comments_char,
    )


if __name__ == "__main__":
    main()
