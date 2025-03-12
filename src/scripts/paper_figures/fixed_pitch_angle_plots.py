"""
fixed_pitch_angle_plots.py

Generates and saves fixed pitch angle plots from simulation results.

This script produces fixed pitch angle plots of diffusion coefficients based
on simulation results.
The user must specify the paths to the simulation result files: path_to_glau
and path_to_cunn.
These paths correspond to results obtained using normalisation factors from
Glauert and Horne 2005 and Cunningham 2023 respectively.

Two figures are generated and saved in the current working directory:
    - Daa_RATIO_XMAX.png (pitch angle diffusion coefficients)
    - DEE_RATIO_XMAX.png (energy diffusion coefficients)

Corresponding .dat files containing the raw numerical data for each figure are
also generated.

Usage:
    python path/to/fixed_pitch_angle_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
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
    ax.loglog(
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
    ratio,
    pitch_angle,
    Xmax,
    xlabel,
    ylabel,
    y_exp_limits,
    legend_loc,
):
    xticks = [0.001, 0.010, 0.100, 1.000]
    yticks = [10 ** (n) for n in range(y_exp_limits[0], y_exp_limits[1], 2)]

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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc=legend_loc)

    title_omega = rf"$\omega_{{\text{{pe}}}}/\omega_{{\text{{ce}}}}={ratio}$"
    title_alpha = rf"$\alpha={pitch_angle}$"
    title_Xmax = rf"$\text{{X}}_{{\text{{max}}}}={Xmax}$"
    ax.set_title(f"{title_omega}, {title_alpha}, {title_Xmax}")


def main():
    # Paths to results
    path_to_glau = ""  # Glauert norm factor
    path_to_cunn = ""  # Cunningham norm factor

    bounce = False

    # Load all data
    xx_g, yy_aa_g, yy_aE_g, yy_EE_g, method_g, ratio_g, X_max_g, pitch_angle_g = (
        load_and_post_process(path_to_glau, bounce, False)
    )
    xx_c, yy_aa_c, yy_aE_c, yy_EE_c, method_c, ratio_c, X_max_c, pitch_angle_c = (
        load_and_post_process(path_to_cunn, bounce, False)
    )

    # Check that the parameters are correct
    if not (ratio_g == ratio_c and ratio_g in [0.75, 1.5, 10.0]):
        raise ValueError(
            "Check plasma-to-gyro ratios. Must be same and one of 0.75, 1.5 or 10.0"
        )
    if not (pitch_angle_g == pitch_angle_c == 0.125):
        raise ValueError("Check pitch angles. Must be 0.125.")
    if not (method_g == 0 and method_c == 1):
        raise ValueError("Check normalisation methods")
    if not (X_max_g == X_max_c):
        raise ValueError("Check X_max. Must be the same")
    if not (xx_g == xx_c):
        raise ValueError("Check energy discretisation")

    # At this point all pitch angles, ratios,
    # X_max and energy discretisations are the same.
    pitch_angle = pitch_angle_g
    ratio = ratio_g
    X_max = X_max_g
    xx = xx_g

    # Init figure and axes
    fig_aa, ax_aa = plt.subplots()
    fig_aE, ax_aE = plt.subplots()
    fig_EE, ax_EE = plt.subplots()

    # Plot data
    color1 = "C0"
    color2 = "C1"
    linestyle = "-"  # solid
    marker = ""
    alpha_val = 0.90
    label1 = "Glauert & Horne (PIRAN)"
    label2 = "Cunningham (PIRAN)"
    plot_figure(ax_aa, xx, yy_aa_g, color1, linestyle, marker, alpha_val, label1)
    plot_figure(ax_aa, xx, yy_aa_c, color2, linestyle, marker, alpha_val, label2)

    plot_figure(ax_aE, xx, yy_aE_g, color1, linestyle, marker, alpha_val, label1)
    plot_figure(ax_aE, xx, yy_aE_c, color2, linestyle, marker, alpha_val, label2)

    plot_figure(ax_EE, xx, yy_EE_g, color1, linestyle, marker, alpha_val, label1)
    plot_figure(ax_EE, xx, yy_EE_c, color2, linestyle, marker, alpha_val, label2)

    # Format figures, add legend, labels etc.
    if bounce:
        langle = r"\langle"
        rangle = r"\rangle"
    else:
        langle = ""
        rangle = ""

    xlabel = "Energy (MeV)"
    ylabel_aa = rf"${langle}\text{{D}}_{{\alpha\alpha}}{rangle} / \text{{p}}^2\ (\text{{s}}^{{-1}})$"
    ylabel_aE = rf"$|{langle}\text{{D}}_{{\alpha\text{{E}}}}{rangle}| / \text{{E}}^2\ (\text{{s}}^{{-1}})$"
    ylabel_EE = rf"${langle}\text{{D}}_{{\text{{EE}}}}{rangle} / \text{{E}}^2\ (\text{{s}}^{{-1}})$"

    if bounce:
        raise ValueError("Not implemented yet")
    else:
        y_exp_limits_aa = (-10, 3)
        y_exp_limits_aE = (-10, 3)
        y_exp_limits_EE = (-10, 3)

        legend_loc_aa = "upper right"
        legend_loc_aE = "upper right"
        legend_loc_EE = "upper right"

    format_figure(
        ax_aa,
        ratio,
        pitch_angle,
        X_max,
        xlabel,
        ylabel_aa,
        y_exp_limits_aa,
        legend_loc_aa,
    )
    format_figure(
        ax_aE,
        ratio,
        pitch_angle,
        X_max,
        xlabel,
        ylabel_aE,
        y_exp_limits_aE,
        legend_loc_aE,
    )
    format_figure(
        ax_EE,
        ratio,
        pitch_angle,
        X_max,
        xlabel,
        ylabel_EE,
        y_exp_limits_EE,
        legend_loc_EE,
    )

    fig_aa.tight_layout()
    fig_aE.tight_layout()
    fig_EE.tight_layout()

    fname_suffix = f"{ratio}ratio_{X_max}xmax"

    fig_aa.savefig(f"{'ba' if bounce else ''}Daa_{fname_suffix}.png", dpi=300)
    # fig_aE.savefig(f"{'ba' if bounce else ''}DaE_{fname_suffix}.png", dpi=300)
    fig_EE.savefig(f"{'ba' if bounce else ''}DEE_{fname_suffix}.png", dpi=300)

    # fig_aa.savefig(f"{'ba' if bounce else ''}Daa_{fname_suffix}.eps")
    # fig_aE.savefig(f"{'ba' if bounce else ''}DaE_{fname_suffix}.eps")
    # fig_EE.savefig(f"{'ba' if bounce else ''}DEE_{fname_suffix}.eps")

    # ======================================================================= #

    # Create dat files
    yy_aa = [yy_aa_g, yy_aa_c]
    yy_aE = [yy_aE_g, yy_aE_c]
    yy_EE = [yy_EE_g, yy_EE_c]
    methods = [method_g, method_c]

    nrows = len(xx)

    res_aa = np.zeros((nrows, 3))
    res_aE = np.zeros((nrows, 3))
    res_EE = np.zeros((nrows, 3))

    comment_aa = "Column 1 is the energy in MeV\n"
    comment_aE = "Column 1 is the energy in MeV\n"
    comment_EE = "Column 1 is the energy in MeV\n"

    res_aa[:, 0] = xx
    res_aE[:, 0] = xx
    res_EE[:, 0] = xx

    for ii in range(len(methods)):
        if methods[ii] == 0:
            m = "Glauert and Horne"
        elif methods[ii] == 1:
            m = "Cunningham"

        comment_aa += f"Column {ii + 2} is the pitch angle {'bounce-averaged ' if bounce else ''}diffusion coefficient at {pitch_angle} degrees using the method of {m} when the ratio of the electron plasma frequency to the unsigned electron gyrofrequency is {ratio} and Xmax={X_max}\n"
        comment_aE += f"Column {ii + 2} is the mixed pitch angle energy {'bounce-averaged ' if bounce else ''}diffusion coefficient at {pitch_angle} degrees using the method of {m} when the ratio of the electron plasma frequency to the unsigned electron gyrofrequency is {ratio} and Xmax={X_max}\n"
        comment_EE += f"Column {ii + 2} is the energy {'bounce-averaged ' if bounce else ''}diffusion coefficient at {pitch_angle} degrees using the method of {m} when the ratio of the electron plasma frequency to the unsigned electron gyrofrequency is {ratio} and Xmax={X_max}\n"

        res_aa[:, ii + 1] = yy_aa[ii]
        res_aE[:, ii + 1] = yy_aE[ii]
        res_EE[:, ii + 1] = yy_EE[ii]

    format_seq = ["%.6f", "%.7e", "%.7e"]
    delimiter_char = "\t"
    newline_char = "\n"
    comments_char = "#"

    np.savetxt(
        f"{'ba' if bounce else ''}Daa_{fname_suffix}.dat",
        res_aa,
        header=comment_aa,
        fmt=format_seq,
        delimiter=delimiter_char,
        newline=newline_char,
        comments=comments_char,
    )

    # np.savetxt(
    #     f"{'ba' if bounce else ''}DaE_{fname_suffix}.dat",
    #     res_aE,
    #     header=comment_aE,
    #     fmt=format_seq,
    #     delimiter=delimiter_char,
    #     newline=newline_char,
    #     comments=comments_char,
    # )

    np.savetxt(
        f"{'ba' if bounce else ''}DEE_{fname_suffix}.dat",
        res_EE,
        header=comment_EE,
        fmt=format_seq,
        delimiter=delimiter_char,
        newline=newline_char,
        comments=comments_char,
    )


if __name__ == "__main__":
    main()
