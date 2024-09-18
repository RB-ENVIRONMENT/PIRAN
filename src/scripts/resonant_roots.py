# Script to reproduce Figure5a-like plots from Glauert 2005.
# Edit the parameters block inside main().
# It will create one plot per wave normal angle X in X_range.
# If you want to save the plots on disk instead of displaying
# them on screen, change the `save=False` to `save=True` in the
# plot_resonant_roots() function call. They will be saved in the
# current working directory as resonant_roots_???.png

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


def get_resonance_condition(cpdr: Cpdr, X, y_list, k_par_sign):
    """
    k_par_sign: 'positive' for theta in [0, pi/2] or
        'negative' for theta in (pi/2, pi].
    """
    psi = np.arctan(X)
    if k_par_sign == "positive":
        theta = psi.to(u.rad)
    elif k_par_sign == "negative":
        theta = (np.pi << u.rad) - psi.to(u.rad)

    n = cpdr.resonance
    electron_gyro = cpdr.plasma.gyro_freq[0]
    gamma = cpdr.gamma
    v_par = cpdr.v_par

    resonance_condition = []
    for y in y_list:
        omega = np.abs(electron_gyro) * y
        res_cond_k = (omega - (n * electron_gyro / gamma)) / (np.cos(theta) * v_par)
        x = res_cond_k * const.c / np.abs(electron_gyro)
        resonance_condition.append((x, y))

    return resonance_condition


def get_dispersion_relation(cpdr: Cpdr, X, y_list):
    dispersion_relation = []
    for y in y_list:
        electron_gyro = cpdr.plasma.gyro_freq[0]
        omega = np.abs(electron_gyro) * y

        # When y=1, omega becomes equal to the absolute value
        # of the gyrofrequency of the first particle/particle
        # of interest. For those cases CPDR becomes NaN,
        # because of the denominator of the Stix parameters.
        # However, when also X=0 then CPDR becomes a polynomial
        # with Sympy's `zoo` complex infinity as coefficients,
        # and we get "PolynomialError: multivariate polynomials not
        # supported" when we call `.as_poly()`.
        try:
            k_root = cpdr.solve_cpdr(omega, X)
            k_filtered = cpdr.filter(X, omega, k_root)
        except Exception:
            continue

        for k in k_filtered:
            if not np.isnan(k):
                x = k * const.c / np.abs(electron_gyro)
                dispersion_relation.append((x, y))

    return dispersion_relation


def plot_resonant_roots(
    cpdr: Cpdr,
    X,
    resonant_roots,
    resonance_condition_kpar_pos,
    resonance_condition_kpar_neg,
    dispersion_relation,
    i,
    save=False,
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    psi = np.arctan(X)
    n = cpdr.resonance
    RKE = cpdr.energy.to(u.MeV)
    alpha = cpdr.alpha
    electron_gyro_abs = np.abs(cpdr.plasma.gyro_freq[0])
    omega_lc = cpdr.omega_lc
    omega_uc = cpdr.omega_uc

    # Plot resonance condition for positive k_par
    res_x1 = [val[0].value for val in resonance_condition_kpar_pos]
    res_y1 = [val[1].value for val in resonance_condition_kpar_pos]
    plt.semilogy(
        res_x1,
        res_y1,
        color="tab:blue",
        linestyle="dashed",
        label=rf"Resonance condition n={n} $(\psi)$",
    )

    # Plot resonance condition for negative k_par
    res_x2 = [val[0].value for val in resonance_condition_kpar_neg]
    res_y2 = [val[1].value for val in resonance_condition_kpar_neg]
    plt.semilogy(
        res_x2,
        res_y2,
        color="tab:blue",
        linestyle="dotted",
        label=rf"Resonance condition n={n} $(\pi - \psi)$",
    )

    # Plot dispersion relation
    disp_x = [val[0].value for val in dispersion_relation]
    disp_y = [val[1].value for val in dispersion_relation]
    plt.semilogy(disp_x, disp_y, color="k", label="Dispersion relation")

    # Plot resonant roots
    for root in resonant_roots:
        if np.isnan(root.omega) or np.isnan(root.k):
            continue
        x = root.k * (const.c.value / electron_gyro_abs.value)
        y = root.omega / electron_gyro_abs.value
        plt.semilogy(x, y, color="tab:blue", marker="o")
    # x = [root.k * (const.c.value / electron_gyro_abs.value) for root in resonant_roots if not (np.isnan(root.omega) or np.isnan(root.k))]
    # y = [root.omega / electron_gyro_abs.value for root in resonant_roots if not (np.isnan(root.omega) or np.isnan(root.k))]
    # plt.semilogy(x, y, "ro")

    # Plot upper and lower cutoffs
    lower_upper_x = np.arange(-1, 25, 1)
    lower_y = [(omega_lc / electron_gyro_abs).value for val in lower_upper_x]
    upper_y = [(omega_uc / electron_gyro_abs).value for val in lower_upper_x]
    plt.semilogy(
        lower_upper_x,
        upper_y,
        "k-.",
        linewidth=0.8,
        label=r"$\frac{\omega_{uc}}{| \Omega_e |}$",
    )
    plt.semilogy(
        lower_upper_x,
        lower_y,
        "k:",
        linewidth=1.0,
        label=r"$\frac{\omega_{lc}}{| \Omega_e |}$",
    )

    plt.minorticks_on()
    plt.xticks(range(0, 21, 5))
    plt.yticks([0.1, 1.0], ["0.1", "1.0"])
    plt.yticks(np.arange(0.2, 1.0, 0.1), [], minor=True)
    plt.xlim(0.0, 20.0)
    plt.ylim(0.1, 1.0)
    plt.xlabel(r"$k \frac{c}{| \Omega_e |}$")
    plt.ylabel(r"$\frac{\omega}{| \Omega_e |}$")
    plt.legend(loc="lower right")
    plt.title(
        rf"$E={RKE}, \alpha={alpha.deg}^\circ, \psi={psi.to(u.deg).value:.2f}^\circ$"
    )
    plt.tight_layout()

    if save:
        plt.savefig(f"resonant_roots_{i:03d}.png", dpi=150)
        plt.close()
    else:
        plt.show()


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 * u.MeV
    alpha = Angle(83, u.deg)
    resonance = -1
    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    X_min = 0.0
    X_max = 1.0
    X_npoints = 101
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    # Dimensionless frequency range
    # To be scaled up by abs(electron gyrofrequency) when used.
    y_min = 0.1 * u.dimensionless_unscaled
    y_max = 1.0 * u.dimensionless_unscaled
    y_list = np.linspace(y_min, y_max, num=201)
    # =================================================

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr_sym = CpdrSymbolic(len(particles))
    cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

    print(f"Plasma: {particles}")
    print(f"Energy: {energy}")
    print(f"Pitch angle: {alpha}")
    print(f"Resonance: {resonance}")
    print(f"Electron plasma over gyro ratio: {plasma_over_gyro_ratio}")
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

    resonant_triplets = cpdr.solve_resonant(X_range)
    for i, X in enumerate(X_range):
        resonance_condition_kpar_pos = get_resonance_condition(
            cpdr, X, y_list, "positive"
        )
        resonance_condition_kpar_neg = get_resonance_condition(
            cpdr, X, y_list, "negative"
        )
        dispersion_relation = get_dispersion_relation(cpdr, X, y_list)
        plot_resonant_roots(
            cpdr,
            X,
            resonant_triplets[i],
            resonance_condition_kpar_pos,
            resonance_condition_kpar_neg,
            dispersion_relation,
            i,
            save=False,
        )


if __name__ == "__main__":
    main()
