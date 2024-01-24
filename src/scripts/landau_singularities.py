# python src/scripts/landau_singularities.py --rke 1.0 --alpha 5.0 [--save]
# where rke in MeV and alpha in degrees.
import math
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magfield import MagField
from piran.normalisation import solve_dispersion_relation
from piran.particles import Particles, PiranParticle
from piran.resonance import calc_lorentz_factor


def plot(
    x,
    y,
    rke,
    alpha,
    save,
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    plt.plot(x, y, "b")

    # # plt.minorticks_on()
    # plt.xticks(x_ticks, [str(v) for v in x_ticks])
    # plt.yticks(y_ticks, [str(v) for v in y_ticks])
    # plt.tick_params("x", which="both", top=True, labeltop=False)
    # plt.tick_params("y", which="both", right=True, labelright=False)
    # # plt.yticks(np.arange(0.2, 1.0, 0.1), [], minor=True)
    # plt.xlim(x_lim_min, x_lim_max)
    # plt.ylim(y_lim_min, y_lim_max)
    plt.xlabel(r"X")
    plt.ylabel(rf"$|v_{{||}} - \partial \omega / \partial k_{{||}}|$")
    plt.title(rf"$E$={rke} MeV, $\alpha$={alpha}$^{{\circ}}$")
    plt.tight_layout()

    filestem = f"E({rke})alpha({alpha})"
    if save:
        plt.savefig(f"{filestem}.png", dpi=150)
    else:
        plt.show()


def compute(RKE, alpha):
    frequency_ratio = 1.5 * u.dimensionless_unscaled
    omega_ratio = 0.1225

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

    Omega_e = (q_e * B) / const.m_e
    Omega_e_abs = abs(Omega_e)
    omega_pe = Omega_e_abs * frequency_ratio

    n_ = omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2
    Omega_p = (q_p * B) / const.m_p
    omega_pp = np.sqrt((n_ * q_p**2) / (const.eps0 * const.m_p))
    # =============================== END =============================== #

    # ============================== START ============================== #
    # We need those because they are input arguments to the new Cpdr class.
    # They are not needed for these tests.


    gamma = calc_lorentz_factor(RKE, const.m_e)
    v = const.c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha.rad)

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

    omega = abs(Omega_e) * omega_ratio

    piran_particle_list = (PiranParticle("e", n_), PiranParticle("H+", n_))
    cpdr_particles = Particles(piran_particle_list, RKE, alpha)
    cpdr_wave_angles = Gaussian(0, 1e10, 0, 0.577)
    cpdr_wave_freqs = Gaussian(omega_lc, omega_uc, omega_m, delta_omega)
    cpdr_mag_field = MagField()
    cpdr_resonances = n_range

    dispersion = Cpdr(
        cpdr_particles,
        cpdr_wave_angles,
        cpdr_wave_freqs,
        cpdr_mag_field,
        mlat,
        l_shell,
        cpdr_resonances,
    )

    dispersion.as_poly_in_k()

    X_min = 0.00
    X_max = 1.00
    X_npoints = 100
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    xwk_roots = solve_dispersion_relation(
        dispersion,
        (Omega_e, Omega_p),
        (omega_pe, omega_pp),
        omega,
        X_range,
    )

    results = np.empty(len(xwk_roots), dtype=np.float64)
    x_axis = np.empty(len(xwk_roots), dtype=np.float64)
    for ii, pair in enumerate(xwk_roots):
        X = pair[0]
        k = pair[2]

        dD_dk = dispersion.stix.dD_dk(omega, X, k / u.m).value
        dD_dw = dispersion.stix.dD_dw(omega, X, k / u.m).value

        results[ii] = v_par.value + (dD_dk / dD_dw) * math.sqrt(1 + X**2)
        x_axis[ii] = X

    return x_axis, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rke", required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    rke = args.rke * u.MeV  # Relativistic kinetic energy (Mega-electronvolts)
    alpha = Angle(args.alpha, u.deg)  # pitch angle in degrees

    x, y = compute(rke, alpha)
    plot(x, y, args.rke, args.alpha, args.save)


if __name__ == "__main__":
    main()
