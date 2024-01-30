import math
import time

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from scipy.optimize import root_scalar

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magfield import MagField
from piran.particles import Particles, PiranParticle
from piran.resonance import (
    calc_lorentz_factor,
    get_valid_roots,
    poly_solver,
    replace_cpdr_symbols,
)


def find_resonant_triplets(
    dispersion, v_par, gamma, gyro_freq, plasma_freq, cutoff_freq, n, X
):
    """
    This is similar to compute_root_pairs() from resonance module but modified
    to work for single resonance number and single X.
    Given `n` and `X=tan(psi)`, simultaneously solve the resonance condition
    and the dispersion relation to get root pairs of wave frequency
    `omega` and wave number `k`.
    """
    Omega_e, Omega_p = gyro_freq
    omega_pe, omega_pp = plasma_freq
    omega_lc, omega_uc = cutoff_freq

    CPDR_omega, _ = dispersion.as_resonant_poly_in_omega()
    CPDR_k, _ = dispersion.as_poly_in_k()

    values_dict = {
        "v_par": v_par.value,
        "gamma": gamma.value,
        "Omega": (Omega_e.value, Omega_p.value),
        "omega_p": (omega_pe.value, omega_pp.value),
    }

    # X, psi, omega and n are still symbols after this
    CPDR_omega2 = replace_cpdr_symbols(CPDR_omega, values_dict)

    # X, k and omega are still symbols after this
    CPDR_k2 = replace_cpdr_symbols(CPDR_k, values_dict)

    psi = math.atan(X) * u.rad

    values_dict2 = {
        "X": X,
        "psi": psi.value,
        "n": n.value,
    }

    # Only omega is a symbol after this
    CPDR_omega3 = replace_cpdr_symbols(CPDR_omega2, values_dict2)

    # Only k and omega are symbols after this
    CPDR_k3 = replace_cpdr_symbols(CPDR_k2, values_dict2)

    # Solve modified CPDR to obtain omega roots for given X
    omega_l = poly_solver(CPDR_omega3)

    # Categorise roots
    # Keep only real, positive and within bounds
    valid_omega_l = get_valid_roots(omega_l)
    valid_omega_l = [x for x in valid_omega_l if omega_lc.value <= x <= omega_uc.value]

    # If valid_omega_l is empty continue
    if len(valid_omega_l) == 0:
        return None

    # We expect at most 1 real positive root
    if len(valid_omega_l) > 1:
        msg = (
            f"n={n.value} X={X.value} We got more than one real positive root for omega"
        )
        # raise ValueError(msg)
        print(msg)
        return None

    # At this point valid_omega_l will contain only one element
    valid_omega = valid_omega_l[0]

    # Substitute omega into CPDR
    CPDR_k4 = replace_cpdr_symbols(CPDR_k3, {"omega": valid_omega})

    # Solve unmodified CPDR to obtain k roots for given X, omega
    k_l = poly_solver(CPDR_k4)

    # Keep only real and positive roots
    valid_k_l = get_valid_roots(k_l)

    # If valid_k_l is empty continue
    if valid_k_l.size == 0:
        return None

    # We expect at most 1 real positive root
    if valid_k_l.size > 1:
        msg = f"n={n.value} X={X.value} omega={valid_omega} We got more than one real positive root for k"
        # raise ValueError(msg)
        print(msg)
        return None

    # Note: At this point valid_k_l will contain only one element
    valid_k = valid_k_l[0]

    return valid_omega, valid_k


def f(
    x,
    dispersion,
    v_par,
    gamma,
    gyro_freq,
    plasma_freq,
    cutoff_freq,
    n,
    dD_dw_lambif,
    dD_dk_lambif,
):
    """
    Function that given x returns the value of `v_par - dw/dk_par`.
    We will pass this to a root solver.
    """
    omega, k = find_resonant_triplets(
        dispersion, v_par, gamma, gyro_freq, plasma_freq, cutoff_freq, n, x
    )
    return v_par.value + (
        dD_dk_lambif(x, omega, k) / dD_dw_lambif(x, omega, k)
    ) * math.sqrt(1 + x**2)


def main():
    frequency_ratio = 1.5 * u.dimensionless_unscaled
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

    n = -2 * u.dimensionless_unscaled
    RKE = 1.0 * u.MeV  # Relativistic kinetic energy (Mega-electronvolts)
    alpha = Angle(66, u.deg)  # pitch angle

    # Calculate the Lorentz factor and particle velocity using input params
    gamma = calc_lorentz_factor(RKE, const.m_e)
    v = const.c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha.rad)  # Is this correct?

    # Lower and upper cut-off frequencies
    omega_m = 0.35 * Omega_e_abs
    delta_omega = 0.15 * Omega_e_abs
    omega_lc = omega_m - 1.5 * delta_omega
    omega_uc = omega_m + 1.5 * delta_omega

    # Resonances
    # NOTE we need those only because they are input argument to Cpdr() class
    n_min = -5
    n_max = 5
    n_range = u.Quantity(
        range(n_min, n_max + 1), unit=u.dimensionless_unscaled, dtype=np.int32
    )

    X_min = 0.4
    X_max = 0.6
    # X_npoints = 41
    # X_range = u.Quantity(
    #     np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    # )

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
        cpdr_resonances,  # NOTE ??
    )

    dispersion.as_poly_in_k()

    values_dict = {
        "Omega": (Omega_e.value, Omega_p.value),
        "omega_p": (omega_pe.value, omega_pp.value),
    }

    dispersion_poly_k = replace_cpdr_symbols(dispersion._poly_k, values_dict)
    dD_dw_sym = dispersion_poly_k.diff("omega")
    dD_dk_sym = dispersion_poly_k.diff("k")

    dD_dw_lambif = sym.lambdify(
        ["X", "omega", "k"],
        dD_dw_sym,
        "numpy",
    )

    dD_dk_lambif = sym.lambdify(
        ["X", "omega", "k"],
        dD_dk_sym,
        "numpy",
    )

    gyro_freq = (Omega_e, Omega_p)
    plasma_freq = (omega_pe, omega_pp)
    cutoff_freq = (omega_lc, omega_uc)

    # Find Landau singularities (roots of `v_par - dw/dk_par`)
    for method in ["bisect", "brentq", "brenth", "ridder", "toms748"]:
        print(f"Method: {method}")
        start = time.perf_counter()
        root_results = root_scalar(
            f,
            args=(
                dispersion,
                v_par,
                gamma,
                gyro_freq,
                plasma_freq,
                cutoff_freq,
                n,
                dD_dw_lambif,
                dD_dk_lambif,
            ),
            bracket=[X_min, X_max],
            method=method,
        )
        end = time.perf_counter()
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.6f} seconds")

        X = root_results.root
        omega, k = find_resonant_triplets(
            dispersion, v_par, gamma, gyro_freq, plasma_freq, cutoff_freq, n, X
        )
        landau = f(
            X,
            dispersion,
            v_par,
            gamma,
            gyro_freq,
            plasma_freq,
            cutoff_freq,
            n,
            dD_dw_lambif,
            dD_dk_lambif,
        )

        print(root_results)
        print(f"{X=:.9f} - {omega=:.5e} - {k=:.5e} - {landau=:12.4e}")
        print()

    # for X in X_range:
    #     omega, k = find_resonant_triplets(dispersion, v_par, gamma, gyro_freq, plasma_freq, cutoff_freq, n, X)
    #     landau = f(X, dispersion, v_par, gamma, gyro_freq, plasma_freq, cutoff_freq, n, dD_dw_lambif, dD_dk_lambif)
    #     print(f"{X=:.3f} - {omega=:.4e} - {k=:.4e} - {landau=:12.4e}")


if __name__ == "__main__":
    main()
