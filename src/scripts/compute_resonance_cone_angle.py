# Script to compute the square root of the resonance cone angle
# (paragraph 23 from Glauert & Horne 2005).
# Takes as input arguments omega and frequency ratios
# with the `--omega_ratio` and `--frequency_ratio` respectively.
# e.g. `python compute_resonance_cone_angle.py
#         --omega_ratio 0.125
#         --frequency_ratio 1.5`
import argparse
import math
import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle


def get_sqrt_res_cone_angle_sym():
    num_particles = 2
    i = sym.Idx("i", num_particles)

    omega = sym.symbols("omega")

    Omega = sym.IndexedBase("Omega")
    Omega_i = Omega[i]

    omega_p = sym.IndexedBase("omega_p")
    omega_p_i = omega_p[i]

    idx = Omega_i.indices[0]

    PS_RANGE = (idx, idx.lower, idx.upper)

    R = 1 - sym.summation((omega_p_i**2) / (omega * (omega + Omega_i)), PS_RANGE)
    L = 1 - sym.summation((omega_p_i**2) / (omega * (omega - Omega_i)), PS_RANGE)
    P = 1 - sym.summation((omega_p_i**2) / (omega**2), PS_RANGE)
    S = (R + L) / 2

    return sym.sqrt(-P / S)


def main():
    parser = argparse.ArgumentParser(
        description="Compute square root of resonance cone angle.",
    )
    parser.add_argument("--omega_ratio", required=True)
    parser.add_argument("--frequency_ratio", required=True)
    args = parser.parse_args()

    frequency_ratio = float(args.frequency_ratio)
    omega_ratio = float(args.omega_ratio)

    q_e = -const.e.si  # Signed electron charge
    q_p = const.e.si  # Signed proton charge

    # Magnetic field
    M = 8.033454e15 * (u.tesla * u.m**3)
    mlat = Angle(0, u.deg)
    l_shell = 4.5 * u.dimensionless_unscaled  # Could absorb const.R_earth into this?
    B = (M * math.sqrt(1 + 3 * math.sin(mlat.rad) ** 2)) / (
        l_shell**3 * const.R_earth**3 * math.cos(mlat.rad) ** 6
    )

    # Gyrofrequency can be calculated directly using electron charge, mass, and
    # the magnetic field.
    Omega_e = (q_e * B) / const.m_e

    # Application of the frequency ratio yields the electron plasma frequency.
    Omega_e_abs = abs(Omega_e)
    omega_pe = Omega_e_abs * frequency_ratio

    # Proton gyro- and plasma- frequencies
    # (including particle number densities)
    # We assume that the number density of electrons and protons are equal.
    # These can be derived from the electron plasma frequency.
    n_ = omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2

    Omega_p = (q_p * B) / const.m_p
    omega_pp = np.sqrt((n_ * q_p**2) / (const.eps0 * const.m_p))

    omega_val = Omega_e_abs * omega_ratio

    values_dict = {
        "Omega": (Omega_e.value, Omega_p.value),
        "omega_p": (omega_pe.value, omega_pp.value),
        "omega": omega_val.value,
    }

    sqrt_res_cone_angle_sym = get_sqrt_res_cone_angle_sym()
    sqrt_res_cone_angle_val = sqrt_res_cone_angle_sym.subs(values_dict)
    print(f"The square root of the resonance cone angle is {sqrt_res_cone_angle_val}")


if __name__ == "__main__":
    main()
