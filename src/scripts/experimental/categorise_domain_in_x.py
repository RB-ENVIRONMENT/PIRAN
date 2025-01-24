# NOT WORKING AS OF a14e5c2ba25e1d9c2740864b904e63b02e7562d7
#
# This relies on the experimental 'meshing' module, which does not work now that
# SymPy has been removed from the codebase (but could be fixed).

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.experimental.meshing import (
    count_roots_per_bucket,
    solve_resonant_for_x,
    split_array,
)
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 * u.MeV
    alpha = Angle(70.8, u.deg)
    resonance = 0
    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    X_min = 0.0
    X_max = 1
    X_npoints = 1001
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )
    # =================================================

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr = Cpdr(plasma_point, energy, alpha, resonance, freq_cutoff_params)

    X_l = solve_resonant_for_x(cpdr, cpdr.omega_lc, X_range)
    X_u = solve_resonant_for_x(cpdr, cpdr.omega_uc, X_range)
    X_all = solve_resonant_for_x(
        cpdr, u.Quantity([cpdr.omega_uc, cpdr.omega_lc]), X_range, True
    )

    print(f"{X_l=}\n{X_u=}\n{X_all=}\n")

    # Split the array X_all into a list of distinct buckets
    buckets = split_array(X_all)
    print(f"Buckets: {buckets}\n")

    # Count roots in each subdomain
    num_roots = count_roots_per_bucket(cpdr, buckets)
    print(f"Roots per bucket: {num_roots}\n")


if __name__ == "__main__":
    main()
