import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr2 import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 * u.MeV
    alpha = Angle(76, u.deg)
    resonance = 0
    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    X_min = 0.0
    X_max = 1.0
    X_npoints = 101
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )
    # =================================================

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr_sym = CpdrSymbolic(len(particles))
    cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

    X_l = cpdr.solve_resonant_for_x(cpdr.omega_lc, X_range)
    X_u = cpdr.solve_resonant_for_x(cpdr.omega_uc, X_range)
    X_all = np.sort(
        cpdr.solve_resonant_for_x(u.Quantity([cpdr.omega_uc, cpdr.omega_lc]), X_range)
    )

    print(f"{X_l=}\n" f"{X_u=}\n" f"{X_all=}\n")

    # Split X_all into subdomains, stored within a domains list.
    # We could simplify this if X_all already included X_min and X_max, which would
    # likely require them to be added during solve_resonant_for_x.
    domains = []
    if X_all.size == 0:
        # No roots, so our whole domain is just [X_min, X_max]
        domains.append(u.Quantity([X_min, X_max], u.dimensionless_unscaled))
    else:
        # First subdomain is X_min to our smallest root...
        domains.append(u.Quantity([X_min, X_all[0]], u.dimensionless_unscaled))

        # Grab all other subdomains in this loop
        # nd.iter returns ndarray elements and strips units :(
        it = np.nditer(X_all)
        with it:
            while not it.finished:
                lower = it.value
                upper = it.value if (it.iternext()) else X_max
                domains.append(u.Quantity([lower, upper], u.dimensionless_unscaled))

    print(f"Subdomains: {domains}\n")

    # Now check how many roots exist in each subdomain
    midpoints = u.Quantity(
        [(subdomain[0] + subdomain[1]) / 2 for subdomain in domains],
        u.dimensionless_unscaled,
    )
    domain_roots = cpdr.solve_resonant(midpoints)
    print(f"Subdomain roots (at midpoints): {domain_roots}\n")

    num_roots = []
    for subdomain in domain_roots:

        # Special case: if we have 1 'root', check for NaN!
        if len(subdomain) == 1 and subdomain[0].count(np.nan):
            num_roots.append(0)
            continue

        # Failing either of the above checks, our number of roots is just equal
        # to the number of (X, omega, k) tuples in our subdomain list
        num_roots.append(len(subdomain))

    print(f"Roots per subdomain: {num_roots}\n")


if __name__ == "__main__":
    main()
