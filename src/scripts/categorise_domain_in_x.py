import numpy as np
import sympy as sym
from astropy import units as u
from astropy.coordinates import Angle
from scipy.optimize import root_scalar

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


@u.quantity_input
def solve_resonant_for_x(
    cpdr: Cpdr,
    omega: u.Quantity[u.rad / u.s],
    X_range: u.Quantity[u.dimensionless_unscaled],
    verbose: bool = False,
) -> u.Quantity[u.dimensionless_unscaled]:
    roots = []
    for om in np.atleast_1d(omega):
        X = cpdr.symbolic.syms.get("X")
        psi = cpdr.symbolic.syms.get("psi")

        # Only psi is a symbol after this
        resonant_cpdr_in_psi = cpdr.resonant_poly_in_omega.subs(
            {X: sym.tan(psi), "omega": om.value}
        )

        # lambdify our func in psi
        resonant_cpdr_in_psi_lambdified = sym.lambdify(psi, resonant_cpdr_in_psi)

        # transform range in X to range in psi
        psi_range = np.arctan(X_range)

        # evaluate func for all psi and store sign of result
        cpdr_signs = np.sign(resonant_cpdr_in_psi_lambdified(psi_range))

        # We want to perform a pairwise comparison of consecutive elements and
        # look for a chance of sign (from 1 to -1 or vice versa).
        # We can do this efficiently by adding an ndarray containing the first
        # element of each pair to an ndarray containing the second element of
        # each pair.
        # Anywhere that the result is 0 indicates a change in sign!
        pairwise_sign_sums = cpdr_signs[:-1] + cpdr_signs[1:]

        # Find indices corresponding to changes of sign.
        # This is faster than looping over the whole pairwise_sign_sums
        # for large arrays.
        sign_change_indices = np.flatnonzero(pairwise_sign_sums == 0)

        # For each index where we have identified that a change of sign occurs,
        # use scipy's root_scalar to hone in on the root.
        for idx in sign_change_indices:
            root_result = root_scalar(
                resonant_cpdr_in_psi_lambdified,
                bracket=[psi_range[idx].value, psi_range[idx + 1].value],
                method="brentq",
            )
            roots.append(root_result.root)

            if verbose:
                print(
                    f"For {om=}\n"
                    f"Change of sign between psi = {psi_range[idx].to_value(u.deg)}, {psi_range[idx+1].to_value(u.deg)}\n"
                    f"Indices = {idx}, {idx+1}\n"
                    f"Root at: {root_result.root * 180 / np.pi}\n"
                )

    # Convert back to X and return
    return u.Quantity(np.tan(roots), u.dimensionless_unscaled)


def split_domain(X_min, X_max, splits):
    # Patition (X_min, X_max) into subdomains according to splits.
    # We could simplify this if splits already included X_min and X_max, which would
    # likely require them to be added during solve_resonant_for_x.
    domains = []
    if splits.size == 0:
        # No roots, so our whole domain is just [X_min, X_max]
        domains.append(u.Quantity([X_min, X_max], u.dimensionless_unscaled))
    else:
        # First subdomain is X_min to our smallest root...
        domains.append(u.Quantity([X_min, splits[0]], u.dimensionless_unscaled))

        # Grab all other subdomains in this loop
        # nd.iter returns ndarray elements and strips units :(
        it = np.nditer(splits)
        with it:
            while not it.finished:
                lower = it.value
                upper = it.value if (it.iternext()) else X_max
                domains.append(u.Quantity([lower, upper], u.dimensionless_unscaled))

    return domains


def count_roots_per_subdomain(
    cpdr: Cpdr,
    domains,
):
    # Now check how many roots exist in each subdomain.
    # We need to pick a point within each subdomain - midpoint will do!
    midpoints = u.Quantity(
        [(subdomain[0] + subdomain[1]) / 2 for subdomain in domains],
        u.dimensionless_unscaled,
    )
    domain_roots = cpdr.solve_resonant(midpoints)

    num_roots = []
    for subdomain in domain_roots:
        # Special case: if we have 1 'root', check for NaN!
        if len(subdomain) == 1 and subdomain[0].count(np.nan):
            num_roots.append(0)
            continue

        # Failing either of the above checks, our number of roots is just equal
        # to the number of (X, omega, k) tuples in our subdomain list
        num_roots.append(len(subdomain))

    return num_roots


def main():
    # ================ Parameters =====================
    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 * u.MeV
    alpha = Angle(70.84, u.deg)
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
    cpdr_sym = CpdrSymbolic(len(particles))
    cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

    X_l = solve_resonant_for_x(cpdr, cpdr.omega_lc, X_range)
    X_u = solve_resonant_for_x(cpdr, cpdr.omega_uc, X_range)
    X_all = np.sort(
        solve_resonant_for_x(cpdr, u.Quantity([cpdr.omega_uc, cpdr.omega_lc]), X_range)
    )

    print(f"{X_l=}\n" f"{X_u=}\n" f"{X_all=}\n")

    # Split the domain (X_min, X_max) into a list of subdomains, partitioned by X_all
    domains = split_domain(X_min, X_max, X_all)
    print(f"Subdomains: {domains}\n")

    # Count roots in each subdomain
    num_roots = count_roots_per_subdomain(cpdr, domains)
    print(f"Roots per subdomain: {num_roots}\n")


if __name__ == "__main__":
    main()
