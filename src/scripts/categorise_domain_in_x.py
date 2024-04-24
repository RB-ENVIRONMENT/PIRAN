from typing import List

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
    """
    Given Cpdr object and a 0d/1d array of omega, solve resonant cpdr for each omega.
    Typical usage: let omega be lower / upper frequency cutoffs so that this will return
    the values of X at which new solutions to the resonant cpdr enter / exit the region
    of interest bounded by [omega_lc, omega_uc].

    Parameters
    ----------
    cpdr: Cpdr
        A Cpdr object.
    omega: u.Quantity[u.rad / u.s]
        A 0d/1d array of values in omega, for which we would like to solve the resonant
        Cpdr to find corresponding solutions in X.
    X_range: u.Quantity[u.rad / u.s]
        An initial discretisation in X. For each omega, we produce values for the
        resonant cpdr for all X in X_range and look for changes in sign (indicating the
        presence of a root). A root finding algorithm then determines the precise
        location of the root.
    verbose: bool
        Controls print statements.

    Returns
    -------
    u.Quantity[u.dimensionless_unscaled]
        A (flat) list of solutions in X.
    """

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
        # look for a change of sign (from 1 to -1 or vice versa).
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


@u.quantity_input
def split_domain(
    X_min: float, X_max: float, splits: u.Quantity[u.dimensionless_unscaled]
) -> List[u.Quantity[u.dimensionless_unscaled]]:
    """
    Patition the domain [X_min, X_max] into subdomains according to splits.
    We could simplify this if splits already included X_min and X_max, which would
    likely require them to be added during solve_resonant_for_x.

    Parameters
    ----------
    X_min : float
        Lower bound
    X_max : float
        Upper bound
    splits: u.Quantity[u.dimensionless_unscaled]
        Values between [X_min, X_max] to be used for partitioning the domain.

    Returns
    -------
    subdomains : List[u.Quantity[u.dimensionless_unscaled]]
        A list of subdomains in the form [[X_min, a], [a, b], [b, c], ... , [z, X_max]].
    """
    subdomains = []
    if splits.size == 0:
        # No roots, so our whole domain is just [X_min, X_max]
        subdomains.append(u.Quantity([X_min, X_max], u.dimensionless_unscaled))
    else:
        # First subdomain is X_min to our smallest root...
        subdomains.append(u.Quantity([X_min, splits[0]], u.dimensionless_unscaled))

        # Grab all other subdomains in this loop
        # nd.iter returns ndarray elements and strips units :(
        it = np.nditer(splits)
        with it:
            while not it.finished:
                lower = it.value
                upper = it.value if (it.iternext()) else X_max
                subdomains.append(u.Quantity([lower, upper], u.dimensionless_unscaled))

    return subdomains


def count_roots_per_subdomain(
    cpdr: Cpdr,
    domains: List[u.Quantity[u.dimensionless_unscaled]],
) -> List[float]:
    """
    Check how many roots exist in each subdomain. Note that this only samples from two
    points within each subdomain (near the endpoints), so is not an exhaustive check!
    For subdomain without a fixed number of roots (likely indicating a singularity),
    this returns np.nan for that subdomain.

    Parameters
    ----------
    cpdr : Cpdr
        A Cpdr object.
    domains: List[u.Quantity[u.dimensionless_unscaled]]
        A list of subdomains (see func split_domain).

    Returns
    -------
    List[float]
        The (fixed?) number of roots within each subdomain. Note: we use `float` instead
        of `int` since np.nan is `float`.
    """
    num_roots = []
    for subdomain in domains:
        left_roots = cpdr.solve_resonant(subdomain[0] * (1 + 1e-4))[0]
        right_roots = cpdr.solve_resonant(subdomain[1] * (1 - 1e-4))[0]

        num_left_roots = len(left_roots)
        num_right_roots = len(right_roots)

        # First, check the number of roots are equal at left and right endpoints of
        # subdomain. If not, uh-oh...
        if num_left_roots != num_right_roots:
            print(
                f"Roots not fixed in {subdomain=}\n"
                f"{num_left_roots=}\n"
                f"{num_right_roots=}\n"
            )
            num_roots.append(np.nan)
            continue

        # Special case: if we have 1 'root', check for NaN!
        if num_left_roots == 1:
            left_root_is_nan = bool(left_roots[0].count(np.nan))
            right_root_is_nan = bool(right_roots[0].count(np.nan))

            if left_root_is_nan and right_root_is_nan:
                num_roots.append(0)
            elif not (left_root_is_nan or right_root_is_nan):
                num_roots.append(1)
            else:
                print(
                    f"Roots not fixed in {subdomain=}\n"
                    f"{num_left_roots=}\n"
                    f"{num_right_roots=}\n"
                )
                num_roots.append(np.nan)

        # Regular case: number of roots is equal to the number of (X, omega, k)
        # tuples in current subdomain.
        else:
            num_roots.append(num_left_roots)

    return num_roots


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
