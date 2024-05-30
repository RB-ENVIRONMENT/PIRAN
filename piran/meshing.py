from typing import List

import numpy as np
import sympy as sym
from astropy import units as u
from cpdr import Cpdr
from scipy.optimize import root_scalar


@u.quantity_input
def split_array(
    array: u.Quantity[u.dimensionless_unscaled],
) -> List[u.Quantity[u.dimensionless_unscaled]]:
    """
    Partition the array [a, b, c, ... , x, y, z] into a list of distinct buckets of the
    form [[a, b], [b, c], ... , [x, y], [y, z]].

    Parameters
    ----------
    array: u.Quantity[u.dimensionless_unscaled]
        An ordered list of values to be used as endpoints for distinct buckets.

    Returns
    -------
    subdomains : List[u.Quantity[u.dimensionless_unscaled]]
        A list of distinct buckets in the form [[a, b], [b, c], ... , [x, y], [y, z]].
    """
    buckets = []

    # If input array consists of 0 or 1 points, we have nothing to do...
    if array.size < 2:
        return array

    # Iterate over pairwise elements of the array and create distinct buckets.
    # nd.iter returns ndarray elements and strips units :(
    it = np.nditer(array)
    with it:
        while not it.finished:
            lower = it.value

            if it.iternext():
                upper = it.value
                buckets.append(u.Quantity([lower, upper], u.dimensionless_unscaled))

    return buckets


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


@u.quantity_input
def solve_resonant_for_x(
    cpdr: Cpdr,
    omega: u.Quantity[u.rad / u.s],
    X_range: u.Quantity[u.dimensionless_unscaled],
    endpoints: bool = False,
    verbose: bool = False,
) -> u.Quantity[u.dimensionless_unscaled]:
    """
    Given Cpdr object and a 0d/1d array of omega, solve the resonant cpdr for each omega
    to obtain solutions in X. It is not possible to know how many solutions in X are
    present within a given range _a priori_, so an initial discretisation in X is
    required as an input argument. A 'coarse' discretisation may not catch every
    possible solution!

    Typical usage: let omega be lower / upper frequency cutoffs so that this will return
    the values of X at which new solutions to the resonant cpdr enter / exit the region
    of interest bounded by [omega_lc, omega_uc].

    Parameters
    ----------
    cpdr:
        A Cpdr object.
    omega: u.Quantity[u.rad / u.s]
        A 0d/1d array of values in omega, for which we would like to solve the resonant
        Cpdr to find corresponding solutions in X.
    X_range: u.Quantity[u.rad / u.s]
        An initial discretisation in X. For each omega, we produce values for the
        resonant cpdr for all X in X_range and look for changes in sign (indicating the
        presence of a root). A root finding algorithm then determines the precise
        location of the root.
    endpoints: bool
        Controls whether endpoints of `X_range` should be included in the return value.
        This should be `True` if used in conjunction with the `split_domain` and
        `count_roots_per_subdomain` methods from the `helpers` module to partition a
        domain in `X` according to Glauert&Horne 2005, Paragraph 23.
    verbose: bool
        Controls print statements.

    Returns
    -------
    u.Quantity[u.dimensionless_unscaled]
        A (flat) list of solutions in X.
    """

    # transform range in X to range in psi
    psi_range = np.arctan(X_range)

    # Grab symbols for X and psi
    X = cpdr.symbolic.syms.get("X")
    psi = cpdr.symbolic.syms.get("psi")

    roots = []

    for om in np.atleast_1d(omega):
        # resonant cpdr contains both `X` and `psi` terms, which are linked by
        # `X = tan(psi)`. We're looking for solns in `X`, but its easier to find
        # solns in `psi`.
        #
        # Transform all `X` in resonant cpdr to `psi`, and substitute in the current
        # fixed value for `omega`. Only `psi` is a symbol after this.
        resonant_cpdr_in_psi = cpdr.resonant_poly_in_omega.subs(
            {X: sym.tan(psi), "omega": om.value}
        )

        # lambdify our func in psi
        resonant_cpdr_in_psi_lambdified = sym.lambdify(psi, resonant_cpdr_in_psi)

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

    # Finalise solution array, including endpoints if specified in input args.
    # Append u.dimensionless_unscaled units in either case.
    solns_in_X = (
        [X_range[0], np.tan(roots), X_range[-1]] if endpoints else np.tan(roots)
    )
    solns_in_X <<= u.dimensionless_unscaled

    return solns_in_X
