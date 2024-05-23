from typing import List

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity

from piran.cpdr import Cpdr


@u.quantity_input
def calc_lorentz_factor(
    E: Quantity[u.Joule],
    m: Quantity[u.kg],
) -> float:
    """
    Calculate the Lorentz factor gamma for a given particle species given the
    relativistic kinetic energy and rest mass.
    Relativistic kinetic energy = Total relativistic energy - Rest mass energy
    RKE = TRE - RME = (gamma - 1) * m_0 * c^2
    Inputs:
        E: Joule (Relativistic kinetic energy)
        m: kg    (Rest mass)
    Returns:
        gamma: unitless (Lorentz factor)

    Note that this is different from plasmapy's `Lorentz_factor` which provides the
    'standard' way of calculating the Lorentz factor using the relative velocity `v`.
    """
    return (E.to(u.Joule) / (m.to(u.kg) * const.c**2)) + 1


@u.quantity_input
def calc_momentum(
    gamma: Quantity[u.dimensionless_unscaled],
    mass: Quantity[u.kg],
) -> Quantity[u.kg * u.m / u.s]:
    """
    Calculate the relativistic momentum for a given particle species given the
    Lorentz factor gamma and rest mass (Glauert & Horne 2005 Eq. 18).

    Parameters
    ----------
        gamma: unitless (Lorentz factor)
        mass:  kg       (Rest mass)

    Returns
    -------
        momentum: kg*m/s (Relativistic momentum)
    """
    return np.sqrt(gamma**2 - 1) * mass * const.c


def get_real_and_positive_roots(values, tol=1e-8):
    """
    Filter roots based on a condition (e.g real and >tol)

    Note: check default tolerance in np.isclose()
    """
    real_part = np.real(values)
    real_part_greater_zero = np.greater(real_part, tol)

    imag_part = np.imag(values)
    imag_part_almost_zero = np.isclose(imag_part, 0.0)

    vals_where_both_true = values[
        np.logical_and(real_part_greater_zero, imag_part_almost_zero)
    ]

    # Return only the real part (the imaginary part is close to zero anyways)
    return np.real(vals_where_both_true)


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
