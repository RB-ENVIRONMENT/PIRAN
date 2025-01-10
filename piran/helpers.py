"""
The `helpers` module provides a collection of utility functions for physics
calculations, particularly related to relativistic particle dynamics and
root finding.

This module includes functions for calculating:

- The Lorentz factor :math:`\\gamma` (`calc_lorentz_factor`).
- The momentum of a particle (`calc_momentum`).
- Filtering real and positive roots from a set of roots (`get_real_and_positive_roots`).

These functions are intended to be used within other modules of the package.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity


@u.quantity_input
def calc_lorentz_factor(
    E: Quantity[u.Joule],
    m: Quantity[u.kg],
) -> float:
    """
    Calculate the Lorentz factor (:math:`\\gamma`) from the relativistic
    kinetic energy (:math:`E_k`) and rest mass (:math:`m_0`).

    The Lorentz factor is calculated as:

    .. math::
        \\gamma = 1 + \\frac{E_k}{m_0 c^2}

    where :math:`c` is the speed of light.

    This function calculates the Lorentz factor using the relativistic
    kinetic energy, which is defined as the difference between the total
    relativistic energy and the rest mass energy:

    .. math::
        E_k = E_t - E_0 = (\\gamma - 1)m_0 c^2

    Notes
    -----
    This calculation differs from the "standard" calculation of the Lorentz
    factor using velocity :math:`v`:

    .. math::
        \\gamma = \\frac{1}{\\sqrt{1 - v^2/c^2}}

    as implemented in packages like `plasmapy`. This function is specifically
    designed for cases where the relativistic kinetic energy is known.

    Parameters
    ----------
    E : astropy.units.Quantity[u.Joule]
        The relativistic kinetic energy of the particle. Must have units
        convertible to Joules.
    m : astropy.units.Quantity[u.kg]
        The rest mass of the particle. Must have units convertible to kilograms.

    Returns
    -------
    float
        The Lorentz factor :math:`\\gamma`, a dimensionless quantity.

    See Also
    --------
    plasmapy.formulary.relativity.Lorentz_factor : The standard Lorentz factor calculation using velocity.

    Examples
    --------
    >>> import astropy.units as u
    >>> energy = 1.0 * u.MeV
    >>> mass = 9.1093837015e-31 * u.kg
    >>> gamma = calc_lorentz_factor(energy, mass)
    """
    return (E.to(u.Joule) / (m.to(u.kg) * const.c**2)) + 1


@u.quantity_input
def calc_momentum(
    gamma: Quantity[u.dimensionless_unscaled],
    mass: Quantity[u.kg],
) -> Quantity[u.kg * u.m / u.s]:
    """
    Calculate the relativistic momentum of a particle given its Lorentz factor
    (:math:`\\gamma`) and rest mass (:math:`m_0`).

    The relativistic momentum (:math:`p`) is given by:

    .. math::
        p = \\sqrt{\\gamma^2 - 1}m_0c

    where :math:`c` is the speed of light, as shown in Glauert & Horne (2005), Equation 18.

    Parameters
    ----------
    gamma : float
        The Lorentz factor, a dimensionless quantity.
    mass : astropy.units.Quantity[u.kg]
        The rest mass of the particle. Must have units convertible to kilograms.

    Returns
    -------
    astropy.units.Quantity[u.kg * u.m / u.s]
        The relativistic momentum of the particle, with units of kg m/s.

    Examples
    --------
    >>> import astropy.units as u
    >>> gamma = 2.9569511835738735
    >>> mass = 9.1093837015e-31 << u.kg
    >>> momentum = calc_momentum(gamma, mass)
    """
    return np.sqrt(gamma**2 - 1) * mass * const.c


def get_real_and_positive_roots(values, tol=1e-8):
    """
    Filter a sequence of values (real or complex), returning only the real
    and positive values that are greater than the specified tolerance.

    Complex numbers with imaginary parts close to zero are treated as real
    numbers, and their real parts are included in the result if they meet
    the other criteria (positive and greater than `tol`).

    Parameters
    ----------
    values : array_like
        A sequence (e.g., list, tuple, NumPy array) containing the values to be filtered.
    tol : float, optional
        The tolerance below which values are considered zero and thus excluded.
        We compare only the real part if the input is a complex number.
        Defaults to 1e-8.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing only the real and positive values from the
        input that are greater than `tol`. Returns an empty NumPy array if no
        values meet the criteria.

    Notes
    -----
    This function uses `numpy.isclose` with the default tolerances to determine
    if a value (the imaginary part) is close to zero. See the documentation of
    `numpy.isclose` for details on how the comparison is performed.

    Examples
    --------
    >>> values = [-1 + 0j, 1.1 + 0.00000001j, 100 + 2j]
    >>> valid_roots = get_real_and_positive_roots(values)
    >>> print(valid_roots)
    [1.1]
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
