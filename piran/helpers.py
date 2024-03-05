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
    return (E.to(u.J) / (m.to(u.kg) * const.c**2)) + 1


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
