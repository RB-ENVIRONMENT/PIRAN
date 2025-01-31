import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson, trapezoid

from piran.cpdr import Cpdr
from piran.gauss import Gaussian

UNIT_NF = u.s / u.m**3


@u.quantity_input
def compute_glauert_norm_factor(
    cpdr: Cpdr,
    omega: u.Quantity[u.rad / u.s],
    X_range: u.Quantity[u.dimensionless_unscaled],
    wave_norm_angle_dist: Gaussian,
    method="simpson",
) -> u.Quantity[UNIT_NF]:
    """
    Calculate the normalisation factor from
    Glauert & Horne 2005 (equation 15).

    Parameters
    ----------
    cpdr : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    omega : astropy.units.quantity.Quantity convertible to rad/second
        Wave frequency.
    X_range : astropy.units.quantity.Quantity (dimensionless_unscaled)
        Wave normal angles.
    wave_norm_angle_dist : piran.gauss.Gaussian
        Distribution of wave normal angles.
    method : str, default="simpson"
        A string representing the integration method. The valid options
        are "trapezoid" and "simpson".

    Returns
    -------
    norm_factor : astropy.units.quantity.Quantity[UNIT_NF]
    """
    # Given omega and X_range calculate wave number k,
    # solution to the dispersion relation (while replacing NaN with 0)
    wave_numbers = np.nan_to_num(cpdr.solve_cpdr_for_norm_factor(omega, X_range), False)

    eval_gx = wave_norm_angle_dist.eval(X_range)

    evaluated_integrand = (
        eval_gx
        * wave_numbers
        * X_range
        * np.abs(cpdr.stix.jacobian(omega, X_range, wave_numbers).value)
    ) / ((1 + X_range**2) ** (1 / 2))

    # `simpson` returns a float
    # `trapezoid` returns a dimensionless `Quantity`
    # Not sure why they behave differently, but we need `.value` on `trapezoid` to avoid
    # a runtime error when trying to add units later.
    if method == "trapezoid":
        integral = trapezoid(evaluated_integrand, x=X_range).value
    elif method == "simpson":
        integral = simpson(evaluated_integrand, x=X_range)
    else:
        raise ValueError(f"Wrong integration rule: {method}")

    norm_factor = integral * (1 / (2 * np.pi**2))

    return norm_factor << UNIT_NF


@u.quantity_input
def compute_cunningham_norm_factor(
    cpdr: Cpdr,
    omega: u.Quantity[u.rad / u.s],
    X_range: u.Quantity[u.dimensionless_unscaled],
) -> u.Quantity[UNIT_NF]:
    """
    Calculate the normalisation factor from
    Cunningham 2023 (denominator of equation 4b).

    Parameters
    ----------
    cpdr : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    omega : astropy.units.quantity.Quantity convertible to rad/second
        Wave frequency.
    X_range : astropy.units.quantity.Quantity (dimensionless_unscaled)
        Wave normal angles.

    Returns
    -------
    norm_factor : astropy.units.quantity.Quantity[UNIT_NF]
    """
    # Given omega and X_range calculate wave number k,
    # solution to the dispersion relation (while replacing NaN with 0 in-place)
    wave_numbers = np.nan_to_num(cpdr.solve_cpdr_for_norm_factor(omega, X_range), False)

    norm_factor = (
        wave_numbers
        * X_range
        * np.abs(cpdr.stix.jacobian(omega, X_range, wave_numbers))
    ) / (2 * Angle(np.pi, u.rad) ** 2 * (1 + X_range**2) ** (1 / 2))

    return norm_factor
