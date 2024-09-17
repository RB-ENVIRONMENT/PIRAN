import numpy as np
import sympy as sym
from astropy import units as u
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

    Returns
    -------
    norm_factor : astropy.units.quantity.Quantity[UNIT_NF]
    """
    # Given omega and X_range calculate wave number k,
    # solution to the dispersion relation.
    # We could add units here, but we'd only have to strip them further down.
    wave_numbers = cpdr.solve_cpdr_for_norm_factor(omega, X_range)  # << u.rad / u.m

    # It is more performant to substitute omega here, since it is the
    # same for all root pairs/triplets, and then lambdify the expression outside
    # the loop and use the lambdified object within the loop to replace X and k.
    values_dict = {"omega": omega.value}

    # Derivative in omega
    cpdr_domega_lamb = sym.lambdify(
        ["X", "k"], cpdr.poly_in_k_domega.subs(values_dict), "numpy"
    )

    # Derivative in k
    cpdr_dk_lamb = sym.lambdify(
        ["X", "k"], cpdr.poly_in_k_dk.subs(values_dict), "numpy"
    )

    eval_gx = wave_norm_angle_dist.eval(X_range)

    evaluated_integrand = np.zeros_like(X_range, dtype=np.float64)
    for i in range(evaluated_integrand.shape[0]):
        X = X_range[i]
        k = wave_numbers[i]

        evaluated_integrand[i] = (
            eval_gx[i] * k.value**2 * np.abs(cpdr_domega_lamb(X.value, k.value)) * X
        ) / ((1 + X**2) ** (3 / 2) * np.abs(cpdr_dk_lamb(X.value, k.value)))

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
    # solution to the dispersion relation.
    # We could add units here, but we'd only have to strip them further down.
    wave_numbers = cpdr.solve_cpdr_for_norm_factor(omega, X_range)  # << u.rad / u.m

    # It is more performant to substitute omega here, since it is the
    # same for all root pairs/triplets, and then lambdify the expression outside
    # the loop and use the lambdified object within the loop to replace X and k.
    values_dict = {"omega": omega.value}

    # Derivative in omega
    cpdr_domega_lamb = sym.lambdify(
        ["X", "k"], cpdr.poly_in_k_domega.subs(values_dict), "numpy"
    )

    # Derivative in k
    cpdr_dk_lamb = sym.lambdify(
        ["X", "k"], cpdr.poly_in_k_dk.subs(values_dict), "numpy"
    )

    norm_factor = np.zeros_like(X_range.value, dtype=np.float64)
    for i in range(norm_factor.shape[0]):
        X = X_range[i]
        k = wave_numbers[i]

        norm_factor[i] = (
            k.value**2 * np.abs(cpdr_domega_lamb(X.value, k.value)) * X
        ) / ((1 + X**2) ** (3 / 2) * np.abs(cpdr_dk_lamb(X.value, k.value)))
    norm_factor /= 2 * np.pi**2

    return norm_factor << UNIT_NF
