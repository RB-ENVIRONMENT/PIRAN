import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import simpson, trapezoid

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.stix import Stix

UNIT_NF = u.s / u.m**3


class Jacobian:

    @u.quantity_input
    def __init__(self, stix: Stix, omega: u.Quantity[u.rad / u.s]) -> None:
        self.__stix = stix
        self.__omega = omega

        self.__R = stix.R(omega)
        self.__L = stix.L(omega)
        self.__P = stix.P(omega)
        self.__S = stix.S(omega)

        self.__dR = stix.dR(omega)
        self.__dL = stix.dL(omega)
        self.__dP = stix.dP(omega)
        self.__dS = stix.dS(omega)

    def __A(
        self, X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.__S * X**2) + self.__P

    def __B(
        self, X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.__R * self.__L * X**2) + ((self.__P * self.__S) * (2 + X**2))

    def __C(
        self, X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.__P * self.__R * self.__L) * (1 + X**2)

    def __dA(self, X: u.Quantity[u.dimensionless_unscaled]) -> u.Quantity[u.s / u.rad]:
        return (self.__dS * X**2) + self.__dP

    def __dB(self, X: u.Quantity[u.dimensionless_unscaled]) -> u.Quantity[u.s / u.rad]:
        return ((self.__dR * self.__L + self.__R * self.__dL) * (X**2)) + (
            (self.__dP * self.__S + self.__P * self.__dS) * (2 + X**2)
        )

    def __dC(self, X: u.Quantity[u.dimensionless_unscaled]) -> u.Quantity[u.s / u.rad]:
        return (
            self.__dP * self.__R * self.__L
            + self.__P * self.__dR * self.__L
            + self.__P * self.__R * self.__dL
        ) * (1 + X**2)

    @u.quantity_input
    def calculate(
        self,
        X: u.Quantity[u.dimensionless_unscaled],
        k: u.Quantity[u.rad / u.m],
    ) -> u.Quantity[u.rad * u.s / u.m**2]:
        mu = const.c * k / self.__omega
        return ((k**2) / (1 + X**2)) * (
            (
                (self.__dA(X) * mu**4 - self.__dB(X) * mu**2 + self.__dC(X))
                / (2 * (2 * self.__A(X) * mu**4 - self.__B(X) * mu**2))
            )
            - (1 / self.__omega)
        )


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
    # solution to the dispersion relation.
    wave_numbers = cpdr.solve_cpdr_for_norm_factor(omega, X_range)

    # Substitute fixed omega to retrieve an expression for the Jacobian in terms of (X, k)
    jacob = Jacobian(cpdr.stix, omega)

    eval_gx = wave_norm_angle_dist.eval(X_range)

    evaluated_integrand = np.zeros_like(X_range, dtype=np.float64)
    for i in range(evaluated_integrand.shape[0]):
        X = X_range[i]
        k = wave_numbers[i]

        # We need this conditional here after refactoring
        # this function and solve_cpdr_for_norm_factor() in
        # commit ed48d76a9d8d1cfdffbe2113e986d94582e461cd.
        # Without it, if one wave number from the list is NaN, then,
        # for that index, `evaluated_integrand` becomes NaN which means
        # that the integration fails (`integral` becomes NaN too).
        if np.isnan(k):
            evaluated_integrand[i] = 0.0
        else:
            evaluated_integrand[i] = (
                eval_gx[i] * k.value * X * np.abs(jacob.calculate(X, k).value)
            ) / ((1 + X**2) ** (1 / 2))

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

    # Substitute fixed omega to retrieve an expression for the Jacobian in terms of (X, k)
    jacob = Jacobian(cpdr.stix, omega)

    norm_factor = np.zeros_like(X_range.value, dtype=np.float64)
    for i in range(norm_factor.shape[0]):
        X = X_range[i]
        k = wave_numbers[i]

        if np.isnan(k):
            norm_factor[i] = 0.0
        else:
            norm_factor[i] = (k.value * X * np.abs(jacob.calculate(X, k).value)) / (
                (1 + X**2) ** (1 / 2)
            )

    norm_factor /= 2 * np.pi**2

    return norm_factor << UNIT_NF
