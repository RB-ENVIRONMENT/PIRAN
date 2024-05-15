import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.special import erf, jv

from piran.cpdr import Cpdr, ResonantRoot


@u.quantity_input
def get_power_spectral_density(
    cpdr: Cpdr,
    wave_amplitude: u.Quantity[u.T],
    omega: u.Quantity[u.rad / u.s],
) -> u.Quantity[u.T**2 * u.s / u.rad]:
    """
    Calculate the power spectral density B_squared(omega) term
    from equation 5 in Glauert & Horne 2005, in units T^2 * s / rad.

    Parameters
    ----------
    cpdr : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    wave_amplitude : astropy.units.quantity.Quantity[u.T]
        Wave ampltitude.
    omega : astropy.units.quantity.Quantity[u.rad / u.s]
        Wave frequency.

    Returns
    -------
    power_spectral_density : astropy.units.quantity.Quantity[u.T**2 * u.s / u.rad]
        Power spectral density.
    """
    delta_omega = cpdr.wave_freqs._width
    mean_omega = cpdr.wave_freqs._peak
    lower_cutoff = cpdr.omega_lc
    upper_cutoff = cpdr.omega_uc

    error = np.sum(
        erf(
            [
                (mean_omega - lower_cutoff) / delta_omega,
                (upper_cutoff - mean_omega) / delta_omega,
            ]
        )
    )
    normalisation_constant = (
        (wave_amplitude**2 / delta_omega) * (2 / np.sqrt(np.pi)) / error
    )  # A_squared

    power_spectral_density = normalisation_constant * cpdr.wave_freqs.eval(omega)
    return power_spectral_density


@u.quantity_input
def get_phi_squared(
    cpdr: Cpdr,
    resonant_root: ResonantRoot,
) -> u.Quantity[u.dimensionless_unscaled]:
    """
    Calculate Phi_{n,k}^2 from equation A13 in Glauert & Horne 2005.

    Parameters
    ----------
    cpdr : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    resonant_root : piran.cpdr.ResonantRoot object
        NamedTuple object containing a resonant root, i.e.,
        root to both dispersion relation and resonance condition.

    Returns
    -------
    phi_squared : astropy.units.quantity.Quantity[u.dimensionless_unscaled]
        Phi_{n,k}^2.
    """
    psi = np.arctan(resonant_root.X)
    mu = const.c * np.abs(resonant_root.k) / resonant_root.omega
    L = cpdr.stix.L(resonant_root.omega)
    S = cpdr.stix.S(resonant_root.omega)
    R = cpdr.stix.R(resonant_root.omega)
    P = cpdr.stix.P(resonant_root.omega)

    point = (
        resonant_root.k_perp
        * cpdr.p_perp
        / (cpdr.plasma.particles[0].mass * cpdr.plasma.gyro_freq[0])
    )

    # Bessel function of the first kind
    J = jv([cpdr.resonance - 1, cpdr.resonance, cpdr.resonance + 1], point)

    term1 = ((mu**2 - L) / (mu**2 - S)) * J[2] + ((mu**2 - R) / (mu**2 - S)) * J[0]
    term2 = (mu**2 * np.sin(psi) ** 2 - P) / (2 * mu**2)
    term3 = (1 / np.tan(cpdr.alpha)) * np.sin(psi) * np.cos(psi) * J[1]
    term4 = ((R - L) / (2 * (mu**2 - S))) ** 2
    term5 = ((P - mu**2 * np.sin(psi) ** 2) / mu**2) ** 2
    term6 = (P * np.cos(psi) / mu**2) ** 2

    phi_squared = (term1 * term2 + term3) ** 2 / (term4 * term5 + term6)
    return phi_squared


@u.quantity_input
def get_singular_term(
    cpdr: Cpdr,
    resonant_root: ResonantRoot,
) -> u.Quantity[u.m / u.s]:
    """
    Calculate the denominator from the last term in equation 5 in Glauert & Horne 2005.
    The term is v_par - d(omega) / d(k_par) evaluated at signed k_par.
    More specifically, the term is v_par - (- (dD/dk) / (dD/domega) ) * (1 / cos(psi)),
    with cos(psi) = ± 1 / sqrt(1 + X^2).
    cos(psi) = + 1 / sqrt(1 + X^2) if psi in [0, 90] (or equivalently, k_par pos)
    cos(psi) = - 1 / sqrt(1 + X^2) if psi in (90, 180] (or equivalently, k_par neg).

    Parameters
    ----------
    cpdr : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    resonant_root : piran.cpdr.ResonantRoot object
        NamedTuple object containing a resonant root, i.e.,
        root to both dispersion relation and resonance condition.

    Returns
    -------
    singular_term : astropy.units.quantity.Quantity[u.m / u.s]
    """
    X = resonant_root.X
    omega = resonant_root.omega
    k = resonant_root.k

    dD_dk = cpdr.stix.dD_dk(omega, X, k)
    dD_dw = cpdr.stix.dD_dw(omega, X, k)

    if resonant_root.k_par >= 0.0:
        singular_term = cpdr.v_par + (dD_dk / dD_dw) * np.sqrt(1 + X**2)
    else:
        singular_term = cpdr.v_par - (dD_dk / dD_dw) * np.sqrt(1 + X**2)

    return singular_term
