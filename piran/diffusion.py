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
    resonance = cpdr.resonance
    alpha = cpdr.alpha
    X = resonant_root.X
    psi = np.arctan(X)
    p_perp = cpdr.p_perp
    mass = cpdr.plasma.particles[0].mass
    gyrofreq = cpdr.plasma.gyro_freq[0]
    k = resonant_root.k
    k_perp = resonant_root.k_perp
    omega = resonant_root.omega
    mu = const.c * np.abs(k) / omega
    L = cpdr.stix.L(omega)
    S = cpdr.stix.S(omega)
    R = cpdr.stix.R(omega)
    P = cpdr.stix.P(omega)

    point = k_perp * p_perp / (mass * gyrofreq)

    # Bessel function of the first kind
    J = jv([resonance - 1, resonance, resonance + 1], point)

    term1 = ((mu**2 - L) / (mu**2 - S)) * J[2] + ((mu**2 - R) / (mu**2 - S)) * J[0]
    term2 = (mu**2 * np.sin(psi) ** 2 - P) / (2 * mu**2)
    term3 = (1 / np.tan(alpha)) * np.sin(psi) * np.cos(psi) * J[1]
    term4 = ((R - L) / (2 * (mu**2 - S))) ** 2
    term5 = ((P - mu**2 * np.sin(psi) ** 2) / mu**2) ** 2
    term6 = (P * np.cos(psi) / mu**2) ** 2

    phi_squared = (term1 * term2 + term3) ** 2 / (term4 * term5 + term6)
    return phi_squared
