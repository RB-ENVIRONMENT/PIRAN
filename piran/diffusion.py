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

    error = np.sum(erf([(mean_omega - lower_cutoff) / delta_omega, (upper_cutoff - mean_omega) / delta_omega]))
    normalisation_constant = (wave_amplitude**2 / delta_omega) * (2 / np.sqrt(np.pi)) / error  # A_squared

    power_spectral_density = normalisation_constant * cpdr.wave_freqs.eval(omega)
    return power_spectral_density

