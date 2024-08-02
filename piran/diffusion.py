import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson
from scipy.special import erf, jv

from piran.cpdr import Cpdr, ResonantRoot
from piran.normalisation import UNIT_NF

UNIT_PSD = u.T**2 * u.s / u.rad
UNIT_BKN = u.T**2 * u.m**3 / u.rad
UNIT_DIFF = (u.kg * u.m / u.s) ** 2 / u.s


@u.quantity_input
def get_power_spectral_density(
    cpdr: Cpdr,
    wave_amplitude: u.Quantity[u.T],
    omega: u.Quantity[u.rad / u.s],
) -> u.Quantity[UNIT_PSD]:
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
    if resonant_root.k_par >= 0.0:
        psi = np.arctan(resonant_root.X)
    else:
        psi = (np.pi << u.rad) - np.arctan(resonant_root.X)
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
    Calculate the denominator from the last term in equation 11 in Glauert & Horne 2005.
    The term is v_par - d(omega) / d(k_par) evaluated at signed k_par and is returned
    without taking its absolute value.
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


@u.quantity_input
def get_normalised_intensity(
    power_spectral_density: u.Quantity[UNIT_PSD],
    wave_norm_angle_dist_eval,
    norm_factor: u.Quantity[UNIT_NF],
) -> u.Quantity[UNIT_BKN]:
    """
    Calculates the normalised intensity |B_{k}^{norm}|^2.

    Depending on the input parameters, this is either equation 4b
    or 5b from Cunningham 2023. Equation 5b is used in Glauert & Horne 2005,
    while equation 4b is the proposed method by Cunningham.
    If we are calculating equation 5b, then `norm_factor` is computed by the
    `compute_glauert_norm_factor()` function in the piran package, which returns
    the N(omega) term in the denominator of 5b, while if we are calculating
    equation 4b, then `norm_factor` shall be computed by piran's
    `compute_cunningham_norm_factor()` function which returns the denominator
    in 4b.
    Note that `wave_norm_angle_dist_eval` must be normalised if we are calculating
    equation 4b (Cunningham's proposed method).

    Parameters
    ----------
    power_spectral_density : u.Quantity[u.T**2 * u.s / u.rad]
        Power spectral density B^2(omega).
    wave_norm_angle_dist_eval :
        Wave normal angle distribution evaluated at X.
    norm_factor :
        Normalisation factor.

    Returns
    -------
    normalised_intensity :
        Normalised intensity |B_{k}^{norm}|^2
    """
    normalised_intensity = (
        power_spectral_density * wave_norm_angle_dist_eval / norm_factor
    )

    return normalised_intensity


@u.quantity_input
def get_DnX_single_root(
    cpdr: Cpdr,
    resonant_root: ResonantRoot,
    normalised_intensity: u.Quantity[UNIT_BKN],
    phi_squared: u.Quantity[u.dimensionless_unscaled],
    singular_term: u.Quantity[u.m / u.s],
) -> tuple[u.Quantity[UNIT_DIFF], u.Quantity[UNIT_DIFF], u.Quantity[UNIT_DIFF]]:
    """
    Calculates the diffusion coefficients in pitch angle DnXaa,
    mixed pitch angle-momentum DnXap and momentum DnXpp, for a
    given resonant root, as defined in equations 11, 12 and 13
    in Glauert & Horne 2005.

    Parameters
    ----------
    cpdr : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    resonant_root : piran.cpdr.ResonantRoot object
        NamedTuple object containing a resonant root, i.e.,
        root to both dispersion relation and resonance condition.
    normalised_intensity :
        Normalised intensity |B_{k}^{norm}|^2
    phi_squared : astropy.units.quantity.Quantity[u.dimensionless_unscaled]
        Phi_{n,k}^2.
    singular_term : astropy.units.quantity.Quantity[u.m / u.s]
        v_par - d(omega) / d(k_par)

    Returns
    -------
    DnXaa, DnXap, DnXpp :
        Pitch angle, mixed pitch angle-momentum and momentum diffusion
        coefficients for a given resonant root.
        Note that since these are wrapped in a tuple, it appears that the
        `@u.quantity_input` decorator does not actually check these! There is an
        appropriate test in place in test_diffusion.py to check the validity of
        these outputs.
    """
    charge = cpdr.plasma.particles[0].charge
    gyrofreq = cpdr.plasma.gyro_freq[0]
    alpha = cpdr.alpha

    term1 = (charge**2 * resonant_root.omega**2) / (
        4 * Angle(np.pi, u.rad) * (1 + resonant_root.X**2)
    )
    term2 = (
        cpdr.resonance * gyrofreq / (cpdr.gamma * resonant_root.omega)
        - np.sin(alpha) ** 2
    ) / np.cos(alpha)
    term3 = normalised_intensity * phi_squared / np.abs(singular_term)

    DnXaa = term1 * term2**2 * term3
    DnXap = DnXaa * np.sin(alpha) / term2
    DnXpp = DnXaa * (np.sin(alpha) / term2) ** 2

    return DnXaa, DnXap, DnXpp


@u.quantity_input
def get_diffusion_coefficients(
    X_range: u.Quantity[u.dimensionless_unscaled],
    DnX_single_res: u.Quantity[UNIT_DIFF],
) -> u.Quantity[UNIT_DIFF]:
    r"""
    Given an array of wave normal angles and an an identically-sized
    array of outputs from Equation 11, 12, or 13, calculate
    $\int_{X_min}^{X_max} X D_{\alpha \alpha}^{nX} dX$, i.e. the integral part
    from equations 8, 9 or 10 in Glauert & Horne 2005. No summation over
    different resonances happens in this function.
    Note: For the integration we use Simpson's rule.

    Parameters
    ----------
    X_range : astropy.units.quantity.Quantity[u.dimensionless_unscaled]
        Array of wave normal angles.
    DnX_single_res :
        Array of diffusion coefficients for a specific resonance,
        one per X (i.e. calculated values from equations 11, 12 or 13
        in Glauert & Horne 2005).

    Returns
    -------
    Either $D_{\alpha \alpha}$ or $D_{\alpha p}$ ($D_{p \alpha}$) or
    $D_{pp}$, i.e. equations 8, 9 and 10 from Glauert 2005) for a single
    resonance.
    """
    if X_range.shape != DnX_single_res.shape:
        raise ValueError("Input arrays should have the same shape")

    integrand = np.multiply(X_range, DnX_single_res)
    integral = simpson(integrand, x=X_range)

    # simpson strips units; let's add them back!
    # This should be UNIT_DIFF, but let's perform a more robust check.
    integral <<= integrand.unit * X_range.unit

    return integral


@u.quantity_input
def get_energy_diffusion_coefficient(
    rel_kin_energy: u.Quantity[u.J],
    rest_mass_energy: u.Quantity[u.J],
    momentum_diff_coef: u.Quantity[UNIT_DIFF],
) -> u.Quantity[u.J**2 / u.s]:
    """
    Given relativistic kinetic energy, rest mass energy and
    relativistic momentum diffusion coefficient calculate
    the energy diffusion coefficient from equation 29 in
    Glauert & Horne 2005.

    Parameters
    ----------
    rel_kin_energy : astropy.units.quantity.Quantity.Quantity[Joule],
        Relativistic kinetic energy.
    rest_mass_energy : astropy.units.quantity.Quantity.Quantity[Joule],
        Rest mass energy.
    momentum_diff_coef :
        Momentum diffusion coefficient $D_{pp}$.

    Returns
    -------
    energy_diff_coef :
        Energy diffusion coefficient $D_{EE}$.
    """
    energy_diff_coef = (
        momentum_diff_coef
        * (const.c**2 * rel_kin_energy * (rel_kin_energy + 2 * rest_mass_energy))
        / (rel_kin_energy + rest_mass_energy) ** 2
    )

    return energy_diff_coef
