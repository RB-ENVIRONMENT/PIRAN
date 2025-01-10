"""
The `plasmapoint` module provides a class for representing a quasineutral plasma
at a specific point in space. It calculates key plasma parameters like
plasma frequency and gyrofrequency based on particle composition and
either the plasma-to-gyro ratio for electrons or number densities for each
species.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from plasmapy.formulary.frequencies import wc_, wp_
from plasmapy.particles import ParticleList, ParticleListLike

from piran.magpoint import MagPoint


class PlasmaPoint:
    """
    Represents a quasineutral plasma at a specific point in space.

    This class calculates plasma parameters, including plasma frequency
    and gyrofrequency, based on the particle composition and provided
    input. It supports two main use cases:

    1. Electron-proton plasmas, where the plasma-to-gyro ratio for
       electrons is provided.
    2. Plasmas with an arbitrary number of species, where the number
       density for each species is provided.

    Parameters
    ----------
    magpoint : MagPoint
        The magnetic field at a given point in space.

    particles : ParticleListLike
        A list-like collection of plasmapy particle-like objects.

    plasma_over_gyro_ratio : float | None, default=None
        The ratio of plasma frequency to gyrofrequency for electrons.
        Supported if the plasma consists of electrons and protons only.

    number_density : astropy.units.Quantity[1 / u.m**3] | None, default=None
        A list or array of number densities for each particle species (in m^-3).
        Required if the plasma contains more than two species or if it does not
        contain exactly electron and proton.

    Attributes
    ----------
    gyro_freq : astropy.units.Quantity[u.rad / u.s]
        An array containing the gyrofrequency for each species. The order of
        the frequencies corresponds to the order of the particles in the
        `particles` list.
    plasma_freq : astropy.units.Quantity[u.rad / u.s]
        An array containing the plasma frequency for each species. The order
        of the frequencies corresponds to the order of the particles in the
        `particles` list.

    Raises
    ------
    ValueError
        If the input parameters are inconsistent. For example:
            * If neither `plasma_over_gyro_ratio` nor `number_density` is provided.
            * If *both* `plasma_over_gyro_ratio` and `number_density` are provided.
            * If `plasma_over_gyro_ratio` is provided but `particles` are not
              exactly electron and proton.

    Examples
    --------
    >>> mag_point = MagPoint(Angle(0 * u.deg), 4.5)
    >>> particles = ("e", "H+")
    >>> plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio=1.5)

    >>> mag_point = MagPoint(Angle(0 * u.deg), 4.5)
    >>> particles = ("e", "p+")
    >>> num_density = [2524781.78, 2524781.78] * (1 / u.m**3)
    >>> plasma_point = PlasmaPoint(mag_point, particles, number_density=num_density)
    """

    @u.quantity_input
    def __init__(
        self,
        magpoint: MagPoint,
        particles: ParticleListLike,
        plasma_over_gyro_ratio: float | None = None,
        number_density: Quantity[1 / u.m**3] | None = None,
    ) -> None:
        self.__magpoint = magpoint
        self.__particles = ParticleList(particles)
        self.__plasma_over_gyro_ratio = plasma_over_gyro_ratio
        self.__number_density = number_density
        self.__gyro_freq = self.__compute_gyro_freq()  # rad/s
        self.__plasma_freq = self.__compute_plasma_freq()  # rad/s

        # =================================================================
        # Currently we support two use cases. We either expect an electron
        # and proton plasma and plasma over gyro ratio for electrons or
        # any number of species and a list of their number densities.
        # Based on these we calculate plasma frequencies and here we
        # calculate plasma over gyro ratio and number densities for all
        # species in both cases.
        if self.number_density is not None and self.plasma_over_gyro_ratio is None:
            self.__plasma_over_gyro_ratio = np.abs(self.plasma_freq / self.gyro_freq)

        if self.plasma_over_gyro_ratio is not None and self.number_density is None:
            self.__number_density = Quantity(
                [
                    omega_p**2 * const.eps0 * p.mass / np.abs(p.charge) ** 2
                    for omega_p, p in zip(self.plasma_freq, self.particles)
                ]
            ).to(1 / u.m**3, equivalencies=u.dimensionless_angles())
            self.__plasma_over_gyro_ratio = np.abs(self.plasma_freq / self.gyro_freq)
        # =================================================================

        self.__plasma_charge = sum(
            [nd * p.charge_number for nd, p in zip(self.number_density, self.particles)]
        )

    @property
    def magpoint(self):
        return self.__magpoint

    @property
    def particles(self):
        return self.__particles

    @property
    def plasma_over_gyro_ratio(self):
        return self.__plasma_over_gyro_ratio

    @property
    def number_density(self):
        return self.__number_density

    @property
    def gyro_freq(self):
        return self.__gyro_freq

    @property
    def plasma_freq(self):
        return self.__plasma_freq

    @property
    def plasma_charge(self):
        return self.__plasma_charge

    @u.quantity_input
    def __compute_gyro_freq(self) -> Quantity[u.rad / u.s]:
        B = self.magpoint.flux_density
        gf = [wc_(B, p, signed=True) for p in self.particles]
        return Quantity(gf)

    @u.quantity_input
    def __compute_plasma_freq(self) -> Quantity[u.rad / u.s]:
        if (
            len(self.particles) == 2
            and self.particles[0].symbol == "e-"
            and (self.particles[1].symbol == "p+" or self.particles[1].symbol == "H 1+")
            and self.plasma_over_gyro_ratio is not None
            and self.number_density is None
            and isinstance(self.plasma_over_gyro_ratio, float)
        ):
            # This is restrictive because, for now, we only support net-zero
            # charge plasmas of electrons and protons.
            electron = self.particles[0]
            electron_pf = np.abs(self.gyro_freq[0]) * self.plasma_over_gyro_ratio
            num_density = (
                electron_pf**2
                * const.eps0
                * electron.mass
                / np.abs(electron.charge) ** 2
            ).to(1 / u.m**3, equivalencies=u.dimensionless_angles())

            pf = [electron_pf]
            for particle in self.particles[1:]:
                pf.append(wp_(num_density, particle))
            return Quantity(pf)
        elif self.number_density is not None and self.plasma_over_gyro_ratio is None:
            pf = []
            for i, particle in enumerate(self.particles):
                pf.append(wp_(self.number_density[i], particle))
            return Quantity(pf)
        else:
            raise ValueError("Not valid combination of input arguments")
