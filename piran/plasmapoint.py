"""
Defines the PlasmaPoint class.
"""

from typing import Sequence

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from plasmapy.formulary.frequencies import wc_, wp_
from plasmapy.particles import ParticleList, ParticleListLike

from piran.magpoint import MagPoint


class IllegalArgumentError(ValueError):
    pass


class PlasmaPoint:
    """
    A representation of a quasineutral charge plasma given a list of particles.

    Parameters
    ----------
    magpoint : MagPoint
        The magnetic field at a given point in space.

    particles : ParticleListLike
        A list-like collection of plasmapy particle-like objects.

    plasma_over_gyro_ratio : TBD, default=None

    number_density : TBD, default=None
    """

    @u.quantity_input
    def __init__(
        self,
        magpoint: MagPoint,
        particles: ParticleListLike,
        plasma_over_gyro_ratio: float | None = None,
        number_density: Sequence[float] | None = None,
    ) -> None:
        self.__magpoint = magpoint
        self.__particles = ParticleList(particles)
        self.__plasma_over_gyro_ratio = plasma_over_gyro_ratio
        self.__number_density = number_density
        self.__gyro_freq = self.__compute_gyro_freq()  # Hz
        self.__plasma_freq = self.__compute_plasma_freq()  # Hz

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

    def __compute_gyro_freq(self) -> Quantity[u.Hz]:
        B = self.magpoint.flux_density
        gf = [2 * np.pi * wc_(B, p, signed=True, to_hz=True) for p in self.particles]
        return Quantity(gf)

    def __compute_plasma_freq(self) -> Quantity[u.Hz]:
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
            electron_pf = abs(self.gyro_freq[0]) * self.plasma_over_gyro_ratio
            num_density = (
                electron_pf**2 * const.eps0 * electron.mass / abs(electron.charge) ** 2
            )

            pf = [electron_pf]
            for particle in self.particles[1:]:
                pf.append(2 * np.pi * wp_(num_density, particle, to_hz=True))
            return Quantity(pf)
        else:
            raise IllegalArgumentError("Not valid combination of input arguments")
