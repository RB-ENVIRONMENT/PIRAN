"""
Defines the PlasmaPoint class.
"""

from typing import Sequence

import numpy as np
from astropy import constants as const
from astropy import units as u
from plasmapy.particles import Particle

from piran.magpoint import MagPoint


class IllegalArgumentError(ValueError):
    pass


class PlasmaPoint:

    @u.quantity_input
    def __init__(
        self,
        magpoint: MagPoint,
        particles: Sequence[str],
        plasma_over_gyro_ratio: float | None = None,
        number_density: Sequence[float] | None = None,
    ) -> None:
        self.__magpoint = magpoint
        self.__particles = tuple(Particle(x) for x in particles)
        self.__plasma_over_gyro_ratio = plasma_over_gyro_ratio
        self.__number_density = number_density
        self.__gyro_freq = self.__compute_gyro_freq()
        self.__plasma_freq = self.__compute_plasma_freq()

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

    def __compute_gyro_freq(self):
        B = self.magpoint.flux_density
        gf = [(x.charge * B) / x.mass for x in self.particles]
        return tuple(gf)

    def __compute_plasma_freq(self):
        if (
            self.plasma_over_gyro_ratio is not None and
            self.number_density is None and
            isinstance(self.plasma_over_gyro_ratio, float)
        ):
            prim_particle = self.particles[0]
            prim_particle_pf = abs(self.gyro_freq[0]) * self.plasma_over_gyro_ratio
            num_density = prim_particle_pf**2 * const.eps0 * prim_particle.mass / abs(prim_particle.charge) ** 2

            pf = [prim_particle_pf]
            for i in range(1, len(self.particles)):
                charge = self.particles[i].charge
                mass = self.particles[i].mass
                pf.append(np.sqrt((num_density * charge**2) / (const.eps0 * mass)))
            return tuple(pf)
        else:
            raise IllegalArgumentError("Not valid combination of input arguments")
