"""
Defines the PiranParticle and Particles classes for use with the Cpdr.
"""

from typing import Sequence

import plasmapy


class PiranParticle(plasmapy.particles.Particle):
    """
    PiranParticle is a convenient subclass of plasmapy.particles.Particle.

    We add the ``density`` property, describing the number density of this particular
    particle within our cold plasma.

    Parameters
    ----------
    particle : ParticleLike
        A particle of interest within our cold plasma.

    density : float
        THe number density of ``particle`` within our cold plasma.
    """

    def __init__(
        self, particle: plasmapy.particles.ParticleLike, density: float
    ) -> None:  # numpydoc ignore=GL08
        plasmapy.particles.Particle.__init__(self, particle)
        self.density = density


class Particles:
    """
    Particles contains all the info related to particles within our cold plasma.

    Parameters
    ----------
    particles : Sequence[PiranParticle]
        A sequence of PiranParticles detailing all particles within the plasma and their
        associated number densities.

    energies : Sequence[float]
        A sequence of particle energies. For each value included here, the PIRAN routine
        will return a set of results for every value included in ``pitch_angles``.

    pitch_angles : Sequence[float]
        A sequence of particle pitch angles. For each value included here, the PIRAN
        routine will return a set of results for every value included in ``energies``.
    """

    def __init__(
        self,
        particles: Sequence[PiranParticle],
        energies: Sequence[float],
        pitch_angles: Sequence[float],
    ) -> None:  # numpydoc ignore=GL08
        self.all = particles
        self.energies = energies
        self.pitch_angles = pitch_angles
