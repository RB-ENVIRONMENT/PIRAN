import plasmapy
from typing import Sequence


class Particles:
    def __init__(
        self,
        particles: Sequence[str],
        densities: Sequence[float],
        energies: Sequence[float],
        pitch_angles: Sequence[float],
    ):
        if len(particles) != len(densities):
            raise Exception("Arguments lists should be of equal length.")

        self.all = plasmapy.particles.ParticleList(particles)
        self.densities = densities

        self.energies = energies  # can be list
        self.pitch_angles = pitch_angles  # can be list
