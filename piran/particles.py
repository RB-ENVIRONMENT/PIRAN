import plasmapy
from typing import Sequence


class PiranParticle(plasmapy.particles.Particle):
    def __init__(self, particle: str, density: float) -> None:
        plasmapy.particles.Particle.__init__(self, particle)
        self.density = density


class Particles:
    def __init__(
        self,
        particles: Sequence[str],
        densities: Sequence[float],
        energies: Sequence[float],
        pitch_angles: Sequence[float],
    ) -> None:
        if len(particles) != len(densities):
            raise Exception("Arguments lists should be of equal length.")

        self.all = [
            PiranParticle(elem[0], elem[1]) for elem in zip(particles, densities)
        ]

        self.energies = energies
        self.pitch_angles = pitch_angles
