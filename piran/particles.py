import plasmapy


class Particles:
    def __init__(self, particles, densities, energies, pitch_angles):
        if len(particles) != len(densities):
            raise Exception("Arguments lists should be of equal length.")

        self.all = plasmapy.particles.ParticleList(particles)
        self.densities = densities

        self.energies = energies  # can be list
        self.pitch_angles = pitch_angles  # can be list
