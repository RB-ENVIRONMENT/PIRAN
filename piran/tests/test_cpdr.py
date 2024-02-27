import math

from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr2 import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


class TestCpdr:
    def setup_method(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        energy = 1.0 * u.MeV
        alpha = Angle(5, u.deg)
        resonance = 2
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        self.cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

    def test_cpdr_1(self):
        assert math.isclose(self.cpdr.energy.value, 1.6021766339e-13)  # Joule
        assert math.isclose(self.cpdr.pitch_angle.value, 0.08726646259)  # radians
        assert math.isclose(self.cpdr.alpha.value, 0.08726646259)  # radians
        assert self.cpdr.resonance == 2
        assert math.isclose(self.cpdr.lorentz_factor, 2.956951183)
        assert math.isclose(self.cpdr.gamma, 2.956951183)
        assert math.isclose(self.cpdr.rel_velocity.value, 2.82128455e+08)  # m/s
        assert math.isclose(self.cpdr.v_par.value, 2.81054871e+08)  # m/s

    def test_cpdr_2(self):
        omega = 7320.627086050828 * (u.rad / u.s)

        X = [0.0, 0.33333333333333337, 1.0, 100] * u.dimensionless_unscaled
        k = self.cpdr.solve_cpdr(omega, X)
        assert math.isclose(k[0], 0.00011414445445389277)
        assert math.isclose(k[1], 0.00011766325510931447)
        assert math.isclose(k[2], 0.00014032247090573543)
        assert math.isnan(k[3]) is True

    def test_cpdr_3(self):
        X = [0.01, 0.99] * u.dimensionless_unscaled
        roots = self.cpdr.solve_resonant(X)

        assert len(roots) == 2
        assert len(roots[0]) == 1
        assert len(roots[1]) == 1
        assert math.isclose(roots[0][0][0], 0.01)
        assert math.isclose(roots[0][0][1], 18549.99508102283)
        assert math.isclose(roots[0][0][2], 0.0002098277253605769)
        assert math.isclose(roots[1][0][0], 0.99)
        assert math.isclose(roots[1][0][1], 19814.68720788155)
        assert math.isclose(roots[1][0][2], 0.0003015784216619821)

    def test_cpdr_4(self):
        """Different pitch angle alpha"""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        energy = 1.0 * u.MeV
        alpha = Angle(71, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

        X = [0.0, 0.3165829145728643] * u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert len(roots) == 2

        assert len(roots[0]) == 1
        assert math.isclose(roots[0][0][0], 0.0)
        assert math.isclose(roots[0][0][1], 22060.04543112965)
        assert math.isclose(roots[0][0][2], 0.00024016935645729707)

        assert len(roots[1]) == 2
        assert math.isclose(roots[1][0][0], 0.3165829145728643)
        assert math.isclose(roots[1][0][1], 34361.48787566025)
        assert math.isclose(roots[1][0][2], 0.0003923953536206822)
        assert math.isclose(roots[1][1][0], 0.3165829145728643)
        assert math.isclose(roots[1][1][1], 21197.313961282573)
        assert math.isclose(roots[1][1][2], 0.00024206540583296198)
