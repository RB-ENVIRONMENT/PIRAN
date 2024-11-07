import math

import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.magpoint import MagPoint
from piran.plasmapoint import IllegalArgumentError, PlasmaPoint


class TestPlasmaPoint:
    def test_plasmapoint_1(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5

        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        assert math.isclose(plasma_point.particles[0].charge.value, -1.602176634e-19)
        assert math.isclose(plasma_point.particles[0].mass.value, 9.1093837015e-31)
        assert math.isclose(plasma_point.particles[1].charge.value, 1.602176634e-19)
        assert math.isclose(plasma_point.particles[1].mass.value, 1.67262192369e-27)
        assert math.isclose(plasma_point.plasma_over_gyro_ratio[0], 1.5)
        assert math.isclose(plasma_point.plasma_over_gyro_ratio[1], 64.27552812)
        assert math.isclose(plasma_point.number_density[0].value, 2524781.78)
        assert math.isclose(plasma_point.number_density[1].value, 2524781.78)
        assert math.isclose(plasma_point.gyro_freq[0].value, -59760.22, rel_tol=1e-6)
        assert math.isclose(plasma_point.gyro_freq[1].value, 32.5464, rel_tol=1e-5)
        assert math.isclose(plasma_point.plasma_freq[0].value, 89640.33, rel_tol=1e-6)
        assert math.isclose(plasma_point.plasma_freq[1].value, 2091.93, rel_tol=1e-5)

    def test_plasmapoint_2(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "H+")
        plasma_over_gyro_ratio = 1.5

        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        assert math.isclose(plasma_point.particles[0].charge.value, -1.602176634e-19)
        assert math.isclose(plasma_point.particles[0].mass.value, 9.1093837015e-31)
        assert math.isclose(plasma_point.particles[1].charge.value, 1.602176634e-19)
        assert math.isclose(plasma_point.particles[1].mass.value, 1.67291244076265e-27)
        assert math.isclose(plasma_point.plasma_over_gyro_ratio[0], 1.5)
        assert math.isclose(plasma_point.plasma_over_gyro_ratio[1], 64.28110987)
        assert math.isclose(plasma_point.number_density[0].value, 2524781.78)
        assert math.isclose(plasma_point.number_density[1].value, 2524781.78)
        assert math.isclose(plasma_point.gyro_freq[0].value, -59760.22, rel_tol=1e-6)
        assert math.isclose(plasma_point.gyro_freq[1].value, 32.5407, rel_tol=1e-5)
        assert math.isclose(plasma_point.plasma_freq[0].value, 89640.33, rel_tol=1e-6)
        assert math.isclose(plasma_point.plasma_freq[1].value, 2091.75, rel_tol=1e-5)

    def test_plasmapoint_3(self):
        """
        If we provide plasma over gyro ratio for electrons then
        number_density must be None.
        """
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        number_density = 1 * (1 / u.m**3)

        with pytest.raises(IllegalArgumentError):
            PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio, number_density)

    def test_plasmapoint_4(self):
        "Get plasma frequency given number density for all species."
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        num_density = [2524781.78, 2524781.78] * (1 / u.m**3)

        plasma_point = PlasmaPoint(mag_point, particles, number_density=num_density)

        assert math.isclose(plasma_point.particles[0].charge.value, -1.602176634e-19)
        assert math.isclose(plasma_point.particles[0].mass.value, 9.1093837015e-31)
        assert math.isclose(plasma_point.particles[1].charge.value, 1.602176634e-19)
        assert math.isclose(plasma_point.particles[1].mass.value, 1.67262192369e-27)

        assert math.isclose(plasma_point.gyro_freq[0].value, -59760.22, rel_tol=1e-6)
        assert math.isclose(plasma_point.gyro_freq[1].value, 32.5464, rel_tol=1e-5)

        assert math.isclose(plasma_point.plasma_freq[0].value, 89640.33, rel_tol=1e-6)
        assert math.isclose(plasma_point.plasma_freq[1].value, 2091.93, rel_tol=1e-5)

        assert math.isclose(plasma_point.number_density[0].value, 2524781.78)
        assert math.isclose(plasma_point.number_density[1].value, 2524781.78)

        assert math.isclose(plasma_point.plasma_over_gyro_ratio[0], 1.5)
        assert math.isclose(plasma_point.plasma_over_gyro_ratio[1], 64.27552812)

        assert math.isclose(plasma_point.plasma_charge.value, 0)
