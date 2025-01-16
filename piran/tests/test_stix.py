import math

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


class TestStix:
    def setup_method(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        self.cpdr = Cpdr(plasma_point)

        omega_ratio = 0.1225
        self.omega = np.abs(self.cpdr.plasma.gyro_freq[0]) * omega_ratio

    def test_stix_1(self):
        # Find (X, omega, k) CPDR roots
        X = u.Quantity(0.5, u.dimensionless_unscaled)
        k = self.cpdr.solve_cpdr(self.omega, X)[0]
        k <<= u.rad / u.m

        # Test dD/domega
        dD_dw = self.cpdr.stix.dD_dw(self.omega, X, k)
        assert math.isclose(dD_dw.value, 25.43102517952277)

        # Test dD/dk
        dD_dk = self.cpdr.stix.dD_dk(self.omega, X, k)
        assert math.isclose(dD_dk.value, -2537950057.1427784)

        # Test jacobian
        jacobian = self.cpdr.stix.jacobian(self.omega, X, k)
        assert math.isclose(jacobian.value, -9.7616573640402e-13)
