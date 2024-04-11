import math

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
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

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        self.cpdr = Cpdr(cpdr_sym, plasma_point)

        omega_ratio = 0.1225
        self.omega = np.abs(self.cpdr.plasma.gyro_freq[0]) * omega_ratio

    def test_stix_1(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        # Find (X, omega, k) CPDR roots
        for X in X_range:
            k = self.cpdr.solve_cpdr(self.omega.value, X.value)
            k <<= u.rad / u.m

            values_dict = {
                "X": X.value,
                "omega": self.omega.value,
                "k": k.value,
            }
            dD_dw = self.cpdr.poly_in_k_domega.subs(values_dict)
            dD_dk = self.cpdr.poly_in_k_dk.subs(values_dict)

            dD_dw <<= u.s / u.rad
            dD_dk <<= u.m / u.rad

            # Test dD/domega
            numeric_dD_dw = self.cpdr.stix.dD_dw(self.omega, X, k)
            assert math.isclose(dD_dw.value, numeric_dD_dw.value)
            assert dD_dw.unit == numeric_dD_dw.unit

            # Test dD/dk
            numeric_dD_dk = self.cpdr.stix.dD_dk(self.omega, X, k)
            assert math.isclose(dD_dk.value, dD_dk.value)
            assert dD_dk.unit == numeric_dD_dk.unit

            # Test jacobian
            sympy_result = (k * dD_dw) / ((1 + X**2) * dD_dk)
            numeric_result = self.cpdr.stix.jacobian(self.omega, X, k)

            assert math.isclose(sympy_result.value, numeric_result.value)
            assert sympy_result.unit == numeric_result.unit
