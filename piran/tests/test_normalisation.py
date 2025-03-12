# Copyright (C) 2025 The University of Birmingham, United Kingdom /
#   Dr Oliver Allanson, ORCiD: 0000-0003-2353-8586, School Of Engineering, University of Birmingham /
#   Dr Thomas Kappas, ORCiD: 0009-0003-5888-2093, Advanced Research Computing, University of Birmingham /
#   Dr James Tyrrell, ORCiD: 0000-0002-2344-737X, Advanced Research Computing, University of Birmingham /
#   Dr Adrian Garcia, ORCiD: 0009-0007-4450-324X, Advanced Research Computing, University of Birmingham
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magpoint import MagPoint
from piran.normalisation import (
    compute_cunningham_norm_factor,
    compute_glauert_norm_factor,
)
from piran.plasmapoint import PlasmaPoint


class TestNormalisationFactors:
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

    def test_glauert_normalisation_1(self):
        X_min = 0.0 << u.dimensionless_unscaled
        X_max = 1.0 << u.dimensionless_unscaled
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        X_m = 0.0 << u.dimensionless_unscaled  # peak
        X_w = 0.577 << u.dimensionless_unscaled  # angular width
        wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

        norm_factor_glauert = compute_glauert_norm_factor(
            self.cpdr, self.omega, X_range, wave_norm_angle_dist, method="simpson"
        )

        expected = 8.53830973e-19
        assert math.isclose(
            norm_factor_glauert.value, expected, rel_tol=1e-09, abs_tol=1e-27
        )

    def test_glauert_normalisation_2(self):
        X_min = 0.0 << u.dimensionless_unscaled
        X_max = 1.0 << u.dimensionless_unscaled
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        X_m = 0.0 << u.dimensionless_unscaled  # peak
        X_w = 0.577 << u.dimensionless_unscaled  # angular width
        wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

        norm_factor_glauert = compute_glauert_norm_factor(
            self.cpdr, self.omega, X_range, wave_norm_angle_dist, method="trapezoid"
        )

        expected = 8.53769363e-19
        assert math.isclose(
            norm_factor_glauert.value, expected, rel_tol=1e-09, abs_tol=1e-27
        )

    def test_glauert_normalisation_3(self):
        X_min = 0.00 << u.dimensionless_unscaled
        X_max = 5.67 << u.dimensionless_unscaled
        X_npoints = 300
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        X_m = 0.0 << u.dimensionless_unscaled  # peak
        X_w = 0.577 << u.dimensionless_unscaled  # angular width
        wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

        norm_factor_glauert = compute_glauert_norm_factor(
            self.cpdr, self.omega, X_range, wave_norm_angle_dist
        )

        expected = 8.87151913e-19
        assert math.isclose(
            norm_factor_glauert.value, expected, rel_tol=1e-09, abs_tol=1e-27
        )

    def test_cunningham_normalisation_1(self):
        X_min = 0.0 << u.dimensionless_unscaled
        X_max = 1.0 << u.dimensionless_unscaled
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        cnf = compute_cunningham_norm_factor(
            self.cpdr,
            self.omega,
            X_range,
        )

        # Check the first, last, a few points in-between and the sum of all values.
        # If an assertion fails, the test fails and the remaining asserts
        # are not evaluated.
        assert cnf[0] == 0.0
        assert math.isclose(cnf[1].value, 6.14971590e-20, rel_tol=1e-09, abs_tol=1e-28)
        assert math.isclose(cnf[20].value, 1.20262320e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf[50].value, 2.71450691e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf[80].value, 3.77673193e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf[-1].value, 4.27878582e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(
            cnf.sum().value, 2.49251344e-16, rel_tol=1e-09, abs_tol=1e-24
        )

        # Check the size of the array
        assert cnf.size == X_npoints

    def test_wrong_integration_rule(self):
        X_min = 0.0 << u.dimensionless_unscaled
        X_max = 1.0 << u.dimensionless_unscaled
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        X_m = 0.0 << u.dimensionless_unscaled  # peak
        X_w = 0.577 << u.dimensionless_unscaled  # angular width
        wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

        with pytest.raises(ValueError):
            compute_glauert_norm_factor(
                self.cpdr, self.omega, X_range, wave_norm_angle_dist, method="unknown"
            )
