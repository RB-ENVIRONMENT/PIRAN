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
