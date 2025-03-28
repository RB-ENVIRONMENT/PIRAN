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

from piran.helpers import (
    calc_lorentz_factor,
    calc_momentum,
    get_real_and_positive_roots,
)


class TestHelpers:
    def test_calc_lorentz_factor_1(self):
        energy = 1.0 * u.MeV
        mass = 9.1093837015e-31 * u.kg

        gamma = calc_lorentz_factor(energy, mass)
        assert math.isclose(gamma, 2.9569511835738735)

    def test_calc_momentum_1(self):
        gamma = 2.9569511835738735
        mass = 9.1093837015e-31 << u.kg

        momentum = calc_momentum(gamma, mass)
        assert math.isclose(momentum.value, 7.5994128855e-22)
        assert momentum.unit == u.kg * u.m / u.s

    def test_get_real_and_positive_roots_1(self):
        test_array_1 = np.array(
            [
                0.0e00 + 0.0e00j,
                1.24900090e-16 - 1.0e0j,
                3.33066907e-16 + 1.0e0j,
                1.0e00 + 3.10894466e-17j,
            ]
        )
        valid_roots = get_real_and_positive_roots(test_array_1)
        assert valid_roots[0] == 1.0
        assert valid_roots.size == 1

        test_array_2 = np.array([-1 + 0j, 1.1 + 0.00000001j, 100 + 2j])
        valid_roots = get_real_and_positive_roots(test_array_2)
        assert valid_roots[0] == 1.1
        assert valid_roots.size == 1

        test_array_3 = np.array([-1, 0, 1])
        valid_roots = get_real_and_positive_roots(
            test_array_3
        )  # Works with integers only
        assert valid_roots[0] == 1.0
        assert valid_roots.size == 1

        test_array_4 = np.array([-1.0, 0.0, 0.00001, 1.0])
        valid_roots = get_real_and_positive_roots(
            test_array_4
        )  # Works with floats only too
        assert valid_roots[0] == 1e-5
        assert valid_roots[1] == 1.0
        assert valid_roots.size == 2
