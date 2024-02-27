import math

import numpy as np
from astropy import units as u

from piran.helpers import calc_lorentz_factor, get_valid_roots


class TestHelpers:
    def test_calc_lorentz_factor_1(self):
        energy = 1.0 * u.MeV
        mass = 9.1093837015e-31 * u.kg

        gamma = calc_lorentz_factor(energy, mass)
        assert math.isclose(gamma, 2.9569511835738735)

    def test_get_valid_roots_1(self):
        test_array_1 = np.array([0.0e+00 + 0.0e+00j,
                                 1.24900090e-16 - 1.0e+0j,
                                 3.33066907e-16 + 1.0e+0j,
                                 1.0e+00 + 3.10894466e-17j])
        valid_roots = get_valid_roots(test_array_1)
        assert valid_roots[0] == 1.0
        assert valid_roots.size == 1

        test_array_2 = np.array([-1 + 0j, 1.1 + 0.00000001j, 100 + 2j])
        valid_roots = get_valid_roots(test_array_2)
        assert valid_roots[0] == 1.1
        assert valid_roots.size == 1

        test_array_3 = np.array([-1, 0, 1])
        valid_roots = get_valid_roots(test_array_3) # Works with integers only
        assert valid_roots[0] == 1.0
        assert valid_roots.size == 1

        test_array_4 = np.array([-1.0, 0.0, 0.00001, 1.0])
        valid_roots = get_valid_roots(test_array_4) # Works with floats only too
        assert valid_roots[0] == 1e-5
        assert valid_roots[1] == 1.0
        assert valid_roots.size == 2
