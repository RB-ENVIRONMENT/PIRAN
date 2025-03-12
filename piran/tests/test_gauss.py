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

from piran.gauss import Gaussian


class TestGaussian:
    def test_gaussian_eval_1(self):
        """Test the `eval` method."""
        X_min = 0.0
        X_max = 1.0
        X_m = 0.0
        X_w = 0.577
        gauss = Gaussian(X_min, X_max, X_m, X_w)

        X_range = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        result = gauss.eval(X_range)
        expected = [0.0, 0.0, 1.0, 0.47193649125899506, 0.04960600324212487, 0.0]

        for i in range(result.shape[0]):
            assert math.isclose(result[i], expected[i])
