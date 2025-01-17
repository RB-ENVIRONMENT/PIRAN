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
