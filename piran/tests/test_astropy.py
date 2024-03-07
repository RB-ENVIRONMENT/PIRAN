import math

import numpy as np
from astropy import units as u


class TestAstropy:
    def test_units_1(self):
        """
        Arctan of a dimensionless unscaled returns radians.
        """
        # Multiplying arrays by units will result in the array
        # being copied in memory.
        xx = 1.0 * u.dimensionless_unscaled
        yy = np.arctan(xx)

        assert yy.unit == u.rad
        assert math.isclose(yy.value, 0.785398163)

        assert yy.to(u.deg).unit == u.deg
        assert math.isclose(yy.to(u.deg).value, 45.0)

        # Same but now using << operator which attaches the unit
        # to the array.
        xx = 1.0 << u.dimensionless_unscaled
        yy = np.arctan(xx)

        assert yy.unit == u.rad
        assert math.isclose(yy.value, 0.785398163)

        assert yy.to(u.deg).unit == u.deg
        assert math.isclose(yy.to(u.deg).value, 45.0)

        # And if we multiply again with rad we get squared radian (steradian)
        xx = 1.0 << u.dimensionless_unscaled
        yy = np.arctan(xx) * u.rad

        assert yy.unit == u.steradian
