import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle


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

    def test_angles_1(self):
        # With Angle we can use .deg and .rad but remember
        # that they return np.float_, so we lose the unit.
        angle = Angle(90, unit=u.deg)

        assert angle.unit == u.deg
        assert math.isclose(angle.deg, 90)
        assert math.isclose(angle.rad, np.pi / 2)

        # AttributeError: 'numpy.float64' object has no attribute 'unit'
        with pytest.raises(AttributeError):
            angle.deg.unit
        with pytest.raises(AttributeError):
            angle.rad.unit

        # While u.Quantity doesn't have .deg or .rad
        # We need to use .to() and we keep the unit.
        quantity = 90 * u.deg

        assert quantity.unit == u.deg
        assert math.isclose(quantity.value, 90)
        assert math.isclose(quantity.to(u.rad).value, np.pi / 2)

        # AttributeError: 'Quantity' object has no ['deg', 'rad'] member
        with pytest.raises(AttributeError):
            quantity.deg
        with pytest.raises(AttributeError):
            quantity.rad
