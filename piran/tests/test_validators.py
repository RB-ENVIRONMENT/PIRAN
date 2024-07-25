import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.magpoint import MagPoint
from piran.validators import set_global_type_checking


class TestValidators:

    def test_check_units_1(self):
        l_shell = 4.5
        mlat_quantity = Angle(45 * u.deg)
        mlat_int = 45

        # Type-checking for MagPoint should be on/True by default.
        # This should behave:
        MagPoint(mlat_quantity, l_shell)

        # This should fail type-checking and raise the following exception:
        # TypeError: Argument 'mlat' to function '__init__' has no 'unit' attribute.
        # You should pass in an astropy Quantity instead.
        with pytest.raises(TypeError):
            MagPoint(mlat_int, l_shell)

        # Lets turn type-checking off
        set_global_type_checking(False)

        # This should still behave:
        MagPoint(mlat_quantity, l_shell)

        # This should now bypass type-checking and raise a later exception in the body
        # of __init__:
        # AttributeError: 'int' object has no attribute 'to'
        with pytest.raises(AttributeError):
            MagPoint(mlat_int, l_shell)

        # Double-check that toggling type-checking back on works and that we see the
        # same results as before. In particular, we want to check that using `mlat_int`
        # returns to raising a `TypeError` (since mlat_quantity behaves in all cases).
        set_global_type_checking(True)

        MagPoint(mlat_quantity, l_shell)

        with pytest.raises(TypeError):
            MagPoint(mlat_int, l_shell)
