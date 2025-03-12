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

import pytest
from astropy import units as u
from astropy.coordinates import Angle
from astropy.units import Quantity

from piran.validators import check_units, set_global_type_checking


class CheckUnits:

    @check_units
    def __init__(
        self,
        quant: Quantity[u.rad],
    ) -> None:
        self.__quant = quant.to(u.rad)


class TestValidators:

    def test_check_units_1(self):
        a_quantity = Angle(45 * u.deg)
        an_int = 45

        # Type-checking for MagPoint should be on/True by default.
        # This should behave:
        CheckUnits(a_quantity)

        # This should fail type-checking and raise the following exception:
        # TypeError: Argument 'quant' to function '__init__' has no 'unit' attribute.
        # You should pass in an astropy Quantity instead.
        with pytest.raises(TypeError):
            CheckUnits(an_int)

        # Lets turn type-checking off
        set_global_type_checking(False)

        # This should still behave:
        CheckUnits(a_quantity)

        # This should now bypass type-checking and raise a later exception in the body
        # of __init__:
        # AttributeError: 'int' object has no attribute 'to'
        with pytest.raises(AttributeError):
            CheckUnits(an_int)

        # Double-check that toggling type-checking back on works and that we see the
        # same results as before. In particular, we want to check that using `an_int`
        # returns to raising a `TypeError` (since a_quantity behaves in all cases).
        set_global_type_checking(True)

        CheckUnits(a_quantity)

        with pytest.raises(TypeError):
            CheckUnits(an_int)
