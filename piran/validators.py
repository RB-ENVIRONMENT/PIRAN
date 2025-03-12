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

"""
Decorator functions for validating input/output.
"""

from functools import wraps
from typing import Callable

from astropy import units as u

_unit_check_enabled = True


def check_units(func: Callable) -> Callable:
    """
    A decorator wrapping Astropy's `units.quantity_input` decorator.

    By default, or following a call to `set_global_type_checking(True)`, this passes
    `func` over to `units.quantity_input` to validate units at `func` entry and exit.

    Following a call to `set_global_type_checking(False)`, this executes `func` directly
    (bypassing the unit validation performed by `units.quantity_input`).

    Skipping unit validation can decrease execution time of PIRAN code but should be
    considered unsafe.
    """

    @wraps(func)
    def wrapper_function(*args, **kwargs):
        return (
            u.quantity_input(func)(*args, **kwargs)
            if _unit_check_enabled
            else func(*args, **kwargs)
        )

    return wrapper_function


def set_global_type_checking(val: bool) -> None:
    """
    Enable/disable unit-checking performed by the `check_units` decorator.

    This sets a global flag on the `validators` module to True / False. If used, this
    setting will persist until this function is called again.

    Skipping unit validation (by calling `set_global_type_checking(False)`) can decrease
    execution time of PIRAN code but should be considered unsafe.

    Parameters
    ----------
    val : bool
        True to enable unit-checking (enabled by default);
        False to disable unit-checking.
    """
    global _unit_check_enabled
    _unit_check_enabled = val
