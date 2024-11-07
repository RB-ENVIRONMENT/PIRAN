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
