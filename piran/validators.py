from functools import wraps
from typing import Callable

from astropy import units as u

_enabled = True


def check_units(func: Callable) -> Callable:
    @wraps(func)
    def wrapper_function(*args, **kwargs):
        return (
            u.quantity_input(func)(*args, **kwargs)
            if _enabled
            else func(*args, **kwargs)
        )

    return wrapper_function


def set_global_type_checking(val: bool) -> None:
    global _enabled
    _enabled = val
