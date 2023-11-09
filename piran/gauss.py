from typing import Sequence


class Gaussian:
    def __init__(self, range: Sequence[float], peak: float, width: float) -> None:
        # if range is < 2 values, assume we are only interested in a single value.
        # if range is exactly 2 values, treate this as a lower and upper cutoff
        # that we will generate a linspace'd set of values from ?
        # if range is > 2 values, treat this as a full list of values to use.
        self._range = range
        self._peak = peak
        self._width = width
