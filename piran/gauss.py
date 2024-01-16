"""
Defines the Gaussian class for use with Cpdr.
"""
import numpy as np
import numpy.typing


class Gaussian:
    """
    Implements a Gaussian distribution.

    The form of the distribution is exp(-((X - peak)/width)**2)) for X between lower and
    upper, or 0 otherwise.

    Parameters
    ----------
    lower : float
        The lower cutoff, below which Gaussian(X) returns 0.

    upper : float
        The upper cutoff, above which Gaussian(X) returns 0.

    peak : float
        The mean or expectation of the distribution.

    width : float
        The standard deviation of the distribution.
    """

    def __init__(
        self, lower: float, upper: float, peak: float, width: float
    ) -> None:  # numpydoc ignore=GL08
        self._lower = lower
        self._upper = upper
        self._peak = peak
        self._width = width

    def eval(self, X: float) -> numpy.typing.ArrayLike:
        """
        Return the value of the distribution at location(s) X.

        Parameters
        ----------
        X : numpy.typing.ArrayLike
            The location(s) at which the distribution is to be sampled.

        Returns
        -------
        numpy.typing.ArrayLike
            The value of the distribution at location(s) X.
        """
        return (
            (X >= self._lower)
            * (X <= self._upper)
            * np.exp(-1.0 * ((X - self._peak) / self._width) ** 2)
        )
