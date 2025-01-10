"""
The `gauss` module provides a class for representing and evaluating
truncated Gaussian distributions.

This module defines the `Gaussian` class, which allows users to create
Gaussian distributions with specified lower and upper cutoffs. The
`eval` method of the `Gaussian` class can then be used to evaluate
the distribution at given locations.
"""

import numpy as np


class Gaussian:
    """
    Implements a truncated Gaussian distribution.

    The distribution is defined as:

    .. math::
        f(x) = \\exp\\left(-\\frac{(x - \\mu)^2}{\\sigma^2}\\right)

    for :math:`lower \\le x \\le upper`, and 0 otherwise. Note that this is not
    normalised.

    Parameters
    ----------
    lower : float
        The lower cutoff; values of *x* below this are treated as having zero probability.
    upper : float
        The upper cutoff; values of *x* above this are treated as having zero probability.
    peak : float
        The mean (:math:`\\mu`) or expectation of the distribution.
    width : float
        The standard deviation (:math:`\\sigma`) of the distribution.

    Examples
    --------
    >>> X_min = 0.0
    >>> X_max = 1.0
    >>> X_m = 0.0
    >>> X_w = 0.577
    >>> gaussian = Gaussian(X_min, X_max, X_m, X_w)
    """

    def __init__(
        self, lower: float, upper: float, peak: float, width: float
    ) -> None:  # numpydoc ignore=GL08
        self._lower = lower
        self._upper = upper
        self._peak = peak
        self._width = width

    def eval(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gaussian distribution at the given locations.

        Parameters
        ----------
        X : np.ndarray
            The location(s) at which to evaluate the distribution.

        Returns
        -------
        np.ndarray
            The value(s) of the Gaussian distribution at the given location(s).
            Returns 0 if the input X is outside of the range [lower, upper].

        Raises
        ------
        TypeError:
            If the input X is of incorrect type (for example a list).

        Examples
        --------
        >>> from astropy import units as u
        >>> gaussian = Gaussian(0.0, 1.0, 0.0, 0.577)
        >>> print(gaussian.eval([-1, 0, 1] * u.dimensionless_unscaled))
        [0.       1.       0.049606]
        """
        return (
            (X >= self._lower)
            * (X <= self._upper)
            * np.exp(-1.0 * ((X - self._peak) / self._width) ** 2)
        )
