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

    for :math:`\\text{lower} \\le x \\le \\text{upper}`, and :math:`f(x) = 0` otherwise.
    Note that this is not normalised.

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


def from_gyrofrequency_params(gyrofreq, mean, delta, lower, upper) -> Gaussian:
    """
    Create a Gaussian distribution using gyrofrequency-based parameters.

    This function computes the lower and upper cutoffs, mean, and width for a truncated
    Gaussian distribution based on the supplied [electron] gyrofrequency and scaling
    parameters.

    Parameters
    ----------
    gyrofreq : float
        The gyrofrequency in rad/s.
    mean : float
        The scaling factor for the mean (center) of the distribution, as a multiple of `gyrofreq`.
    delta : float
        The scaling factor for the width (standard deviation) of the distribution, as a multiple of `gyrofreq`.
    lower : float
        The lower cutoff, as a multiple of the width away from the mean.
    upper : float
        The upper cutoff, as a multiple of the width away from the mean.

    Returns
    -------
    Gaussian
        A Gaussian object with the computed lower cutoff, upper cutoff, mean, and width.

    """
    omega_mean_cutoff = mean * abs(gyrofreq)
    omega_delta_cutoff = delta * abs(gyrofreq)

    omega_lc = omega_mean_cutoff + lower * omega_delta_cutoff
    omega_uc = omega_mean_cutoff + upper * omega_delta_cutoff

    return Gaussian(omega_lc, omega_uc, omega_mean_cutoff, omega_delta_cutoff)
