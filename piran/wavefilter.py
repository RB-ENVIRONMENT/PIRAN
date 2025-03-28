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

from abc import ABC, abstractmethod

import numpy as np
from astropy import constants as const
from astropy import units as u

from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix


class WaveFilter(ABC):
    """
    An ABC for filtering solutions to the CPDR in/out depending on whether they
    correspond to a desired wave mode.

    This provides a single method, `filter`, that is called from within the Cpdr class
    to determine whether an `(X, omega, k)` triplet belongs to the desired wave mode.

    It is the responsibility of the user to subclass `WaveFilter` and implement the
    `filter` method themselves! The function signature should match the `filter`
    method exactly. An object of the subclass should then be passed to the `Cpdr` on
    creation in order for the `filter` method to be called within the `Cpdr` solution
    algorithm.

    An example subclass is given with the `WhistlerFilter` class for identifying
    Whistler waves.
    """

    @abstractmethod
    @u.quantity_input
    def filter(
        self,
        X: u.Quantity[u.dimensionless_unscaled],
        omega: u.Quantity[u.rad / u.s],
        k: u.Quantity[u.rad / u.m],
        plasma: PlasmaPoint,
        stix: Stix,
    ) -> bool:
        """
        Given [resonant] solution(s) to the CPDR, filter solutions in/out depending on
        criteria defined within this function.

        `X` `omega` and `k` are expected to be 0d;
        each `(X, omega, k)` triplet is a [resonant] solution to the CPDR.

        If using this within the `Cpdr`, there is no need for the user to supply the
        input parameters below (this is handled within the `Cpdr` methods already).

        Parameters
        ----------
        X : Quantity[u.dimensionless_unscaled]
            Wave normal angle.
            Size: 0d.
        omega : Quantity[u.rad / u.s]
            Wave frequency.
            Size: 0d.
        k : Quantity[u.rad / u.m]
            Wavenumber.
            Size: 0d.
        plasma : PlasmaPoint
            Plasma composition (e.g. particle plasma- and gyro-frequencies).
        stix: Stix
            Methods for calculating Stix parameters.

        Returns
        -------
        Boolean
            True if `(X, omega, k)` satisfy the criteria defined within this
            function (i.e. fits the desired wave mode), or False otherwise.
        """

        raise NotImplementedError


class WhistlerFilter(WaveFilter):
    """
    Filter solutions to the CPDR to accept only Whistler mode waves.

    For more info on the selection criteria used in this function, see:
    Introduction to Plasma Physics (Second Edition) by Gurnett & Bhattacharjee,
    Section 4.4.4, Figures 4.37 and 4.38.
    """

    @u.quantity_input
    def filter(
        self,
        X: u.Quantity[u.dimensionless_unscaled],
        omega: u.Quantity[u.rad / u.s],
        k: u.Quantity[u.rad / u.m],
        plasma: PlasmaPoint,
        stix: Stix,
    ) -> bool:

        # Frequency for Whistlers does not exceed electron plasma- or gyro-frequency
        if omega > min(abs(plasma.gyro_freq[0]), abs(plasma.plasma_freq[0])):
            return False

        # The square of the index of refraction is:
        # - bounded by R below
        # - unbounded above

        # Calculate index of refraction for all k.
        # Exclude any k for which index of refraction does not exceed R.
        mu2 = (const.c * k / omega) ** 2

        # Due to floating point arithmetic we might get
        # mu2 slightly smaller than R while this solution is still
        # a whistler-mode wave. The following is equivalent to
        # mu2 >= R - max(rel_tol * abs(R), abs_tol)
        cmp1 = mu2 >= stix.R(omega)
        cmp2 = np.isclose(mu2, stix.R(omega), rtol=1e-04, atol=1e-09)

        if cmp1 or cmp2:
            return True
        else:
            return False
