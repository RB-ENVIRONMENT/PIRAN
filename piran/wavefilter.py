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
    ) -> u.Quantity[u.rad / u.m]:
        """
        Given [resonant] solution(s) to the CPDR, filter solutions in/out depending on
        criteria defined within this function.

        `X` and `omega` are expected to be 0d while `k` is 1d with up to two values;
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
            Size: 1d.
        plasma : PlasmaPoint
            Plasma composition (e.g. particle plasma- and gyro-frequencies).
        stix: Stix
            Methods for calculating Stix parameters.


        Returns
        -------
        u.Quantity[u.rad / u.s]
            Any `k` for which `(X, omega, k)` satisfies the criteria defined within this
            function (i.e. fits the desired wave mode).
        """

        raise NotImplementedError


class DefaultFilter(WaveFilter):
    """
    Filter solutions to the CPDR, raising an exception if we have more than one value
    of `k` for a given `(X, omega)` pair.
    """

    @u.quantity_input
    def filter(
        self,
        X: u.Quantity[u.dimensionless_unscaled],
        omega: u.Quantity[u.rad / u.s],
        k: u.Quantity[u.rad / u.m],
        plasma: PlasmaPoint,
        stix: Stix,
    ) -> u.Quantity[u.rad / u.m]:

        if k.size == 0:
            return np.nan << u.rad / u.m
        elif k.size == 1:
            return k[0]
        else:
            msg = "We got more than one real positive root for k"
            raise ValueError(msg)


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
    ) -> u.Quantity[u.rad / u.m]:

        # Frequency for Whistlers does not exceed electron plasma- or gyro-frequency
        if omega > min(abs(plasma.gyro_freq[0]), abs(plasma.plasma_freq[0])):
            return np.nan << u.rad / u.m

        # The square of the index of refraction is:
        # - bounded by R below
        # - unbounded above

        k = np.atleast_1d(k)

        # Calculate index of refraction for all k.
        # Exclude any k for which index of refraction does not exceed R.
        mu2 = (const.c * k / omega) ** 2
        k = k[mu2 > stix.R(omega)]

        return k if k.size > 0 else np.nan << u.rad / u.m
