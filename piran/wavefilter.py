from abc import ABC, abstractmethod

import numpy as np
from astropy import constants as const
from astropy import units as u

from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix


class WaveFilter(ABC):

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
        raise NotImplementedError


class DefaultFilter(WaveFilter):

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

    def filter(
        self,
        X: u.Quantity[u.dimensionless_unscaled],
        omega: u.Quantity[u.rad / u.s],
        k: u.Quantity[u.rad / u.m],
        plasma: PlasmaPoint,
        stix: Stix,
    ) -> u.Quantity[u.rad / u.m]:

        # Frequency for Whistlers does not exceed electron gyro frequency
        if omega > abs(plasma.gyro_freq[0]):
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
