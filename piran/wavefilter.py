from abc import ABC, abstractmethod

import numpy as np
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
