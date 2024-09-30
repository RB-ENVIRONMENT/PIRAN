"""
Defines the MagPoint class.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity


class MagPoint:
    """
    A representation of the magnetic field of a planet (by default Earth's),
    at a specific point defined by magnetic latitude and L-shell value.
    Given magnetic latitude, L-shell value, radius and magnetic dipole moment,
    we compute the magnetic flux density and store it in self.flux_density.

    Parameters
    ----------
    mlat : astropy.coordinates.Angle[u.rad]
        The magnetic latitude, given in units convertible to radians.

    l_shell : float
        The "L-shell", "L-value", or "McIlwain L-parameter".

    planetary_radius : astropy.units.Quantity[u.m],
            default=astropy.constants.R_earth
        The radius of the planet of interest, given in units convertible to meters.

    mag_dipole_moment : astropy.units.Quantity[u.tesla * u.m**3],
            default=8.033454e15 * (u.tesla * u.m**3)
        The magnetic dipole moment of the planet of interest, given in units convertible
        to tesla metres cubed.
    """

    def __init__(
        self,
        mlat: Quantity[u.rad],
        l_shell: float,
        planetary_radius: Quantity[u.m] = const.R_earth,
        mag_dipole_moment: Quantity[u.tesla * u.m**3] = 8.033454e15
        * (u.tesla * u.m**3),
    ) -> None:
        self.__mlat = mlat.to(u.rad)
        self.__l_shell = l_shell
        self.__planetary_radius = planetary_radius.to(u.m)
        self.__mag_dipole_moment = mag_dipole_moment.to(u.tesla * u.m**3)
        self.__flux_density = self.__compute_flux_density()  # T

    @property
    def mlat(self):
        return self.__mlat

    @property
    def l_shell(self):
        return self.__l_shell

    @property
    def planetary_radius(self):
        return self.__planetary_radius

    @property
    def mag_dipole_moment(self):
        return self.__mag_dipole_moment

    @property
    def flux_density(self):
        return self.__flux_density

    def __compute_flux_density(self) -> Quantity[u.tesla]:
        """
        Calculates the magnetic flux density.

        Returns
        -------
        astropy.units.Quantity[u.tesla]
            The magnetic flux density.
        """
        return (
            (self.__mag_dipole_moment * np.sqrt(1 + 3 * np.sin(self.__mlat) ** 2))
            / (
                self.__l_shell**3
                * self.__planetary_radius**3
                * np.cos(self.__mlat) ** 6
            )
        ).to(u.tesla)
