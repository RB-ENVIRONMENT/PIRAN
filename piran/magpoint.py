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
The `magpoint` module provides a class for representing the magnetic field
of a planet (defaulting to Earth) at a specific point in space. It calculates
the magnetic flux density based on magnetic latitude, L-shell value, planetary
radius, and magnetic dipole moment.

This module uses the dipole approximation for the planetary magnetic field.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity


class MagPoint:
    """
    Represents the magnetic field of a planet (by default Earth's) at a
    specific point defined by magnetic latitude and L-shell value.

    This class uses the dipole approximation to calculate the magnetic flux
    density.

    Parameters
    ----------
    mlat : astropy.coordinates.Angle[u.rad]
        The magnetic latitude, given in units convertible to radians.

    l_shell : float
        The "L-shell", "L-value", or "McIlwain L-parameter". A dimensionless
        quantity representing the radial distance of a magnetic field line
        from the center of the planet, normalised to the planetary radius.

    planetary_radius : astropy.units.Quantity[u.m], default=astropy.constants.R_earth
        The radius of the planet of interest, given in units convertible to meters.
        Defaults to Earth's radius.

    mag_dipole_moment : astropy.units.Quantity[u.tesla * u.m**3], default=8.033454e15 * (u.tesla * u.m**3)
        The magnetic dipole moment of the planet of interest, given in units convertible
        to Tesla metres cubed (T m^3). Defaults to Earth's magnetic dipole moment (8.033454e15 T m^3).

    Attributes
    ----------
    flux_density : astropy.units.Quantity[u.tesla]
        The magnetic flux density vector at the specified point, as an
        `astropy.units.Quantity` with units of Tesla.

    Notes
    -----
    The magnetic field is calculated using the dipole model, as described in
    Glauert & Horne (2005), paragraph 13.

    Examples
    --------
    >>> mlat_deg = Angle(0 * u.deg)
    >>> l_shell = 4.5 * u.dimensionless_unscaled
    >>> mag_point = MagPoint(mlat_deg, l_shell)
    """

    @u.quantity_input
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
