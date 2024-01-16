"""
Defines the MagField class for use with the Cpdr.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle


class MagField:
    """
    A representation of the Earth's magnetic field at a given point.

    Parameters
    ----------
    mlat : float
        Geomagnetic latitude in radians.

    l_shell : float
        The "L-shell", "L-value", or "McIlwain L-parameter".

    planetary_radius : astropy.units.Quantity[u.m]
        The radius of the planet of interest, given in units convertible to meters.

    planetary_mag_dipole_moment : astropy.units.Quantity[u.tesla * u.m**3]
        The magnetic dipole moment of the planet of interest, given in units convertible
        to tesla metres cubed.
    """

    @u.quantity_input
    def __init__(
        self,
        mlat: Angle,
        l_shell: float,
        planetary_radius: u.Quantity[u.m] = const.R_earth,
        planetary_mag_dipole_moment: u.Quantity[u.tesla * u.m**3] = 8.033454e15
        * (u.tesla * u.m**3),
    ) -> None:  # numpydoc ignore=GL08
        self._mlat = mlat.rad
        self._l_shell = l_shell
        self._radius = planetary_radius.to(u.m)
        self._mag_dipole_moment = planetary_mag_dipole_moment.to(u.tesla * u.m**3)

    def __call__(self) -> u.Quantity[u.tesla]:
        """
        Calculates the strength of the magnetic field.

        Returns
        -------
        u.Quantity[u.tesla]
            The strength of the magnetic field.
        """
        return (
            (self._mag_dipole_moment * np.sqrt(1 + 3 * np.sin(self._mlat) ** 2))
            / (self._l_shell**3 * self._radius**3 * np.cos(self._mlat) ** 6)
        ).to(u.tesla)
