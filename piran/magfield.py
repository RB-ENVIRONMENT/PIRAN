"""
Defines the MagField class for use with the Cpdr.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle


class MagField:
    """
    A representation of the magnetic field of a planet (by default Earth's).

    Parameters
    ----------
    planetary_radius : astropy.units.Quantity[u.m]
        The radius of the planet of interest, given in units convertible to meters.

    planetary_mag_dipole_moment : astropy.units.Quantity[u.tesla * u.m**3]
        The magnetic dipole moment of the planet of interest, given in units convertible
        to tesla metres cubed.
    """

    @u.quantity_input
    def __init__(
        self,
        planetary_radius: u.Quantity[u.m] = const.R_earth,
        planetary_mag_dipole_moment: u.Quantity[u.tesla * u.m**3] = 8.033454e15
        * (u.tesla * u.m**3),
    ) -> None:  # numpydoc ignore=GL08
        self._radius = planetary_radius.to_value(u.m)
        self._mag_dipole_moment = planetary_mag_dipole_moment.to_value(u.tesla * u.m**3)

    def get_strength(
        self,
        mlat: Angle,
        l_shell: float,
    ) -> np.number:
        """
        Calculates the strength of the magnetic field.

        Parameters
        ----------
        mlat : astropy.coordinates.Angle
            Geomagnetic latitude in radians.

        l_shell : float
            The "L-shell", "L-value", or "McIlwain L-parameter".

        Returns
        -------
        np.number
            The strength of the magnetic field.
        """
        return (self._mag_dipole_moment * np.sqrt(1 + 3 * np.sin(mlat) ** 2)) / (
            l_shell**3 * self._radius**3 * np.cos(mlat) ** 6
        )
