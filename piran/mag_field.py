"""
Defines the MagField class for use with the Cpdr.
"""

from astropy import constants as const
import numpy as np


class MagField:
    """
    A representation of the Earth's magnetic field at a given point.

    Parameters
    ----------
    mlat : float
        Geomagnetic latitude in radians.

    l_shell : float
        The "L-shell", "L-value", or "McIlwain L-parameter".
    """

    def __init__(
        self,
        mlat: float,
        l_shell: float,
    ) -> None:  # numpydoc ignore=GL08
        self._mlat = mlat
        self._l_shell = l_shell

        self._mag_dipole_moment = 8.033454e15

    def __call__(self) -> np.number:
        """
        Return the strength of the magnetic field.

        Returns
        -------
        np.number
            The strength of the magnetic field.
        """
        return (self._mag_dipole_moment * np.sqrt(1 + 3 * np.sin(self._mlat) ** 2)) / (
            self._l_shell**3 * const.R_earth**3 * np.cos(self._mlat) ** 6
        )
