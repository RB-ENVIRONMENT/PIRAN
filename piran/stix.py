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
The `stix` module provides the `Stix` class for quickly calculating the Stix
parameters (and a few other things).

"""

import functools

from astropy import constants as const
from astropy import units as u


class Stix:
    """
    Provides methods for calculating the Stix parameters R, L, P, D, and S.
    This also includes methods for calculating the derivative of the CPDR with
    respect to :math:`k` and :math:`\omega`, as well as the jacobian
    :math:`J\\left(\\frac{k_\perp, k_\parallel}{\omega, X}\\right)`.

    We cache values of the plasma and cyclotron frequencies to avoid needing to
    provide them as arguments to every method.
    
    Parameters
    ----------
    omega_p : Quantity[u.rad / u.s]
        Plasma frequency.
    omega_c : Quantity[u.rad / u.s]
        Cyclotron frequency.
    """
    @u.quantity_input
    def __init__(
        self, omega_p: u.Quantity[u.rad / u.s], omega_c: u.Quantity[u.rad / u.s]
    ) -> None:  # numpydoc ignore=GL08
        self.__omega_p = omega_p
        self.__omega_c = omega_c

    @functools.lru_cache
    @u.quantity_input
    def R(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        """
        Calculate the R parameter.
        
        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            The wave frequency, which should be a scalar quantity with units of radians per second.
            
        Returns
        -------
        u.Quantity[u.dimensionless_unscaled]
            The calculated dimensionless quantity R.
                
        Raises
        ------
        ValueError
            If the wave frequency omega is not a scalar.
        """
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        R = 1

        for idx in range(len(self.__omega_p)):
            R -= (self.__omega_p[idx] ** 2) / (omega * (omega + self.__omega_c[idx]))

        return R

    @functools.lru_cache
    @u.quantity_input
    def L(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        """
        Calculate the L parameter.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            The wave frequency, which should be a scalar quantity with units of radians per second.
        
        Returns
        -------
        u.Quantity[u.dimensionless_unscaled]
            The calculated dimensionless quantity L.
        
        Raises
        ------
        ValueError
            If the wave frequency omega is not a scalar.
        """
        
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        L = 1

        for idx in range(len(self.__omega_p)):
            L -= (self.__omega_p[idx] ** 2) / (omega * (omega - self.__omega_c[idx]))

        return L

    @functools.lru_cache
    @u.quantity_input
    def P(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        """
        Calculate the P parameter.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            The wave frequency, which should be a scalar quantity with units of radians per second.
        
        Returns
        -------
        u.Quantity[u.dimensionless_unscaled]
            The calculated dimensionless quantity P.
        
        Raises
        ------
        ValueError
            If the wave frequency omega is not a scalar.
        """
        
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        P = 1

        for idx in range(len(self.__omega_p)):
            P -= (self.__omega_p[idx] / omega) ** 2

        return P

    @functools.lru_cache
    @u.quantity_input
    def S(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        """
        Calculate the S parameter.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            The wave frequency, which should be a scalar quantity with units of radians per second.
        
        Returns
        -------
        u.Quantity[u.dimensionless_unscaled]
            The calculated dimensionless quantity S.
        """
        return (self.R(omega) + self.L(omega)) / 2

    @functools.lru_cache
    @u.quantity_input
    def D(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        """
        Calculate the D parameter.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            The wave frequency, which should be a scalar quantity with units of radians per second.
        
        Returns
        -------
        u.Quantity[u.dimensionless_unscaled]
            The calculated dimensionless quantity D.
        """
        return (self.R(omega) - self.L(omega)) / 2

    @functools.lru_cache
    @u.quantity_input
    def _dR(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        R = 0

        for idx in range(len(self.__omega_p)):
            R += ((self.__omega_p[idx] ** 2) * (2 * omega + self.__omega_c[idx])) / (
                (omega**2) * ((omega + self.__omega_c[idx]) ** 2)
            )

        return R

    @functools.lru_cache
    @u.quantity_input
    def _dL(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        L = 0

        for idx in range(len(self.__omega_p)):
            L += ((self.__omega_p[idx] ** 2) * (2 * omega - self.__omega_c[idx])) / (
                (omega**2) * ((omega - self.__omega_c[idx]) ** 2)
            )

        return L

    @functools.lru_cache
    @u.quantity_input
    def _dP(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        P = 0

        for idx in range(len(self.__omega_p)):
            P += (2 * (self.__omega_p[idx] ** 2)) / (omega**3)

        return P

    @functools.lru_cache
    @u.quantity_input
    def _dS(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        return (self._dR(omega) + self._dL(omega)) / 2

    @functools.lru_cache
    @u.quantity_input
    def _dD(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        return (self._dR(omega) - self._dL(omega)) / 2

    @u.quantity_input
    def _A(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.S(omega) * X**2) + self.P(omega)

    @u.quantity_input
    def _B(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.R(omega) * self.L(omega) * X**2) + (
            (self.P(omega) * self.S(omega)) * (2 + X**2)
        )

    @u.quantity_input
    def _C(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.P(omega) * self.R(omega) * self.L(omega)) * (1 + X**2)

    @u.quantity_input
    def _dA(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.s / u.rad]:
        return (self._dS(omega) * X**2) + self._dP(omega)

    @u.quantity_input
    def _dB(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.s / u.rad]:
        return (
            (self._dR(omega) * self.L(omega) + self.R(omega) * self._dL(omega)) * (X**2)
        ) + (
            (self._dP(omega) * self.S(omega) + self.P(omega) * self._dS(omega))
            * (2 + X**2)
        )

    @u.quantity_input
    def _dC(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.s / u.rad]:
        return (
            self._dP(omega) * self.R(omega) * self.L(omega)
            + self.P(omega) * self._dR(omega) * self.L(omega)
            + self.P(omega) * self.R(omega) * self._dL(omega)
        ) * (1 + X**2)

    @u.quantity_input
    def jacobian(
        self,
        omega: u.Quantity[u.rad / u.s],
        X: u.Quantity[u.dimensionless_unscaled],
        k: u.Quantity[u.rad / u.m],
    ) -> u.Quantity[u.rad * u.s / u.m**2]:
        """
        Calculate the value of the Jacobian
        :math:`J\\left(\\frac{k_\perp, k_\parallel}{\omega, X}\\right)`.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            Wave frequency.
        X : u.Quantity[u.dimensionless_unscaled]
            Wave normal angles.
        k : u.Quantity[u.rad / u.m]
            Wavenumber.

        Returns
        -------
        u.Quantity[u.rad * u.s / u.m**2]
            The calculated Jacobian value.

        """
        mu = const.c * k / omega
        return ((k**2) / (1 + X**2)) * (
            (
                (
                    self._dA(omega, X) * mu**4
                    - self._dB(omega, X) * mu**2
                    + self._dC(omega, X)
                )
                / (2 * (2 * self._A(omega, X) * mu**4 - self._B(omega, X) * mu**2))
            )
            - (1 / omega)
        )

    @u.quantity_input
    def dD_dk(
        self,
        omega: u.Quantity[u.rad / u.s],
        X: u.Quantity[u.dimensionless_unscaled],
        k: u.Quantity[u.rad / u.m],
    ) -> u.Quantity[u.m / u.rad]:
        """
        Calculate the value of the derivative of the CPDR with respect to the
        wavenumber :math:`k`.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            Wave frequency.
        X : u.Quantity[u.dimensionless_unscaled]
            Wave normal angles.
        k : u.Quantity[u.rad / u.m]
            Wavenumber.

        Returns
        -------
        u.Quantity[u.m / u.rad]
            The value of the derivative of the CPDR with respect to the wavenumber
            :math:`k`.
        """
        mu = const.c * k / omega

        return (2 / k) * (2 * self._A(omega, X) * mu**4 - self._B(omega, X) * mu**2)

    @u.quantity_input
    def dD_dw(
        self,
        omega: u.Quantity[u.rad / u.s],
        X: u.Quantity[u.dimensionless_unscaled],
        k: u.Quantity[u.rad / u.m],
    ) -> u.Quantity[u.s / u.rad]:
        """
        Calculate the value of the derivative of the CPDR with respect to the wave
        frequency :math:`\omega`.

        Parameters
        ----------
        omega : u.Quantity[u.rad / u.s]
            Wave frequency.
        X : u.Quantity[u.dimensionless_unscaled]
            Wave normal angles.
        k : u.Quantity[u.rad / u.m]
            Wavenumber.

        Returns
        -------
        u.Quantity[u.s / u.rad]
            The value of the derivative of the CPDR with respect to the wave
            frequency :math:`\omega`.
        """
        mu = const.c * k / omega

        return (
            (self._dA(omega, X) - 4 * self._A(omega, X) / omega) * mu**4
            - (self._dB(omega, X) - 2 * self._B(omega, X) / omega) * mu**2
            + self._dC(omega, X)
        )
