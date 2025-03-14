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
Defines the Stix class.
"""

import functools

from astropy import constants as const
from astropy import units as u


class Stix:
    @u.quantity_input
    def __init__(
        self, omega_p: u.Quantity[u.rad / u.s], omega_c: u.Quantity[u.rad / u.s]
    ) -> None:  # numpydoc ignore=GL08
        self.__omega_p = omega_p
        self.__omega_c = omega_c

    @functools.lru_cache
    @u.quantity_input
    def R(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        R = 1

        for idx in range(len(self.__omega_p)):
            R -= (self.__omega_p[idx] ** 2) / (omega * (omega + self.__omega_c[idx]))

        return R

    @functools.lru_cache
    @u.quantity_input
    def L(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        L = 1

        for idx in range(len(self.__omega_p)):
            L -= (self.__omega_p[idx] ** 2) / (omega * (omega - self.__omega_c[idx]))

        return L

    @functools.lru_cache
    @u.quantity_input
    def P(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        P = 1

        for idx in range(len(self.__omega_p)):
            P -= (self.__omega_p[idx] / omega) ** 2

        return P

    @functools.lru_cache
    @u.quantity_input
    def S(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.R(omega) + self.L(omega)) / 2

    @functools.lru_cache
    @u.quantity_input
    def D(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.R(omega) - self.L(omega)) / 2

    @functools.lru_cache
    @u.quantity_input
    def dR(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
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
    def dL(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
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
    def dP(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        if not omega.isscalar:
            raise ValueError("Frequency omega should be a scalar")

        P = 0

        for idx in range(len(self.__omega_p)):
            P += (2 * (self.__omega_p[idx] ** 2)) / (omega**3)

        return P

    @functools.lru_cache
    @u.quantity_input
    def dS(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        return (self.dR(omega) + self.dL(omega)) / 2

    @functools.lru_cache
    @u.quantity_input
    def dD(self, omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.s / u.rad]:
        return (self.dR(omega) - self.dL(omega)) / 2

    @u.quantity_input
    def A(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.S(omega) * X**2) + self.P(omega)

    @u.quantity_input
    def B(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.R(omega) * self.L(omega) * X**2) + (
            (self.P(omega) * self.S(omega)) * (2 + X**2)
        )

    @u.quantity_input
    def C(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.dimensionless_unscaled]:
        return (self.P(omega) * self.R(omega) * self.L(omega)) * (1 + X**2)

    @u.quantity_input
    def dA(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.s / u.rad]:
        return (self.dS(omega) * X**2) + self.dP(omega)

    @u.quantity_input
    def dB(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.s / u.rad]:
        return (
            (self.dR(omega) * self.L(omega) + self.R(omega) * self.dL(omega)) * (X**2)
        ) + (
            (self.dP(omega) * self.S(omega) + self.P(omega) * self.dS(omega))
            * (2 + X**2)
        )

    @u.quantity_input
    def dC(
        self, omega: u.Quantity[u.rad / u.s], X: u.Quantity[u.dimensionless_unscaled]
    ) -> u.Quantity[u.s / u.rad]:
        return (
            self.dP(omega) * self.R(omega) * self.L(omega)
            + self.P(omega) * self.dR(omega) * self.L(omega)
            + self.P(omega) * self.R(omega) * self.dL(omega)
        ) * (1 + X**2)

    @u.quantity_input
    def jacobian(
        self,
        omega: u.Quantity[u.rad / u.s],
        X: u.Quantity[u.dimensionless_unscaled],
        k: u.Quantity[u.rad / u.m],
    ) -> u.Quantity[u.rad * u.s / u.m**2]:
        mu = const.c * k / omega
        return ((k**2) / (1 + X**2)) * (
            (
                (
                    self.dA(omega, X) * mu**4
                    - self.dB(omega, X) * mu**2
                    + self.dC(omega, X)
                )
                / (2 * (2 * self.A(omega, X) * mu**4 - self.B(omega, X) * mu**2))
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
        mu = const.c * k / omega

        return (2 / k) * (2 * self.A(omega, X) * mu**4 - self.B(omega, X) * mu**2)

    @u.quantity_input
    def dD_dw(
        self,
        omega: u.Quantity[u.rad / u.s],
        X: u.Quantity[u.dimensionless_unscaled],
        k: u.Quantity[u.rad / u.m],
    ) -> u.Quantity[u.s / u.rad]:
        mu = const.c * k / omega

        return (
            (self.dA(omega, X) - 4 * self.A(omega, X) / omega) * mu**4
            - (self.dB(omega, X) - 2 * self.B(omega, X) / omega) * mu**2
            + self.dC(omega, X)
        )
