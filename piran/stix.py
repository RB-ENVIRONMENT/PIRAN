"""
Defines the Stix class.
"""

from astropy import constants as const
from astropy import units as u


class Stix:
    @u.quantity_input
    def __init__(self, omega_p: u.Hz, omega_c: u.Hz) -> None:  # numpydoc ignore=GL08
        self._w_p = omega_p
        self._w_c = omega_c

    @u.quantity_input
    def R(self, w: u.Hz) -> u.dimensionless_unscaled:
        if not w.isscalar:
            raise ValueError("Frequency w should be a scalar")

        R = 1

        for idx in range(len(self._w_p)):
            R -= (self._w_p[idx] ** 2) / (w * (w + self._w_c[idx]))

        return R

    @u.quantity_input
    def L(self, w: u.Hz) -> u.dimensionless_unscaled:
        if not w.isscalar:
            raise ValueError("Frequency w should be a scalar")

        L = 1

        for idx in range(len(self._w_p)):
            L -= (self._w_p[idx] ** 2) / (w * (w - self._w_c[idx]))

        return L

    @u.quantity_input
    def P(self, w: u.Hz) -> u.dimensionless_unscaled:
        if not w.isscalar:
            raise ValueError("Frequency w should be a scalar")

        P = 1

        for idx in range(len(self._w_p)):
            P -= (self._w_p[idx] / w) ** 2

        return P

    @u.quantity_input
    def S(self, w: u.Hz) -> u.dimensionless_unscaled:
        return (self.R(w) + self.L(w)) / 2

    @u.quantity_input
    def D(self, w: u.Hz) -> u.dimensionless_unscaled:
        return (self.R(w) - self.L(w)) / 2

    @u.quantity_input
    def dR(self, w: u.Hz) -> u.s:
        if not w.isscalar:
            raise ValueError("Frequency w should be a scalar")

        R = 0

        for idx in range(len(self._w_p)):
            R += ((self._w_p[idx] ** 2) * (2 * w + self._w_c[idx])) / (
                (w**2) * ((w + self._w_c[idx]) ** 2)
            )

        return R

    @u.quantity_input
    def dL(self, w: u.Hz) -> u.s:
        if not w.isscalar:
            raise ValueError("Frequency w should be a scalar")

        L = 0

        for idx in range(len(self._w_p)):
            L += ((self._w_p[idx] ** 2) * (2 * w - self._w_c[idx])) / (
                (w**2) * ((w - self._w_c[idx]) ** 2)
            )

        return L

    @u.quantity_input
    def dP(self, w: u.Hz) -> u.s:
        if not w.isscalar:
            raise ValueError("Frequency w should be a scalar")

        P = 0

        for idx in range(len(self._w_p)):
            P += (2 * (self._w_p[idx] ** 2)) / (w**3)

        return P

    @u.quantity_input
    def dS(self, w: u.Hz) -> u.s:
        return (self.dR(w) + self.dL(w)) / 2

    @u.quantity_input
    def dD(self, w: u.Hz) -> u.s:
        return (self.dR(w) - self.dL(w)) / 2

    @u.quantity_input
    def A(self, w: u.Hz, X: u.dimensionless_unscaled) -> u.dimensionless_unscaled:
        return (self.S(w) * X**2) + self.P(w)

    @u.quantity_input
    def B(self, w: u.Hz, X: u.dimensionless_unscaled) -> u.dimensionless_unscaled:
        return (self.R(w) * self.L(w) * X**2) + ((self.P(w) * self.S(w)) * (2 + X**2))

    @u.quantity_input
    def C(self, w: u.Hz, X: u.dimensionless_unscaled) -> u.dimensionless_unscaled:
        return (self.P(w) * self.R(w) * self.L(w)) * (1 + X**2)

    @u.quantity_input
    def dA(self, w: u.Hz, X: u.dimensionless_unscaled) -> u.s:
        return (self.dS(w) * X**2) + self.dP(w)

    @u.quantity_input
    def dB(self, w: u.Hz, X: u.dimensionless_unscaled) -> u.s:
        return ((self.dR(w) * self.L(w) + self.R(w) * self.dL(w)) * (X**2)) + (
            (self.dP(w) * self.S(w) + self.P(w) * self.dS(w)) * (2 + X**2)
        )

    @u.quantity_input
    def dC(self, w: u.Hz, X: u.dimensionless_unscaled) -> u.s:
        return (
            self.dP(w) * self.R(w) * self.L(w)
            + self.P(w) * self.dR(w) * self.L(w)
            + self.P(w) * self.R(w) * self.dL(w)
        ) * (1 + X**2)

    @u.quantity_input
    def jacobian(
        self, w: u.Hz, X: u.dimensionless_unscaled, k: u.Quantity[1 / u.m]
    ) -> u.Quantity[u.s / u.m**2]:
        mu = const.c * k / w
        return ((k**2) / (1 + X**2)) * (
            (
                (self.dA(w, X) * mu**4 - self.dB(w, X) * mu**2 + self.dC(w, X))
                / (2 * (2 * self.A(w, X) * mu**4 - self.B(w, X) * mu**2))
            )
            - (1 / w)
        )

    @u.quantity_input
    def dD_dk(
        self, w: u.Hz, X: u.dimensionless_unscaled, k: u.Quantity[1 / u.m]
    ) -> u.m:
        mu = const.c * k / w

        return (2 / k) * (2 * self.A(w, X) * mu**4 - self.B(w, X) * mu**2)

    @u.quantity_input
    def dD_dw(
        self, w: u.Hz, X: u.dimensionless_unscaled, k: u.Quantity[1 / u.m]
    ) -> u.s:
        mu = const.c * k / w

        return (
            (self.dA(w, X) - 4 * self.A(w, X) / w) * mu**4
            - (self.dB(w, X) - 2 * self.B(w, X) / w) * mu**2
            + self.dC(w, X)
        )
