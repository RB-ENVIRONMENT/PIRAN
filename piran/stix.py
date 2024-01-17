"""
Defines the Stix class.
"""

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
