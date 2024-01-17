"""
Defines the Stix class.
"""

from typing import Sequence


class Stix:
    def __init__(
        self, omega_p: Sequence[float], omega_c: Sequence[float]
    ) -> None:  # numpydoc ignore=GL08
        self._w_p = omega_p
        self._w_c = omega_c

    def R(self, w) -> float:
        R = 1

        for idx in range(len(self._w_p)):
            R -= (self._w_p[idx] ** 2) / (w * (w + self._w_c[idx]))

        return R

    def L(self, w) -> float:
        L = 1

        for idx in range(len(self._w_p)):
            L -= (self._w_p[idx] ** 2) / (w * (w - self._w_c[idx]))

        return L

    def P(self, w) -> float:
        P = 1

        for idx in range(len(self._w_p)):
            P -= (self._w_p[idx] / w) ** 2

        return P

    def S(self, w) -> float:
        return (self.R(w) + self.L(w)) / 2

    def D(self, w) -> float:
        return (self.R(w) - self.L(w)) / 2
