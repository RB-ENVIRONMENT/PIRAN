import math
from typing import NamedTuple, Sequence

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity

from piran.cpdrsymbolic import CpdrSymbolic
from piran.gauss import Gaussian
from piran.helpers import calc_lorentz_factor, get_real_and_positive_roots
from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix


class Cpdr:
    """
    The first particle is expected to be the particle of interest
    and the one for which energy and pitch angle is provided.

    Parameters
    ----------
    energy : Quantity[u.Joule] | None = None
        Relativistic kinetic energy.

    freq_cutoff_params : Sequence[float] | None = None
        Frequency cutoff parameters (mean_factor, delta_factor, l_factor, u_factor)
    """

    @u.quantity_input
    def __init__(
        self,
        symbolic: CpdrSymbolic,
        plasma: PlasmaPoint,
        energy: Quantity[u.Joule] | None = None,
        pitch_angle: Quantity[u.rad] | None = None,
        resonance: int | None = None,
        freq_cutoff_params: Sequence[float] | None = None,
    ) -> None:
        self.__symbolic = symbolic
        self.__plasma = plasma

        # CPDR after replacing gyro and plasma frequencies
        self.__poly_in_k = self.__symbolic.poly_in_k.subs(
            {
                "omega_c": tuple(self.__plasma.gyro_freq.value),
                "omega_p": tuple(self.__plasma.plasma_freq.value),
            }
        )

        # CPDR derivatives
        self.__poly_in_k_dk = self.__poly_in_k.diff("k")
        self.__poly_in_k_domega = self.__poly_in_k.diff("omega")

        # Stix parameters
        self.__stix = Stix(self.__plasma.plasma_freq, self.__plasma.gyro_freq)

        if (
            energy is not None
            and pitch_angle is not None
            and resonance is not None
            and freq_cutoff_params is not None
        ):
            self.__energy = energy.to(u.Joule)
            self.__pitch_angle = pitch_angle.to(u.rad)
            self.__resonance = resonance
            self.__gamma = calc_lorentz_factor(
                self.__energy,
                self.__plasma.particles[0].mass,
            )
            self.__rel_velocity = const.c * np.sqrt(1 - (1 / self.__gamma**2))
            self.__v_par = self.__rel_velocity * np.cos(self.__pitch_angle)
            self.__resonant_poly_in_omega = self.__symbolic.resonant_poly_in_omega.subs(
                {
                    "omega_c": tuple(self.__plasma.gyro_freq.value),
                    "omega_p": tuple(self.__plasma.plasma_freq.value),
                    "gamma": self.__gamma.value,
                    "n": self.__resonance,
                    "v_par": self.__v_par.value,
                }
            )
            omega_mean_cutoff = freq_cutoff_params[0] * abs(self.__plasma.gyro_freq[0])
            omega_delta_cutoff = freq_cutoff_params[1] * abs(self.__plasma.gyro_freq[0])
            self.__omega_lc = (
                omega_mean_cutoff + freq_cutoff_params[2] * omega_delta_cutoff
            )
            self.__omega_uc = (
                omega_mean_cutoff + freq_cutoff_params[3] * omega_delta_cutoff
            )
            self.__wave_freqs = Gaussian(
                self.__omega_lc, self.__omega_uc, omega_mean_cutoff, omega_delta_cutoff
            )

    @property
    def symbolic(self):
        return self.__symbolic

    @property
    def plasma(self):
        return self.__plasma

    @property
    def poly_in_k(self):
        return self.__poly_in_k

    @property
    def poly_in_k_dk(self):
        return self.__poly_in_k_dk

    @property
    def poly_in_k_domega(self):
        return self.__poly_in_k_domega

    @property
    def stix(self):
        return self.__stix

    @property
    def resonant_poly_in_omega(self):
        return self.__resonant_poly_in_omega

    @property
    def energy(self):
        return self.__energy

    @property
    def pitch_angle(self):
        return self.__pitch_angle

    @property
    def alpha(self):
        return self.__pitch_angle

    @property
    def resonance(self):
        return self.__resonance

    @property
    def lorentz_factor(self):
        return self.__gamma

    @property
    def gamma(self):
        return self.__gamma

    @property
    def rel_velocity(self):
        return self.__rel_velocity

    @property
    def v_par(self):
        return self.__v_par

    @property
    def omega_lc(self):
        return self.__omega_lc

    @property
    def omega_uc(self):
        return self.__omega_uc

    @property
    def wave_freqs(self):
        return self.__wave_freqs

    @u.quantity_input
    def solve_cpdr_for_norm_factor(
        self,
        omega: Quantity[u.rad / u.s],
        X_range: Quantity[u.dimensionless_unscaled],
    ) -> Sequence[float]:
        """
        Given wave frequency omega, solve the dispersion relation for each
        wave normal angle X=tan(psi) in X_range to get wave number k.
        Optimised version, similar to solve_cpdr, but we lambdify in X
        after we substitute omega and is more efficient when we have
        a single value for omega and a range of X values (for example when
        computing the normalisation factor).

        Parameters
        ----------
        omega : astropy.units.quantity.Quantity convertible to rad/second
            Wave frequency.
        X_range : astropy.units.quantity.Quantity or float
            Wave normal angles.

        Returns
        -------
        k_sol : list of k
            The k are given in the same order as X_range.
            This means that each (X, omega, k) triplet is a solution
            to the cold plasma dispersion relation.
            If we get NaN then for this X, omega pair the CPDR has no
            roots.
        """

        # Replace omega and lambdify
        cpdr_in_X_k = sym.lambdify(
            ["X"], self.__poly_in_k.subs({"omega": omega.value}), "numpy"
        )

        k_sol = []
        for i, X in enumerate(X_range):
            # We've lambified `X` but `k` is still a symbol. When we call it with an
            # argument it substitutes `X` with the value and returns a
            # `sympy.core.add.Add` object, that's why calling `numpy.roots`
            # still works.
            cpdr_in_k = cpdr_in_X_k(X.value)
            k_l = np.roots(cpdr_in_k.as_poly().all_coeffs())
            valid_k_l = get_real_and_positive_roots(k_l)

            if valid_k_l.size == 0:
                k_sol.append(np.nan)
            elif valid_k_l.size == 1:
                k_sol.append((valid_k_l[0]))
            else:
                msg = "We got more than one real positive root for k."
                raise ValueError(msg)

        return k_sol

    @u.quantity_input
    def solve_resonant(
        self,
        X_range: Quantity[u.dimensionless_unscaled],
    ) -> Sequence[Sequence[NamedTuple]]:
        """
        Simultaneously solve the resonance condition and the dispersion relation
        to get root pairs of wave frequency omega and wave number k, including its
        parallel and perpendicular components, given tangent of wave normal angle
        X=tan(psi).

        Returns
        -------
        # roots :
        """
        Root = NamedTuple(
            "Root",
            [
                ("X", Quantity[u.dimensionless_unscaled]),
                ("omega", Quantity[u.rad / u.s]),
                ("k", Quantity[u.rad / u.m]),
                ("k_par", Quantity[u.rad / u.m]),
                ("k_perp", Quantity[u.rad / u.m]),
            ],
        )

        roots = []
        for X in X_range:
            psi = np.arctan(X)  # arctan of dimensionless returns radians

            # Only omega is a symbol after this
            resonant_cpdr_in_omega = self.__resonant_poly_in_omega.subs(
                {
                    "X": X.value,
                    "psi": psi.value,
                }
            )

            # Solve modified CPDR to obtain omega roots for given X
            omega_l = np.roots(resonant_cpdr_in_omega.as_poly().all_coeffs())

            # Categorise roots
            # Keep only real, positive and within bounds
            valid_omega_l = get_real_and_positive_roots(omega_l)
            valid_omega_l = [
                x
                for x in valid_omega_l
                if self.__omega_lc.value <= x <= self.__omega_uc.value
            ]

            # If valid_omega_l is empty append NaN and continue
            if len(valid_omega_l) == 0:
                root = Root(
                    X=X << u.dimensionless_unscaled,
                    omega=np.nan << u.rad / u.s,
                    k=np.nan << u.rad / u.m,
                    k_par=np.nan << u.rad / u.m,
                    k_perp=np.nan << u.rad / u.m,
                )
                roots.append([root])
                continue

            # Find values of k for each valid omega root
            # yielding some kind of nested named tuples of
            # X, omega, k, k_par, k_perp values
            # for later use in numerical integration.
            roots_tmp = []
            for valid_omega in valid_omega_l:
                k = self.solve_cpdr(valid_omega, X.value) << u.rad / u.m
                k_par = self.find_resonant_parallel_wavenumber(
                    X << u.dimensionless_unscaled,
                    valid_omega << u.rad / u.s,
                    k << u.rad / u.m,
                )
                k_perp = k * np.sin(psi)

                root = Root(
                    X=X << u.dimensionless_unscaled,
                    omega=valid_omega << u.rad / u.s,
                    k=k << u.rad / u.m,
                    k_par=k_par << u.rad / u.m,
                    k_perp=k_perp << u.rad / u.m,
                )
                roots_tmp.append(root)
            roots.append(roots_tmp)

        return roots

    def solve_cpdr(
        self,
        omega: float,
        X: float,
    ) -> float:
        """
        Solve the cold plasma dispersion relation given wave frequency
        omega and wave normal angle X=tan(psi).

        Parameters
        ----------
        omega : float
            Wave frequency.
        X_range : float
            Wave normal angles.

        Returns
        -------
        k : float (or np.nan)
        """
        # Substitute omega and X into CPDR.
        # Only k is a symbol after this.
        cpdr_in_k_omega = self.__poly_in_k.subs(
            {
                "X": X,
                "omega": omega,
            }
        )

        # Solve unmodified CPDR to obtain k roots for given X, omega
        k_l = np.roots(cpdr_in_k_omega.as_poly().all_coeffs())

        # Keep only real and positive roots
        valid_k_l = get_real_and_positive_roots(k_l)

        if valid_k_l.size == 0:
            return np.nan
        elif valid_k_l.size == 1:
            return valid_k_l[0]
        else:
            msg = "We got more than one real positive root for k"
            raise ValueError(msg)

    @u.quantity_input
    def find_resonant_parallel_wavenumber(
        self,
        X: Quantity[u.dimensionless_unscaled],
        omega: Quantity[u.rad / u.s],
        k: Quantity[u.rad / u.m],
    ) -> Quantity[u.rad / u.m]:
        """
        Given triplet X, omega and k, solution to the resonant cpdr,
        find if k_par = k * cos(psi) or k_par = k * cos(pi - psi),
        i.e. if k_par is positive or negative respectively
        (since wavenumber k is always positive and psi in
        [0, 90] degrees).

        Parameters
        ----------
        X : float
            Wave normal angle.
        omega : float
            Wave frequency.
        k : float
            Wavenumber.

        Returns
        -------
        k_par : float (or np.nan)
        """
        if np.isnan(k):
            return np.nan << u.rad / u.m

        psi = np.arctan(X)
        k_par = k * np.cos(psi)
        gyrofreq = self.plasma.gyro_freq[0]
        reson = self.resonance
        v_par = self.v_par
        gamma = self.gamma

        result1 = omega - k_par * v_par - reson * gyrofreq / gamma  # [0, pi/2]
        result2 = omega + k_par * v_par - reson * gyrofreq / gamma  # (pi/2, pi]

        if math.isclose(result1.value, 0.0, abs_tol=1e-6) and math.isclose(
            result2.value, 0.0, abs_tol=1e-6
        ):
            raise ValueError("Both are roots")
        elif not math.isclose(result1.value, 0.0, abs_tol=1e-6) and not math.isclose(
            result2.value, 0.0, abs_tol=1e-6
        ):
            raise ValueError("None of them is root")
        elif math.isclose(result1.value, 0.0, abs_tol=1e-6):
            # k_par is positive
            return k_par
        elif math.isclose(result2.value, 0.0, abs_tol=1e-6):
            # k_par is negative
            return -k_par
