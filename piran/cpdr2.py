from typing import Sequence

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity

from piran.cpdrsymbolic import CpdrSymbolic
from piran.gauss import Gaussian
from piran.helpers import calc_lorentz_factor, get_real_and_positive_roots
from piran.plasmapoint import PlasmaPoint


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
    def solve_cpdr(
        self,
        omega: Quantity[u.rad / u.s],
        X_range: Quantity[u.dimensionless_unscaled],
    ) -> Sequence[float]:
        """
        Given wave frequency omega, solve the dispersion relation for each
        wave normal angle X=tan(psi) in X_range to get wave number k.

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
    ):
        """
        Simultaneously solve the resonance condition and the dispersion relation
        to get root pairs of wave frequency omega and wave number k given
        resonance n and tangent of wave normal angle X=tan(psi).

        Returns
        -------
        # roots :
        """

        roots = []
        for X in X_range:
            psi = np.arctan(X) * u.rad

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

            # If valid_omega_l is empty continue
            if len(valid_omega_l) == 0:
                roots.append([X.value, np.nan, np.nan])
                continue

            # Find values of k for each valid omega root
            # yielding some kind of nested dict of X, omega, k values
            # for later use in numerical integration.
            roots_tmp = []
            for valid_omega in valid_omega_l:
                # Substitute omega into CPDR.
                # Only k is a symbol after this
                cpdr_in_k_omega = self.__poly_in_k.subs(
                    {
                        "X": X.value,
                        "psi": psi.value,
                        "omega": valid_omega,
                    }
                )

                # Solve unmodified CPDR to obtain k roots for given X, omega
                k_l = np.roots(cpdr_in_k_omega.as_poly().all_coeffs())

                # Keep only real and positive roots
                valid_k_l = get_real_and_positive_roots(k_l)

                if valid_k_l.size == 0:
                    roots_tmp.append((X.value, valid_omega, np.nan))
                elif valid_k_l.size == 1:
                    roots_tmp.append((X.value, valid_omega, valid_k_l[0]))
                else:
                    msg = "We got more than one real positive root for k"
                    raise ValueError(msg)

            roots.append(roots_tmp)

        return roots
