import math
from typing import NamedTuple, Sequence

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from scipy.optimize import root_scalar

from piran.cpdrsymbolic import CpdrSymbolic
from piran.gauss import Gaussian
from piran.helpers import (
    calc_lorentz_factor,
    calc_momentum,
    get_real_and_positive_roots,
)
from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix

ResonantRoot = NamedTuple(
    "ResonantRoot",
    [
        ("X", Quantity[u.dimensionless_unscaled]),
        ("omega", Quantity[u.rad / u.s]),
        ("k", Quantity[u.rad / u.m]),
        ("k_par", Quantity[u.rad / u.m]),
        ("k_perp", Quantity[u.rad / u.m]),
    ],
)


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

            self.__momentum = calc_momentum(
                self.__gamma, self.__plasma.particles[0].mass
            )
            self.__p_par = self.__momentum * np.cos(self.__pitch_angle)
            self.__p_perp = self.__momentum * np.sin(self.__pitch_angle)

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
    def momentum(self):
        return self.__momentum

    @property
    def p_par(self):
        return self.__p_par

    @property
    def p_perp(self):
        return self.__p_perp

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

        Parameters
        ----------
        X_range : astropy.units.quantity.Quantity[u.dimensionless_unscaled]
            Wave normal angles.

        Returns
        -------
        Resonant roots as a list of lists of ResonantRoot objects.
        """

        roots = []
        for X in np.atleast_1d(X_range):
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
                root = ResonantRoot(
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

                root = ResonantRoot(
                    X=X << u.dimensionless_unscaled,
                    omega=valid_omega << u.rad / u.s,
                    k=k << u.rad / u.m,
                    k_par=k_par << u.rad / u.m,
                    k_perp=k_perp << u.rad / u.m,
                )
                roots_tmp.append(root)
            roots.append(roots_tmp)

        return roots

    @u.quantity_input
    def solve_resonant_for_x(
        self,
        omega: u.Quantity[u.rad / u.s],
        X_range: u.Quantity[u.dimensionless_unscaled],
        verbose: bool = False,
    ) -> u.Quantity[u.dimensionless_unscaled]:
        """
        Given Cpdr object and a 0d/1d array of omega, solve resonant cpdr for each omega.
        Typical usage: let omega be lower / upper frequency cutoffs so that this will return
        the values of X at which new solutions to the resonant cpdr enter / exit the region
        of interest bounded by [omega_lc, omega_uc].

        Parameters
        ----------
        omega: u.Quantity[u.rad / u.s]
            A 0d/1d array of values in omega, for which we would like to solve the resonant
            Cpdr to find corresponding solutions in X.
        X_range: u.Quantity[u.rad / u.s]
            An initial discretisation in X. For each omega, we produce values for the
            resonant cpdr for all X in X_range and look for changes in sign (indicating the
            presence of a root). A root finding algorithm then determines the precise
            location of the root.
        verbose: bool
            Controls print statements.

        Returns
        -------
        u.Quantity[u.dimensionless_unscaled]
            A (flat) list of solutions in X.
        """

        roots = []

        for om in np.atleast_1d(omega):

            X = self.__symbolic.syms.get("X")
            psi = self.__symbolic.syms.get("psi")

            # Only psi is a symbol after this
            resonant_cpdr_in_psi = self.__resonant_poly_in_omega.subs(
                {X: sym.tan(psi), "omega": om.value}
            )

            # lambdify our func in psi
            resonant_cpdr_in_psi_lambdified = sym.lambdify(psi, resonant_cpdr_in_psi)

            # transform range in X to range in psi
            psi_range = np.arctan(X_range)

            # evaluate func for all psi and store sign of result
            cpdr_signs = np.sign(resonant_cpdr_in_psi_lambdified(psi_range))

            # We want to perform a pairwise comparison of consecutive elements and
            # look for a change of sign (from 1 to -1 or vice versa).
            # We can do this efficiently by adding an ndarray containing the first
            # element of each pair to an ndarray containing the second element of
            # each pair.
            # Anywhere that the result is 0 indicates a change in sign!
            pairwise_sign_sums = cpdr_signs[:-1] + cpdr_signs[1:]

            # Find indices corresponding to changes of sign.
            # This is faster than looping over the whole pairwise_sign_sums
            # for large arrays.
            sign_change_indices = np.flatnonzero(pairwise_sign_sums == 0)

            # For each index where we have identified that a change of sign occurs,
            # use scipy's root_scalar to hone in on the root.
            for idx in sign_change_indices:
                root_result = root_scalar(
                    resonant_cpdr_in_psi_lambdified,
                    bracket=[psi_range[idx].value, psi_range[idx + 1].value],
                    method="brentq",
                )
                roots.append(root_result.root)

                if verbose:
                    print(
                        f"For {om=}\n"
                        f"Change of sign between psi = {psi_range[idx].to_value(u.deg)}, {psi_range[idx+1].to_value(u.deg)}\n"
                        f"Indices = {idx}, {idx+1}\n"
                        f"Root at: {root_result.root * 180 / np.pi}\n"
                    )

        # Convert back to X and return
        return u.Quantity(np.tan(roots), u.dimensionless_unscaled)

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
        return k_par = k * cos(psi) or k_par = k * cos(pi - psi)
        according to the resonance condition,
        i.e. the _signed_ value of k_par.
        (We have to check this manually since the wavenumber k is
        always positive and psi is in [0, 90] degrees but the
        resonant cpdr returns solutions for psi in [0, 180],
        i.e. including _negative_ k_par, and we are interested in all
        of these solutions!)

        Parameters
        ----------
        X : Quantity[u.dimensionless_unscaled]
            Wave normal angle.
        omega : Quantity[u.rad / u.s]
            Wave frequency.
        k : Quantity[u.rad / u.m]
            Wavenumber.

        Returns
        -------
        k_par : Quantity[u.rad / u.m]
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

        k_par_is_pos = math.isclose(result1.value, 0.0, abs_tol=1e-6)
        k_par_is_neg = math.isclose(result2.value, 0.0, abs_tol=1e-6)

        if k_par_is_pos and not k_par_is_neg:
            # only positive k_par is root
            return k_par
        elif k_par_is_neg and not k_par_is_pos:
            # only negative k_par is root
            return -k_par
        elif k_par_is_pos and k_par_is_neg:
            raise ValueError("Both are roots")
        else:
            raise ValueError("None of them is root")
