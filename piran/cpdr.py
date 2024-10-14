import math
from typing import NamedTuple, Sequence

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity

from piran.cpdrsymbolic import CpdrSymbolic
from piran.gauss import Gaussian
from piran.helpers import (
    calc_lorentz_factor,
    calc_momentum,
    get_real_and_positive_roots,
)
from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix
from piran.wavefilter import WaveFilter, WhistlerFilter

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
        wave_filter: WaveFilter = WhistlerFilter(),
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

        # Wave mode filter
        self.__wave_filter = wave_filter

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
    ) -> Sequence[Quantity[u.rad / u.m]]:
        """
        FIXME
        Given wave frequency omega, solve the dispersion relation for each
        wave normal angle X=tan(psi) in X_range to get wave number k.
        Optimised version, similar to solve_cpdr, but we lambdify in X
        after we substitute omega and is more efficient when we have
        a single value for omega and a range of X values (for example when
        computing the normalisation factor). We are also filtering directly
        for wave modes of our interest here.

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
            valid_k_l = get_real_and_positive_roots(k_l) << u.rad / u.m

            is_desired_wave_mode = [self.filter(X, omega, k) for k in valid_k_l]
            filtered_k = valid_k_l[is_desired_wave_mode]

            if filtered_k.size == 0:
                k_sol.append(np.nan << u.rad / u.m)
            elif filtered_k.size == 1:
                k_sol.append(filtered_k[0])
            else:
                raise AssertionError(
                    "In solve_cpdr_for_norm_factor we got more than 1 solutions for k"
                )

        return k_sol

    @u.quantity_input
    def solve_resonant(
        self,
        X_range: Quantity[u.dimensionless_unscaled],
    ) -> Sequence[Sequence[NamedTuple]]:
        """
        FIXME
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
            valid_omega_l = get_real_and_positive_roots(omega_l) << u.rad / u.s
            valid_omega_l = [
                x for x in valid_omega_l if self.__omega_lc <= x <= self.__omega_uc
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

            # Given X and omega solve the resonance condition
            # to find values of k for each valid omega root
            # yielding some kind of nested named tuples of
            # X, omega, k, k_par, k_perp values
            # for later use in numerical integration.
            roots_tmp = []
            for valid_omega in valid_omega_l:
                k, k_par, k_perp = self.find_resonant_wavenumber(X, valid_omega)

                is_desired_wave_mode = self.filter(X, valid_omega, k)
                if is_desired_wave_mode:
                    root = ResonantRoot(
                        X=X << u.dimensionless_unscaled,
                        omega=valid_omega,
                        k=k,
                        k_par=k_par,
                        k_perp=k_perp,
                    )
                else:
                    root = ResonantRoot(
                        X=X << u.dimensionless_unscaled,
                        omega=valid_omega,
                        k=np.nan << u.rad / u.m,
                        k_par=np.nan << u.rad / u.m,
                        k_perp=np.nan << u.rad / u.m,
                    )
                roots_tmp.append(root)

            roots.append(roots_tmp)

        return roots

    @u.quantity_input
    def solve_cpdr(
        self,
        omega: Quantity[u.rad / u.s],
        X: Quantity[u.dimensionless_unscaled],
    ) -> Quantity[u.rad / u.m]:
        """
        FIXME
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
                "X": X.value,
                "omega": omega.value,
            }
        )

        # Solve unmodified CPDR to obtain k roots for given X, omega
        k_l = np.roots(cpdr_in_k_omega.as_poly().all_coeffs())

        # Keep only real and positive roots
        valid_k_l = get_real_and_positive_roots(k_l) << u.rad / u.m

        return valid_k_l

    @u.quantity_input
    def find_resonant_wavenumber(
        self,
        X: Quantity[u.dimensionless_unscaled],
        omega: Quantity[u.rad / u.s]
    ) -> Quantity[u.rad / u.m]:
        """
        FIXME
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

        Returns
        -------
        [k, k_par, k_perp] : Quantity[u.rad / u.m]
            Astropy array containing wavenumber `k` along with the parallel
            and perpendicular components. `k_par` can be either positive
            or negative depending on the direction of wave propagation,
            while `k` and `k_perp` are non-negative as we take the absolute
            value and `psi` is in [0, 90] since X >= 0.
        """

        psi = np.arctan(X)
        gyrofreq = self.plasma.gyro_freq[0]
        reson = self.resonance
        v_par = self.v_par
        gamma = self.gamma

        k_par = (omega - (reson * gyrofreq / gamma)) / v_par
        k = np.abs(k_par / np.cos(psi))  # NOTE: we take the absolute value
        k_perp = k * np.sin(psi)

        return [k, k_par, k_perp] << u.rad / u.m

    @u.quantity_input
    def filter(
        self,
        X: Quantity[u.dimensionless_unscaled],
        omega: Quantity[u.rad / u.s],
        k: Quantity[u.rad / u.m],
    ) -> bool:
        """
        FIXME
        Parameters
        ----------
        X : Quantity[u.dimensionless_unscaled]
            Wave normal angle (0d).
        omega : Quantity[u.rad / u.s]
            Wave frequency (0d).
        k : Quantity[u.rad / u.m]
            Wavenumber (1d array).

        Returns
        -------
        k : Quantity[u.rad / u.m] (1d array)
            Wavenumbers for the selected wave mode.
        """
        return self.__wave_filter.filter(X, omega, k, self.plasma, self.stix)
