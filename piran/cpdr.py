import functools
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
        numpy_polynomials: bool = True,
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

        # Numpy polynomials
        self.__numpy_polynomials = numpy_polynomials

        if self.__numpy_polynomials:

            # Polynomial cache
            # We store several polynomials (**in omega**) for reuse later.
            # In particular, we store polynomials corresponding to the Stix parameters and
            # the A, B, and C polynomials from the CPDR, with the latter set split into
            # constant and non-constant (i.e. dependent on X) parts.
            #
            # To obtain polynomials in `omega`, we need to remove `omega` from the
            # denominator of each term in the Stix parameters.
            # We do this by multiplying each Stix param by the lowest common multiple of
            # all the denominators used in the parameter.

            # For P this is easy: we just multiply by `omega**2`
            self.__P = np.polynomial.polynomial.Polynomial(
                [-np.sum(self.__plasma.plasma_freq.value**2), 0, 1]
            )

            # For S, R, and L we multiply by the following...
            #
            # NOTE: these correspond to the 'constant' (1) part of the parameter prior
            # to multiplication; we add on the 'summation' part of the parameter below.
            self.__S = np.polynomial.polynomial.Polynomial.fromroots(
                [*self.__plasma.gyro_freq.value, *-self.__plasma.gyro_freq.value]
            )
            self.__R = np.polynomial.polynomial.Polynomial.fromroots(
                [0, *-self.__plasma.gyro_freq.value]
            )
            self.__L = np.polynomial.polynomial.Polynomial.fromroots(
                [0, *self.__plasma.gyro_freq.value]
            )

            # ...which results in a product nested within the summation part of the
            # parameter. The indices for the summation and the product both run over all
            # particles within the plasma, *except that* the index on the product skips the
            # current index being used in the summation.

            for j in range(len(self.plasma.particles)):
                S_i = 1
                R_i = 1
                L_i = 1
                for i in range(len(self.plasma.particles)):
                    if i == j:
                        continue
                    S_i *= np.polynomial.polynomial.Polynomial.fromroots(
                        [
                            self.__plasma.gyro_freq[i].value,
                            -self.__plasma.gyro_freq[i].value,
                        ]
                    )
                    R_i *= np.polynomial.polynomial.Polynomial.fromroots(
                        [-self.__plasma.gyro_freq[i].value]
                    )
                    L_i *= np.polynomial.polynomial.Polynomial.fromroots(
                        [self.__plasma.gyro_freq[i].value]
                    )

                self.__S -= (self.__plasma.plasma_freq[j].value ** 2) * S_i
                self.__R -= (self.__plasma.plasma_freq[j].value ** 2) * R_i
                self.__L -= (self.__plasma.plasma_freq[j].value ** 2) * L_i

            # Store parts of A, B, C polynomials with coefficients that are dependent on X

            self.__Ax = np.polynomial.polynomial.Polynomial.fromroots([0, 0]) * self.__S
            self.__Bx = np.polynomial.polynomial.Polynomial.fromroots([0, 0]) * (
                (self.__R * self.__L) + (self.__P * self.__S)
            )
            self.__Cx = (
                np.polynomial.polynomial.Polynomial.fromroots([0, 0])
                * self.__P
                * self.__R
                * self.__L
            )

            # Store remaining parts of A, B, C polynomials with constant coefficients

            self.__Ac = self.__P * np.polynomial.polynomial.Polynomial.fromroots(
                [*self.__plasma.gyro_freq.value, *-self.__plasma.gyro_freq.value]
            )
            self.__Bc = (
                2
                * np.polynomial.polynomial.Polynomial.fromroots([0, 0])
                * self.__P
                * self.__S
            )
            self.__Cc = (
                np.polynomial.polynomial.Polynomial.fromroots([0, 0])
                * self.__P
                * self.__R
                * self.__L
            )

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

    @property
    def numpy_polynomials(self):
        return self.__numpy_polynomials

    @u.quantity_input
    def solve_cpdr_for_norm_factor(
        self,
        omega: Quantity[u.rad / u.s],
        X_range: Quantity[u.dimensionless_unscaled],
    ) -> Sequence[Quantity[u.rad / u.m]]:
        """
        Given wave frequency `omega`, solve the dispersion relation for each
        wave normal angle X=tan(psi) in `X_range` to get wave number `k`.
        Optimised version, similar to `solve_cpdr`, but we lambdify in `X`
        after we substitute omega and is more efficient when we have
        a single value for omega and a range of X values (for example when
        computing the normalisation factor).

        **Note:** A key difference between this function and `solve_cpdr` is
        that we filter for specific wave modes here, while `solve_cpdr` does
        not.

        Parameters
        ----------
        omega : astropy.units.quantity.Quantity convertible to rad/second
            Wave frequency.
        X_range : astropy.units.quantity.Quantity[u.dimensionless_unscaled]
            Wave normal angles.

        Returns
        -------
        k_sol : List[Quantity[u.rad / u.m]]
            The solutions are given in the same order as X_range.
            This means that each `(X, omega, k)` triplet is a solution
            to the cold plasma dispersion relation.
            If we get NaN then for this `(X, omega)` pair the CPDR has no
            roots.
        """

        if self.__numpy_polynomials:
            # This isn't an exact replacement for 'lambdification';
            # we're just passing a consistent value to an inner func without
            # 'hard-coding' the value of omega in the inner func.
            cpdr_in_X_k = functools.partial(self.numpy_poly_in_k, omega=omega.value)
        else:
            # Replace omega and lambdify
            cpdr_in_X_k = sym.lambdify(
                ["X"], self.__poly_in_k.subs({"omega": omega.value}), "numpy"
            )

        k_sol = []
        for i, X in enumerate(X_range):

            if self.__numpy_polynomials:
                cpdr_in_k = cpdr_in_X_k(X.value)
                k_l = cpdr_in_k.roots()
            else:
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

    def __get_ABC_polynomials(self, X):

        # Add X-dependent and 'constant' parts of A, B, and C polynomials together
        # to create larger polynomials (in omega).
        A = self.__Ax * (X**2) + self.__Ac
        B = self.__Bx * (X**2) + self.__Bc
        C = self.__Cx * (X**2) + self.__Cc

        return (A, B, C)

    def numpy_poly_in_omega(self, X) -> np.polynomial.polynomial.Polynomial:

        # Sub in X to return polynomials in omega for A, B, and C
        A, B, C = self.__get_ABC_polynomials(X)

        # Represent k in terms of omega
        k_res = np.polynomial.polynomial.Polynomial.fromroots(
            [self.resonance * self.plasma.gyro_freq[0].value / self.gamma]
        ) / (self.v_par.value * np.cos(np.atan(X)).value)
        ck = const.c.value * k_res

        # Bring everything together to return a single polynomial in omega
        return A * ck**4 - B * ck**2 + C

    def numpy_poly_in_k(self, X, omega) -> np.polynomial.polynomial.Polynomial:

        # Sub in X to return polynomials in omega for A, B, and C
        A, B, C = self.__get_ABC_polynomials(X)

        # Sub in omega to return a single (biquadratic) polynomial in k
        return np.polynomial.polynomial.Polynomial(
            [C(omega), 0, -B(omega) * const.c.value**2, 0, A(omega) * const.c.value**4]
        )

    @u.quantity_input
    def solve_resonant(
        self,
        X_range: Quantity[u.dimensionless_unscaled],
    ) -> Sequence[Sequence[NamedTuple]]:
        """
        Given the tangent of wave normal angle `X=tan(psi)`, simultaneously solve
        the resonance condition and dispersion relation to obtain root pairs of
        wave frequency `omega` and wave number `k`, including their parallel and
        perpendicular components.

        **Note:** We filter out solutions that do not correspond to the desired
        wave modes.

        Parameters
        ----------
        X_range : astropy.units.quantity.Quantity[u.dimensionless_unscaled]
            Tangent of wave normal angles.

        Returns
        -------
        roots : List[List[ResonantRoot]]
            Resonant roots as a list of lists of `ResonantRoot` objects.
        """

        roots = []
        for X in np.atleast_1d(X_range):

            if self.__numpy_polynomials:
                resonant_cpdr_in_omega = self.numpy_poly_in_omega(X)
                omega_l = resonant_cpdr_in_omega.roots()
            else:
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
        Given wave frequency `omega` and tangent of the wave normal angle
        `X=tan(psi)`, solves the cold plasma dispersion relation for the
        wavenumber `k`.

        Parameters
        ----------
        omega : Quantity[u.rad / u.s]
            Scalar astropy Quantity representing the wave frequency in
            units convertible to radians per second.
        X : Quantity[u.dimensionless_unscaled]
            Scalar astropy Quantity representing the tangent of the wave
            normal angle in units convertible to dimensionless unscaled.

        Returns
        -------
        k : Quantity[u.rad / u.m]
            1d astropy Quantity representing the real and positive wavenumbers
            in radians per meter.
        """
        if self.__numpy_polynomials:
            cpdr_in_k = self.numpy_poly_in_k(X.value, omega.value)
            k_l = cpdr_in_k.roots()
        else:
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
        self, X: Quantity[u.dimensionless_unscaled], omega: Quantity[u.rad / u.s]
    ) -> Quantity[u.rad / u.m]:
        """
        Substitute the resonant `omega` into the resonance condition to obtain
        `k_par`. Then, using the resonant `X = tan(psi)`, we can calculate `k`
        and `k_perp`. Because `psi` is in the range [0, 90] degrees, `k` and
        `k_perp` are always positive (we ensure this by taking the absolute value).
        However, the resonant cpdr returns solutions in the range [0, 180] degrees,
        which means that `k_par` can be negative.

        Parameters
        ----------
        X : Quantity[u.dimensionless_unscaled]
            Tangent of wave normal angle in units convertible to dimensionless
            unscaled.
        omega : Quantity[u.rad / u.s]
            Wave frequency in units convertible to radians per second.

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
        This method calls the `filter` method of the `WaveFilter` class.
        Please refer to the `WaveFilter.filter` documentation for details about
        the parameters and their corresponding meanings.
        """
        return self.__wave_filter.filter(X, omega, k, self.plasma, self.stix)
