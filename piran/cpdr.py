import functools
from typing import NamedTuple, Sequence

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from numpy.polynomial.polynomial import Polynomial

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
        plasma: PlasmaPoint,
        energy: Quantity[u.Joule] | None = None,
        pitch_angle: Quantity[u.rad] | None = None,
        resonance: int | None = None,
        freq_cutoff_params: Sequence[float] | None = None,
        wave_filter: WaveFilter = WhistlerFilter(),
    ) -> None:
        self.__plasma = plasma

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

        # Polynomial cache
        #
        # The CPDR in it's 'standard' form is a multivariate polynomial, given by
        #
        # .. math::
        #    A(X, w) \mu^4 - B(X, w) \mu^2 + C = 0
        #
        # where 'w' is shorthand for 'omega', mu = ck/w, and
        #
        # .. math::
        #    A(X, w) = SX^2 + P
        #    B(X, w) = RLX^2 + PS(2 + X^2)
        #    C(X, w) = PRL(1 + X^2)
        #
        # In particular, it is:
        #
        # - Quadratic in X
        # - Biquadratic in k
        # - Not a polynomial in omega
        #
        # Glauert & Horne 2005 states that by substituting the parallel component of k
        # from the resonance condition into the CPDR, we can obtain a polynomial
        # expression for the frequency omega.
        #
        # In practice, we have found that this requires some additional work. The CPDR
        # is an equation that includes various occurrences of omega in the denominators
        # of its many terms (originating from the Stix parameters and mu). Since the
        # equation is 'CPDR = 0', we are able to 'multiply out' these omega from the
        # denominators to obtain an equation that *is* a polynomial in omega.
        #
        # The multiplication factor we use is:
        #
        # .. math::
        #    w^6 \prod_s (w + w_{c,s})(w - w_{c,s})
        #
        # where 'w' is shorthand for 'omega', 'w_c' is gyrofrequency, and the subscript
        # 's' refers to a particular particle.
        #
        # This factor has been found by inspection of the CPDR and the Stix terms it is
        # comprised of. Multiplying by this results in a polynomial in omega with order
        # 6 + 2N (for a plasma comprised of N particle species), consistent with Glauert
        # and Horne's description.
        #
        # We build the CPDR piece-by-piece from smaller polynomials in omega, stored as
        # (private) class properties for convenient reuse. The fundamental building
        # blocks are the Stix parameters, which again require multiplication by
        # expressions involving omega in order to be given in polynomial form. We do not
        # multiply anything by the whole 'multiplication factor' above; this applies to
        # the _entire_ CPDR, of which the Stix parameters are just one contributing
        # part. Once we begin combining Stix parameters to form the complete CPDR, we
        # will consider what further expressions in omega we need to multiply each part
        # by to ensure that each term has ultimately been multiplied by the same
        # multiplication factor given above.
        #
        # The expression for P is given by:
        #
        # .. math::
        #    1 - \sum_s \frac{w_{p,s}^2}{w^2}
        #
        # which we multiply by :math:`w^2`.
        #
        # The expression for S is given by:
        #
        # .. math::
        #    1 - \sum_s \frac{w_{p,s}^2}{(w + w_{c,s})(w - w_{c,s})}
        #
        # which we multiply by :math:`\prod_s (w + w_{c,s})(w - w_{c,s})`.
        #
        # The expression for R is given by:
        #
        # .. math::
        #    1 - \sum_s \frac{w_{p,s}^2}{w(w + w_{c,s})}
        #
        # which we multiply by :math:`w\prod_s (w + w_{c,s})`
        #
        # The expression for L is given by:
        #
        # .. math::
        #    1 - \sum_s \frac{w_{p,s}^2}{w(w - w_{c,s})}
        #
        # which we multiply by :math:`w\prod_s (w - w_{c,s})`

        # Following multiplication, P becomes a simple quadratic in `w`:

        self.__P = Polynomial([-np.sum(self.__plasma.plasma_freq.value**2), 0, 1])

        # For S, R, and L, the process is more complicated. Following multiplication:
        #
        # - the previously constant '1' term becomes a simple polynomial in `w`,
        # - the summation becomes a summation over a product, where the product skips
        #   the current index of summation (since this is the part that has been
        #   'multiplied out').
        #
        # For example, our new expression for S looks like:
        #
        # .. math::
        #    \prod_s (w + w_{c,s})(w - w_{c,s}) - \sum_j w_{p,j}^2 * \prod_{i \ne j} (w + w_{c,i})(w - w_{c,j})
        #
        # with R and L following similarly. Note that the indices s, i, and j here all
        # run over all particles within the plasma.
        #
        # In all cases (S, R, L), we build these expressions up piece-by-piece by
        # combining smaller polynomials. Starting with the first product term for each,
        # we use Polynomial.fromroots to build a Polynomial:

        self.__S = Polynomial.fromroots(
            [*self.__plasma.gyro_freq.value, *-self.__plasma.gyro_freq.value]
        )
        self.__R = Polynomial.fromroots([0, *-self.__plasma.gyro_freq.value])
        self.__L = Polynomial.fromroots([0, *self.__plasma.gyro_freq.value])

        # To handle the remaining part, the summation over the product, we use a
        # double-for-loop and add to the existing term. The double-for-loop
        # helps with skipping over an index in the product (the inner loop),
        # although similar may be achieved with clever use of ndarrays.

        for j in range(len(self.plasma.particles)):
            S_i = 1
            R_i = 1
            L_i = 1
            for i in range(len(self.plasma.particles)):
                if i == j:
                    continue
                S_i *= Polynomial.fromroots(
                    [
                        self.__plasma.gyro_freq[i].value,
                        -self.__plasma.gyro_freq[i].value,
                    ]
                )
                R_i *= Polynomial.fromroots([-self.__plasma.gyro_freq[i].value])
                L_i *= Polynomial.fromroots([self.__plasma.gyro_freq[i].value])

            self.__S -= (self.__plasma.plasma_freq[j].value ** 2) * S_i
            self.__R -= (self.__plasma.plasma_freq[j].value ** 2) * R_i
            self.__L -= (self.__plasma.plasma_freq[j].value ** 2) * L_i

        # Now returning to the expressions A(X, w), B(X, w), and C(X, w), we perform two
        # modifications:
        #
        # - we 'absorb' the 1\w terms originating from mu into A, B ,C, and
        # - we split A(X, w) into A_x(w) * X^2 + A_c(w) (and similar for B and C).
        #
        # The intention here is to group polynomials in omega and clearly demarcate
        # parts that are constant-in-omega (A_c(w)) and X-dependent (A_x(w) * X^2)).
        #
        # This is also the point at which we need to ensure we have multiplied by a
        # consistent multiplication factor across all the terms.
        #
        # For A_x, B_x, and C_x, this involves multiplying by an additional w^6, w^4,
        # and w^2 respectively. When additionally accounting for the 1/w terms from mu,
        # this actually results in multiplying each by a consistent w^2 (which we
        # represent using Polynomial.fromroots([0,0])).

        self.__Ax = Polynomial.fromroots([0, 0]) * self.__S
        self.__Bx = Polynomial.fromroots([0, 0]) * (
            (self.__R * self.__L) + (self.__P * self.__S)
        )
        self.__Cx = Polynomial.fromroots([0, 0]) * self.__P * self.__R * self.__L

        # B_c, and C_c follow similarly. A_c is the exception, in which the 'missing'
        # component is the product involving gyrofrequencies.

        self.__Ac = self.__P * Polynomial.fromroots(
            [*self.__plasma.gyro_freq.value, *-self.__plasma.gyro_freq.value]
        )
        self.__Bc = 2 * Polynomial.fromroots([0, 0]) * self.__P * self.__S
        self.__Cc = Polynomial.fromroots([0, 0]) * self.__P * self.__R * self.__L

    @property
    def plasma(self):
        return self.__plasma

    @property
    def stix(self):
        return self.__stix

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
    ) -> Quantity[u.rad / u.m]:
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
        k_sol : Quantity[u.rad / u.m]
            The solutions are given in the same order as X_range.
            This means that each `(X, omega, k)` triplet is a solution
            to the cold plasma dispersion relation.
            If we get NaN then for this `(X, omega)` pair the CPDR has no
            roots.
        """

        # Replace omega to retrieve a function in X only.
        # omega is a fixed quantity at this point, whereas we still have a range in X.
        roots_in_k_with_fixed_omega = functools.partial(
            self.__roots_in_k, omega=omega.value
        )

        k_sol = u.Quantity(np.zeros(X_range.size), u.rad / u.m)
        for i, X in enumerate(X_range):

            k_l = roots_in_k_with_fixed_omega(X.value)
            valid_k_l = get_real_and_positive_roots(k_l) << u.rad / u.m

            is_desired_wave_mode = [self.filter(X, omega, k) for k in valid_k_l]
            filtered_k = valid_k_l[is_desired_wave_mode]

            if filtered_k.size == 0:
                k_sol[i] = np.nan << u.rad / u.m
            elif filtered_k.size == 1:
                k_sol[i] = filtered_k[0]
            else:
                raise AssertionError(
                    "In solve_cpdr_for_norm_factor we got more than 1 solutions for k"
                )

        return k_sol

    def __get_ABC_polynomials(
        self, X: float
    ) -> tuple[Polynomial, Polynomial, Polynomial]:
        """
        Given the tangent of wave normal angle `X=tan(psi)`, return polynomials
        (in omega) for the A, B, and C parts of the resonant CPDR.

        Note in particular that these polynomials have:

        1. already been multiplied by the "resonant CPDR multiplication factor":

        .. math::
           w^6 \\prod_s (w + w_{c,s})(w - w_{c,s})

        where 'w' is shorthand for 'omega', 'w_c' is gyrofrequency, and the
        subscript 's' refers to a particular particle.

        2. have absorbed the `1/w` terms from the wave refractive index mu.

        Parameters
        ----------
        X : float
            Tangent of wave normal angle.

        Returns
        -------
        A tuple of NumPy polynomials corresponding to the A, B, and C terms in
        the CPDR. These are polynomials in 'omega' with 'X' fixed according to
        the input parameter provided to this method.
        """

        # Add X-dependent and 'constant' parts of A, B, and C polynomials together
        # to create larger polynomials (in omega).
        A = self.__Ax * (X**2) + self.__Ac
        B = self.__Bx * (X**2) + self.__Bc
        C = self.__Cx * (X**2) + self.__Cc

        return (A, B, C)

    def __resonant_roots_in_omega(self, X: float) -> np.ndarray:
        """
        Given the tangent of wave normal angle `X=tan(psi)`, simultaneously solve
        the resonance condition and dispersion relation to obtain roots in `omega`.

        Parameters
        ----------
        X : float
            Tangent of wave normal angle.

        Returns
        -------
        np.ndarray
            Array containing the roots (in omega) of the resonant CPDR.
        """

        # Sub in X to return polynomials in omega for A, B, and C
        A, B, C = self.__get_ABC_polynomials(X)

        # Represent k in terms of omega
        k_res = Polynomial.fromroots(
            [self.resonance * self.plasma.gyro_freq[0].value / self.gamma]
        ) / (self.v_par.value * np.cos(np.arctan(X)))
        ck = const.c.value * k_res

        # Bring everything together to solve a single polynomial in omega
        return (A * ck**4 - B * ck**2 + C).roots()

    def __roots_in_k(self, X: float, omega: float) -> np.ndarray:
        """
        Given the tangent of wave normal angle `X=tan(psi)` and frequency `omega`,
        solve the dispersion relation to obtain roots in `omega`.

        Note that we use the same common base (provided by __get_ABC_polynomials)
        as is used in __resonant_roots_in_omega. i.e. the equation we actually
        solve here is the CPDR multiplied by the "resonant CPDR multiplication
        factor". Mathematically, for a given (X, omega) this should yield the
        same results for roots in 'k'.

        Parameters
        ----------
        X : float
            Tangent of wave normal angle.
        omega : float
            Wave frequency.

        Returns
        -------
        np.ndarray
            Array containing the roots (in 'k') of the CPDR.
        """

        # Sub in X to return polynomials in omega for A, B, and C
        A, B, C = self.__get_ABC_polynomials(X)

        # Sub in omega and solve a single (biquadratic) polynomial in k
        return Polynomial(
            [C(omega), 0, -B(omega) * const.c.value**2, 0, A(omega) * const.c.value**4]
        ).roots()

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

            # Solve resonant CPDR to obtain omega roots for given X
            omega_l = self.__resonant_roots_in_omega(X.value)

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
        # Solve CPDR to obtain k roots for given (X, omega)
        k_l = self.__roots_in_k(X.value, omega.value)

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
