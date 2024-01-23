"""
Defines the Cpdr class.
"""

from typing import Sequence

import astropy.constants as const
import astropy.units as u
import numpy as np
import sympy as sym
from astropy.coordinates import Angle

from piran.gauss import Gaussian
from piran.magfield import MagField
from piran.particles import Particles
from piran.stix import Stix


class Cpdr:
    """
    A class for manipulating the cold plasma dispersion relation.

    Parameters
    ----------
    particles : ParticleListLike
        A sequence of physical particles comprising our plasma.
        See: [ParticleListLike](https://docs.plasmapy.org/en/stable/api/plasmapy.particles.particle_collections.ParticleListLike.html#particlelistlike).
    wave_angles : Gaussian
        Wave normal angles.
    wave_freqs : Gaussian
        Wave frequencies.
    mag_field : MagField
        Magnetic field.
    resonances : Sequence[int]
        Resonances.
    """

    def __init__(
        self,
        particles: Particles,
        wave_angles: Gaussian,
        wave_freqs: Gaussian,
        mag_field: MagField,
        mlat: Angle,
        l_shell: float,
        resonances: Sequence[int],
    ) -> None:  # numpydoc ignore=GL08
        self._particles = particles
        self._wave_angles = wave_angles
        self._wave_freqs = wave_freqs
        self._mag_field = mag_field
        self._mlat = mlat
        self._l_shell = l_shell
        self._resonances = resonances

        # Dict of symbols used throughout these funcs
        self._syms = self._generate_syms()

        # cpdr as a biquadratic polynomial in k, generated on request.
        self._poly_k = None

        # cpdr x resonant condition as a polynomial in omega, generated on request.
        self._resonant_poly_omega = None

        # gyrofrequency = charge * mag field / mass
        self._w_c = u.Quantity(
            [
                particle.charge
                * self._mag_field.get_strength(self._mlat, self._l_shell)
                / particle.mass
                for particle in self._particles.all
            ],
            1 / u.s,
        )

        # plasma frequency = sqrt( (number density * charge^2) /
        #                          (vacuum permittivity * mass) )
        self._w_p = u.Quantity(
            [
                np.sqrt(
                    (particle.density * particle.charge**2)
                    / (const.eps0 * particle.mass)
                )
                for particle in self._particles.all
            ],
            1 / u.s,
        )

        self.stix = Stix(self._w_p, self._w_c)

    def _generate_syms(self):
        """
        Generate the symbols used in other class methods.

        Returns
        -------
        dict
            A dict of `key : value` pairs containing common symbols that can be
            retrieved and used by other class methods.
        """
        # Use this for indexing w.r.t. particle species
        i = sym.Idx("i", len(self._particles.all))

        # Particle gyrofrequency
        Omega = sym.IndexedBase("Omega")
        Omega_i = Omega[i]

        # Particle plasma frequency
        omega_p = sym.IndexedBase("omega_p")
        omega_p_i = omega_p[i]

        # We're also going to need some additional symbols for the timebeing.
        # TODO: Replace all of these with constant values / input params?
        v_par, psi, n, gamma = sym.symbols("v_par, psi, n, gamma")

        # These are our remaining 'top-level' symbols
        omega, X, k = sym.symbols("omega, X, k")

        # A quick note on indexed objects in sympy.
        # https://docs.sympy.org/latest/modules/tensor/indexed.html
        #
        # sympy offers three classes for working with an indexed object:
        #
        # - Idx         : an integer index
        # - IndexedBase : the base or stem of an indexed object
        # - Indexed     : a mathematical object with indices
        #
        # Perhaps counterintuitively, it is the combination of an Idx and an IndexedBase
        # that are used to create the 'larger' Indexed object.
        #
        # The `.base` and `.indices` methods provide access to the IndexedBase and Idx
        # associated with an Indexed object.
        #
        # When propagating symbols outside of this function, we will keep track of the
        # Indexed object only since this contains the 'totality' of the information.

        # Add all newly-defined symbols to a dictionary to be returned by this function.
        return {
            s.name: s for s in (omega, X, k, Omega_i, omega_p_i, v_par, psi, n, gamma)
        }

    def _generate_poly_in_k(self):
        """
        Generate baseline cpdr as a biquadratic polynomial in k.

        Returns
        -------
        symp.polys.polytools.Poly
            A biquadratic polynomial in `k`.

        dict
            A dict of `key : value` pairs containing symbol information for the
            polynomial.

        See Also
        --------
        as_poly_in_k : For further details.
        """
        ### SYMBOL RETRIEVAL

        # CPDR variables
        omega = self._syms["omega"]
        X = self._syms["X"]
        k = self._syms["k"]

        # Gyrofrequencies
        Omega_i = self._syms["Omega[i]"]

        # Plasma frequencies
        omega_p_i = self._syms["omega_p[i]"]

        # Grab the index associated with our Indexed object Omega_i
        # NOTE: This is the same as is used by omega_p_i
        idx = Omega_i.indices[0]

        # Define this for convenient use with Sympy products / summations
        PS_RANGE = (idx, idx.lower, idx.upper)

        ### SYMBOLIC STIX PARAMETERS

        # Use sym.summation (rather than sym.Sum) to force expansion of the sum.
        # We could delay this until later (beyond the scope of this function, even)
        # using a combination of sym.Sum and .doit() but would need to be careful.
        # e.g. if self._poly_k contains unevaluated sums, calling .doit() on each coeff
        # individually works fine, but self._poly_k.as_expr().doit() can get stuck.
        # Why? Dunno...
        R = 1 - sym.summation((omega_p_i**2) / (omega * (omega + Omega_i)), PS_RANGE)
        L = 1 - sym.summation((omega_p_i**2) / (omega * (omega - Omega_i)), PS_RANGE)
        P = 1 - sym.summation((omega_p_i**2) / (omega**2), PS_RANGE)
        S = (R + L) / 2

        ### SYMBOLIC COMPONENTS OF BIQUADRATIC POLYNOMIAL IN k

        # CPDR = A*mu**4 - B*mu**2 + C where mu = c*k/omega
        #
        # We previously used sym.simplify here which made later calls to sym.as_poly
        # much more efficient. We are no longer doing this because:
        # - sym.simplify can be a little slow.
        # - sym.as_poly can be *really* slow in any case (e.g. never returning?)
        #
        # We also tried using other more 'concrete' funcs (e.g. factor, powsimp)
        # instead of simplify, but none seemed to produce a result as compact as
        # simplify (which, as per the source code, does a lot more than just calling
        # other public Sympy funcs in sequence).
        #
        # So, rather than using sym.simplify and sym.as_poly we now:
        # - Leave things unsimplified where possible
        # - Rely on sym.Poly.from_list, which is a bit more restrictive but much
        #   more reliable than sym.as_poly.
        #
        # TODO: I don't like using `.value` here - too many symbols still present.
        # Can we move the substitution of c to somewhere further along?...
        # It's a shame we can't provide units for symbols to check consistency.
        A = S * (X**2) + P
        B = R * L * (X**2) + P * S * (2 + (X**2))
        C = P * R * L * (1 + (X**2))

        self._A = A
        self._B = B
        self._C = C
        self._R = R
        self._L = L
        self._P = P
        self._S = S
        self._D = (R - L) / 2

        A = A * ((const.c.value / omega) ** 4)
        B = B * ((const.c.value / omega) ** 2)

        # Return cpdr as a biquadratic polynomial in k.
        return sym.Poly.from_list([A, 0, -B, 0, C], k)

    def as_poly_in_k(self):  # numpydoc ignore=RT05
        """
        Retrieve baseline cpdr as a biquadratic polynomial in ``k``.

        As a function in the wave refractive index ``mu = c*k/omega``, the cpdr has the
        form ``A*mu**4 - B*mu**2 + C`` in which ``A``, ``B``, and ``C`` are themselves
        composed of several symbolic variables.

        We skip the intermediary representation in ``mu`` and return this as a sympy
        Polynomial in ``k`` for convenience.

        This baseline cpdr is generated when the ``Cpdr`` object is first created and is
        stored in memory.

        Returns
        -------
        symp.polys.polytools.Poly
            A biquadratic polynomial in ``k``.

        dict
            A dict of ``key : value`` pairs, in which:

            - each ``value`` corresponds to a symbol used somewhere in our polynomial,
            - the ``key`` is given by the name/label for that symbol.

            A 'symbol' in this case refers to either a ``sympy.core.symbol.Symbol``
            object or a ``sympy.tensor.indexed.Indexed`` object
            (for objects with indices).

            The symbols stored in this dict are:

            ============== ===== ===================================
            Symbol         Units Description
            ============== ===== ===================================
            ``X``          None  ``tan(psi)`` for wave angle ``psi``
            ``Omega[i]``   rad/s Particle gyrofrequencies
            ``omega_p[i]`` rad/s Particle plasma frequencies
            ``omega``      rad/s Wave frequency
            ``k``          1/m   Wave number
            ============== ===== ===================================
        """
        if self._poly_k is None:
            self._poly_k = self._generate_poly_in_k()

        return self._poly_k, self._syms

    def as_resonant_poly_in_omega(self):  # numpydoc ignore=RT05
        """
        Retrieve resonant cpdr as a polynomial in ``omega``.

        As described by Glauert [1]_ in paragraph 17:

            For any given value of ``X``, we substitute ``k_par`` from the resonance
            condition into the dispersion relation to obtain a polynomial expression
            for the frequency ``omega``.

        This function returns that polynomial expression for ``omega``. The polynomial
        is generated once on the first call to this func and then stored in memory for
        immediate retrieval in subsequent calls.

        The order of the polynomial is equal to ``6 + (2 * num_particles)``.

        .. [1] Glauert, S. A., and R. B. Horne (2005),
           Calculation of pitch angle and energy diffusion coefficients with the PADIE
           code, J. Geophys. Res., 110, A04206, `doi:10.1029/2004JA010851
           <https://doi.org/10.1029/2004JA010851>`_.

        Returns
        -------
        symp.polys.polytools.Poly
            A polynomial in ``omega``.

        dict
            A dict of ``key : value`` pairs, in which:

            - each ``value`` corresponds to a symbol used somewhere in our polynomial,
            - the ``key`` is given by the name/label for that symbol.

            A 'symbol' in this case refers to either a ``sympy.core.symbol.Symbol``
            object or a ``sympy.tensor.indexed.Indexed`` object
            (for objects with indices).

            The symbols stored in this dict are:

            ============== ===== =======================================
            Symbol         Units Description
            ============== ===== =======================================
            ``X``          None  ``tan(psi)`` for wave angle ``psi``
            ``Omega[i]``   rad/s Particle gyrofrequencies
            ``omega_p[i]`` rad/s Particle plasma frequencies
            ``omega``      rad/s Wave frequency
            ``k``          1/m   Wave number
            ``n``          None  Cyclotron resonance
            ``v_par``      m/s   Parallel component of particle velocity
            ``gamma``      None  Lorentz factor
            ``psi``        rad   Wave normal angle
            ============== ===== =======================================
        """
        if self._resonant_poly_omega is None:
            self._resonant_poly_omega = self._generate_resonant_poly_in_omega()

        return self._resonant_poly_omega, self._syms

    def _generate_resonant_poly_in_omega(self):
        """
        Generate resonant cpdr as a polynomial in ``omega``.

        Returns
        -------
        symp.polys.polytools.Poly
            A polynomial in ``omega``.

        See Also
        --------
        as_resonant_poly_in_omega : For further details.
        """
        # To retrieve cpdr as a polynomial function in omega, we need to:
        # - Rewrite mu in terms of omega only (not k) by using the resonance condition
        # - Multiply by MULTIPLICATION_FACTOR to remove all traces of omega from the
        #   denominator of any coefficients.

        ### SYMBOL RETRIEVAL

        # CPDR variables (minus k, which gets subbed out)
        omega = self._syms["omega"]
        X = self._syms["X"]

        # Gyrofrequencies
        Omega_i = self._syms["Omega[i]"]
        Omega = Omega_i.base

        # Plasma frequencies
        omega_p_i = self._syms["omega_p[i]"]
        omega_p = omega_p_i.base

        # Grab the index associated with our Indexed object Omega_i
        # NOTE: This is the same as is used by omega_p_i
        idx = Omega_i.indices[0]

        # We're also going to need the following for the resonance condition
        n = self._syms["n"]
        v_par = self._syms["v_par"]
        psi = self._syms["psi"]
        gamma = self._syms["gamma"]

        ### PROCEDURE

        # Define this for convenient use with Sympy products / summations
        PS_RANGE = (idx, idx.lower, idx.upper)

        # To obtain a polynomial in `omega`, we need to remove `omega` from the
        # denominator of each term in the Stix parameters.
        # We do this by multiplying each Stix param by the lowest common multiple of
        # all the denominators used in the parameter.

        # For P this is easy: we just multiply by `omega**2`
        P = omega**2 - sym.summation(omega_p_i**2, PS_RANGE)

        # For S, R, and L we multiply by the following...
        #
        # NOTE: these correspond to the 'constant' (1) part of the parameter prior
        # to multiplication; we add on the 'summation' part of the parameter below.
        S = sym.product((omega + Omega_i) * (omega - Omega_i), PS_RANGE)
        R = omega * sym.product(omega + Omega_i, PS_RANGE)
        L = omega * sym.product(omega - Omega_i, PS_RANGE)

        # ...which results in a product nested within the summation part of the
        # parameter. The indices for the summation and the product both run over all
        # particles within the plasma, *except that* the index on the product skips the
        # current index being used in the summation. Sympy isn't designed to handle this
        # kind of complexity, so we need to implement this ourselves.
        #
        # NOTE: Sympy expressions are immutable so updates to S, S_i etc. in the below
        # do not actually modify variables 'in-place'. Hopefully this isn't costly.
        for j in range(idx.lower, idx.upper + 1):
            S_i = 1
            R_i = 1
            L_i = 1
            for i in range(idx.lower, idx.upper + 1):
                if i == j:
                    continue
                S_i *= (omega + Omega[i]) * (omega - Omega[i])
                R_i *= omega + Omega[i]
                L_i *= omega - Omega[i]
            S -= (omega_p[j] ** 2) * S_i
            R -= (omega_p[j] ** 2) * R_i
            L -= (omega_p[j] ** 2) * L_i

        # Ultimately, all terms in the cpdr need to be multiplied by a common factor.
        # That common factor is the lowest common multiple of all of the above,
        # multiplied further by `omega**4` to account for the `omega` in the denominator
        # of `mu`. That is:
        #
        # (omega ** 6) * sym.Product(
        #     (omega + Omega_i) * (omega - Omega_i), (idx, idx.lower, idx.upper)
        # )
        #
        # Each term in A, B, C needs to be adjusted to account for this, giving...

        A = (omega**2) * S * (X**2) + sym.product(
            (omega + Omega_i) * (omega - Omega_i), PS_RANGE
        ) * P
        B = (omega**2) * (R * L * (X**2) + P * S * (2 + (X**2)))
        C = (omega**2) * (P * R * L * (1 + (X**2)))

        # Substitute k for the resonance condition, as given by Eq 5 in Lyons 1974b.
        # Note that we are also using `k = k_par * cos(psi)` here, 'losing' the sign
        # of `k_par` in the process since `k` >= 0 by defn and `cos(psi)` >= 0 for
        # `psi` in [0, 90]. If we need to determine the sign of `k_par` again at any
        # point in the future, we can do so by (re)checking the resonance condition.
        k_res = (omega - n * Omega[0] / gamma) / (v_par * sym.cos(psi))
        ck = const.c.value * k_res

        # Return a polynomial in omega.
        # NOTE: This uses .as_poly which can be painfully slow for large expressions.
        # The hope is that the procedure above will result in an expression that is
        # almost already in polynomial form, allowing this Sympy func to perform well.

        return (A * ck**4 - B * ck**2 + C).as_poly(omega)
