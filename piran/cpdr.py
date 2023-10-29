"""
Defines the Cpdr class.
"""

import sympy as sym
import astropy.constants as const

import timing


class Cpdr:
    """
    A class for manipulation of the cold plasma dispersion relation.

    Creating an instance of this class will immediately generate a symbolic
    representation for the cpdr as a biquadratic function in the wave number `k`,
    which can take some time!

    Parameters
    ----------
    num_particles : int
        The total number of particle species in the plasma being considered.
        e.g. for proton-electron plasma, `num_particles`=2.
    """

    @timing.timing
    def __init__(self, num_particles):  # numpydoc ignore=GL08
        # TODO: Replace 'num_particles' param with a tuple of Plasmapy Particles
        # (including charge, mass, etc.)
        self._num_particles = num_particles

        # Our 'baseline' representation of the cpdr is as a biquadratic polynomial in k,
        # which we generate immediately here.
        self._poly_k, self._syms = self._generate_poly_in_k()

        # cpdr x resonant condition as a polynomial in omega, generated on request.
        self._resonant_poly_omega = None

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
        ### SYMBOL DEFINITIONS

        # Use this for indexing w.r.t. particle species
        i = sym.Idx("i", self._num_particles)
        PS_RANGE = (i, i.lower, i.upper)

        # Particle gyrofrequency
        Omega = sym.IndexedBase("Omega")
        Omega_i = Omega[i]

        # Particle plasma frequency
        omega_p = sym.IndexedBase("omega_p")
        omega_p_i = omega_p[i]

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
        _syms = {s.name: s for s in (omega, X, k, Omega_i, omega_p_i)}

        ### SYMBOLIC STIX PARAMETERS

        # Use .doit() to force expansion of the sum.
        # Doing this early seems to make later code run faster.
        #
        # Strictly necessary when trying to obtain a polynomial in omega?
        # In this case, future use of sympy.cancel and multiplication by
        # MULTIPLICATION_FACTOR should remove all traces of omega from the denominator
        # of each term.
        R = 1 - sym.Sum((omega_p_i**2) / (omega * (omega + Omega_i)), PS_RANGE).doit()
        L = 1 - sym.Sum((omega_p_i**2) / (omega * (omega - Omega_i)), PS_RANGE).doit()
        P = 1 - sym.Sum((omega_p_i**2) / (omega**2), PS_RANGE).doit()
        S = (R + L) / 2

        ### SYMBOLIC COMPONENTS OF BIQUADRATIC POLYNOMIAL IN MU

        # CPDR = A*mu**4 - B*mu**2 + C
        # Using sym.factor appears vital for *massively* reducing the time taken by
        # some operations (e.g. sym.as_poly) outside of this func.
        #
        # I've tried using other more 'concrete' funcs for this purpose
        # (e.g. sym.factor, sym.powsimp) but none of them seem to produce a result
        # that is compact as sym.simplify.
        #
        # sym.simplify *can* be very slow in other cases... but it does the job here
        # so we'll stick with it for now.
        A = sym.simplify(S * (X**2) + P)
        B = sym.simplify(R * L * (X**2) + P * S * (2 + (X**2)))
        C = sym.simplify(P * R * L * (1 + (X**2)))

        # TODO: I don't like using `.value` here - too many symbols still present.
        # Can we move the substitution of c to somewhere further along?...
        # It's a shame we can't provide units for symbols to check consistency.
        mu = const.c.value * k / omega

        # Return cpdr as a biquadratic polynomial in k
        return (A * mu**4 - B * mu**2 + C).as_poly(k), _syms

    def as_poly_in_k(self):
        """
        Retrieve baseline cpdr as a biquadratic polynomial in `k`.

        As a function in the wave refractive index `mu = c*k/omega`, the cpdr has the
        form `A*mu**4 - B*mu**2 + C` in which `A`, `B`, and `C` are themselves composed
        of several symbolic variables.

        We skip the intermediary representation in `mu` and return this as a sympy
        Polynomial in `k` for convenience.

        This baseline cpdr is generated when the `Cpdr` object is first created and is
        stored in memory.

        Returns
        -------
        symp.polys.polytools.Poly
            A biquadratic polynomial in `k`.

        dict
            A dict of `key : value` pairs, in which:
            - each `value` corresponds to a symbol used somewhere in our polynomial, and
            - the `key` is given by the name/label for that symbol.
            A 'symbol' in this case refers to either a sympy.core.symbol.Symbol object
            or a sympy.tensor.indexed.Indexed object (for objects with indices).
            The symbols stored in this dict are:
            ========== ===== ===============================
            Symbol     Units Description
            ========== ===== ===============================
            X          -     `tan(psi)` for wave angle `psi`
            Omega[i]   rad/s Particle gyrofrequencies
            omega_p[i] rad/s Particle plasma frequencies
            omega      rad/s Wave frequency
            k          -     Wave number
            ========== ===== ===============================
            .
        """
        return self._poly_k, self._syms

    def as_resonant_poly_in_omega(self):
        """
        Retrieve resonant cpdr as a polynomial in `omega`.

        As described by Glauert [1]_ in paragraph 17:

            For any given value of `X`, we substitute `k_par` from the resonance
            condition into the dispersion relation to obtain a polynomial expression
            for the frequency `omega`.

        This function returns that polynomial expression for `omega`. The polynomial is
        generated once on the first call to this func and then stored in memory for
        immediate retrieval in subsequent calls.

        The order of the polynomial is equal to 6 + (2 * `num_particles`).

        .. [1] Glauert, S. A., and R. B. Horne (2005),
        Calculation of pitch angle and energy diffusion coefficients with the PADIE
        code, J. Geophys. Res., 110, A04206, doi:10.1029/2004JA010851.

        Returns
        -------
        symp.polys.polytools.Poly
            A polynomial in `omega`.

        dict
            A dict of `key : value` pairs, in which:
            - each `value` corresponds to a symbol used somewhere in our polynomial, and
            - the `key` is given by the name/label for that symbol.
            A 'symbol' in this case refers to either a sympy.core.symbol.Symbol object
            or a sympy.tensor.indexed.Indexed object (for objects with indices).
            The symbols stored in this dict are:
            ========== ===== =======================================
            Symbol     Units Description
            ========== ===== =======================================
            X          -     `tan(psi)` for wave angle `psi`
            Omega[i]   rad/s Particle gyrofrequencies
            omega_p[i] rad/s Particle plasma frequencies
            omega      rad/s Wave frequency
            k          -     Wave number
            n          -     Cyclotron resonance
            v_par      m/s   Parallel component of particle velocity
            gamma      -     Lorentz factor
            psi        rad   Wave normal angle
            ========== ===== =======================================
            .
        """
        if self._resonant_poly_omega is None:
            self._generate_resonant_poly_in_omega()

        return self._resonant_poly_omega, self._syms

    def _generate_resonant_poly_in_omega(self):
        """
        Generate resonant cpdr as a polynomial in `omega`.

        See Also
        --------
        as_resonant_poly_in_omega : For further details.
        """
        # To retrieve cpdr as a polynomial function in omega, we need to:
        # - Rewrite mu in terms of omega only (not k) by using the resonance condition
        # - Multiply by MULTIPLICATION_FACTOR to remove all traces of omega from the
        #   denominator of any coefficients.

        # Start by grabbing our symbolic variables.
        # mu, omega, and Omega_i are already defined in cpdr_syms.
        k = self._syms["k"]
        omega = self._syms["omega"]
        Omega_i = self._syms["Omega[i]"]

        # Grab the base and index associated with our Indexed object Omega_i
        Omega = Omega_i.base
        i = Omega_i.indices[0]

        # We're also going to need some additional symbols for the timebeing.
        # TODO: Replace all of these with constant values / input params?
        v_par, psi, n, gamma = sym.symbols("v_par, psi, n, gamma")

        # Substitute k for the resonance condition, as given by Eq 5 in Lyons 1974b.
        # Note that we are also using `k = k_par * cos(psi)` here, 'losing' the sign
        # of `k_par` in the process since `k` >= 0 by defn and `cos(psi)` >= 0 for
        # `psi` in [0, 90]. If we need to determine the sign of `k_par` again at any
        # point in the future, we can do so by (re)checking the resonance condition.
        k_sub = (omega - n * Omega[0] / gamma) / (v_par * sym.cos(psi))

        # Define our MULTIPLICATION_FACTOR
        MULTIPLICATION_FACTOR = sym.Pow(omega, 6) * sym.product(
            (omega + Omega_i) * (omega - Omega_i), (i, i.lower, i.upper)
        )

        # Replace mu with mu_sub and multiply by MULTIPLICATION_FACTOR
        self._resonant_poly_omega = sym.cancel(
            (MULTIPLICATION_FACTOR * self._poly_k).subs(k, k_sub)
        ).as_poly(omega)
