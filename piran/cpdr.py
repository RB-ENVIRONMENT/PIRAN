import sympy as sym
import astropy.constants as const

import timing


class Cpdr:
    """cold plasma dispersion relation"""

    @timing.timing
    def __init__(self, num_particles):
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
        A function for obtaining the cold plasma dispersion relation in the form
        `A*mu**4 - B*mu**2 + C`.

        That is, as a biquadratic polynomial in the wave-refractive-index `mu`.

        Note that the coefficients `A`, `B`, and `C` are themselves composed of several
        symbolic variables.

        Parameters
        ----------
        PARTICLE_SPECIES : integer
        The total number of particle species in the plasma being considered.
        e.g. for proton-electron plasma, PARTICLE_SPECIES=2.
        TESTED ONLY WITH 2 SPECIES

        Returns
        -------
        cpdr : the cold plasma dispersion relation polynomial as a
        sympy.polys.polytools.Poly object.

        cpdr_syms: a dict of `key : value` pairs, in which:
          - each `value` corresponds to a symbol used somewhere in `cpdr`, and
          - the `key` is given by the name/label for that symbol.
          A 'symbol' in this case refers to either a sympy.core.symbol.Symbol object or
          a sympy.tensor.indexed.Indexed object (for objects with indices, e.g. arrays).
          The symbols stored in `cpdr_syms` are:
            X:          ?        (?)
            Omega[i]:   (rad/s,) (Particle gyrofrequencies)
            omega_p[i]: (rad/s,) (Particle plasma frequencies)
            omega:      rad/s    (Wave frequency)
            mu:         ?        (Wave refractive index)

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
        Input:
            PARTICLE_SPECIES: defines total number of particle species in plasma
                              e.g. for proton-electron plasma, PARTICLE_SPECIES=2
                              TESTED ONLY WITH 2 SPECIES
        Returns:
            CPDR: the cold plasma dispersion relation polynomial
                  as a sympy.polys.polytools.Poly object with free symbols:
                  X:          ?        (?)
                  Omega:      (rad/s,) (Tuple of gyrofrequencies)
                  omega_p:    (rad/s,) (Tuple of plasma frequencies)
                  omega:      rad/s    (Wave resonant frequency)
                  k:          ?        (Wavenumber)
        """
        return self._poly_k, self._syms

    def as_resonant_poly_in_omega(self):
        if self._resonant_poly_omega is None:
            self._generate_resonant_poly_in_omega()

        return self._resonant_poly_omega, self._syms

    def _generate_resonant_poly_in_omega(self):
        """
        Input:
            PARTICLE_SPECIES: defines total number of particle species in plasma
                              e.g. for proton-electron plasma, PARTICLE_SPECIES=2
                              TESTED ONLY WITH 2 SPECIES
        Returns:
            CPDR: the cold plasma dispersion relation polynomial
                  as a sympy.polys.polytools.Poly object with free symbols:
                  X:          ?        (?)
                  Omega:      (rad/s,) (Tuple of gyrofrequencies)
                  n:          ?        (Cyclotron resonance)
                  omega_p:    (rad/s,) (Tuple of plasma frequencies)
                  omega:      rad/s    (Wave resonant frequency)
                  v_par:      m/s      (Parallel component of particle velocity)
                  gamma:      unitless (Lorentz factor)
                  psi:        rad      (Wave normal angle)
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
