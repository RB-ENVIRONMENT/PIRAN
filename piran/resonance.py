import sympy as sym


def get_cpdr_poly_omega(PARTICLE_SPECIES=2):
    """
    Input:
        PARTICLE_SPECIES: defines total number of particle species in plasma
                          e.g. for proton-electron plasma, PARTICLE_SPECIES=2
                          TESTED ONLY WITH 2 SPECIES
    Returns:
        CPDR: the cold plasma dispersion relation polynomial
              as a sympy.polys.polytools.Poly object
    """

    # Use this for indexing
    i = sym.symbols("i", cls=sym.Idx)

    # These are used for convenience
    PS_INDEX = PARTICLE_SPECIES - 1
    PS_RANGE = (i, 0, PS_INDEX)

    # Define lots of algebraic symbols
    A, B, C, X, R, L, P, S, omega = sym.symbols("A, B, C, X, R, L, P, S, omega")

    # Indexed symbols (one per particle species)
    Omega_Base = sym.IndexedBase("Omega_Base")
    Omega_i = Omega_Base[i]

    omega_p = sym.IndexedBase("omega_p")
    omega_p_i = omega_p[i]

    # Substitution of the resonance condition into the CPDR yields an expression
    # in negative powers of omega.
    # Multiply by this thing to remove omega from the denominator of all terms
    # in our expression.
    MULTIPLICATION_FACTOR = sym.Pow(omega, 6) * sym.product(
        (omega + Omega_i) * (omega - Omega_i), PS_RANGE
    )

    # Stix Parameters
    # Use .doit() to force expansion of the sum, so that multiplication by
    # MULTIPLICATION_FACTOR (to be completed shortly) properly removes all
    # traces of omega from the denominator of each term
    R = 1 - sym.Sum((omega_p_i**2) / (omega * (omega + Omega_i)), PS_RANGE).doit()
    L = 1 - sym.Sum((omega_p_i**2) / (omega * (omega - Omega_i)), PS_RANGE).doit()
    P = 1 - sym.Sum((omega_p_i**2) / (omega**2), PS_RANGE).doit()
    S = (R + L) / 2

    # CPDR = A*mu**4 - B*mu**2 + C
    # Use MULTIPLICATION_FACTOR pre-emptively here with 'simplify' to force SymPy
    # to remove omega from the denominator of each term.
    # NB. 'simplify' is a very non-targeted way of doing this; it 'works', but I'd
    # be much more comfortable if we were using something more specific!
    A = sym.simplify(MULTIPLICATION_FACTOR * (S * (X**2) + P))
    B = sym.simplify(
        MULTIPLICATION_FACTOR * (R * L * (X**2) + P * S * (2 + (X**2)))
    )
    C = sym.simplify(MULTIPLICATION_FACTOR * (P * R * L * (1 + (X**2))))

    # More symbols for mu
    # NB. mu has *another* instance of omega in the denominator, so we're going to
    # need to ask SymPy to simplify our expression again...
    c, v_par, psi, n, gamma, mu = sym.symbols("c, v_par, psi, n, gamma, mu")
    mu = (c / (v_par * sym.cos(psi))) * (1 - (n * Omega_Base[0] / (gamma * omega)))

    CPDR_A = sym.simplify(A * sym.Pow(mu, 4))
    CPDR_B = sym.simplify(-B * sym.Pow(mu, 2))
    CPDR_C = sym.simplify(C)

    # Pull everything together, request polynomial form, and return
    CPDR = sym.collect(sym.expand(CPDR_A + CPDR_B + CPDR_C), omega).as_poly(omega)

    return CPDR


def replace_cpdr_symbols(CPDR, values):
    """
    Input:
        CPDR: the cold plasma dispersion relation polynomial
              as a sympy.polys.polytools.Poly object
        values: a dict of {symbol: value}
    Returns:
        CPDR2: a sympy.polys.polytools.Poly object
               where the symbols defined in the values dict
               are replaced with their values
    """
    # Let's now replace the symbols in CPDR with actual values
    CPDR2 = CPDR.subs(values)

    # return sym.Poly(CPDR2)
    return CPDR2
