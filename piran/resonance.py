import sympy as sym
import numpy as np
import math

import timing


def get_cpdr_poly_k(PARTICLE_SPECIES=2):
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

    # Stix Parameters
    R = 1 - sym.Sum((omega_p_i**2) / (omega * (omega + Omega_i)), PS_RANGE).doit()
    L = 1 - sym.Sum((omega_p_i**2) / (omega * (omega - Omega_i)), PS_RANGE).doit()
    P = 1 - sym.Sum((omega_p_i**2) / (omega**2), PS_RANGE).doit()
    S = (R + L) / 2

    # CPDR = A*mu**4 - B*mu**2 + C
    A = sym.simplify(S * (X**2) + P)
    B = sym.simplify(R * L * (X**2) + P * S * (2 + (X**2)))
    C = sym.simplify(P * R * L * (1 + (X**2)))

    # More symbols for mu
    c, k, mu = sym.symbols("c, k, mu")
    mu = c * k / omega

    CPDR_A = sym.simplify(A * sym.Pow(mu, 4))
    CPDR_B = sym.simplify(-B * sym.Pow(mu, 2))
    CPDR_C = sym.simplify(C)

    # Pull everything together, request polynomial form, and return
    CPDR = sym.collect(sym.expand(CPDR_A + CPDR_B + CPDR_C), k).as_poly(k)

    return CPDR


@timing.timing
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


@timing.timing
def replace_cpdr_symbols(CPDR, values):
    """
    Input:
        CPDR: the cold plasma dispersion relation polynomial
              as a sympy.polys.polytools.Poly object with symbols:
                psi:        rad      (Wave normal angle)
                v_par:      m/s      (Parallel component of particle velocity)
                c:          m/s      (Speed of light in vacuum)
                gamma:      unitless (Lorentz factor)
                n:          ?        (Cyclotron resonance)
                Omega_Base: (rad/s,) (Tuple of gyrofrequencies)
                omega_p:    (rad/s,) (Tuple of plasma frequencies)
                k:          ?        (Wavenumber)
                X:          ?        (?)
                omega:      rad/s    (Wave resonant frequency)
        values: a dict of {symbol: value}
    Returns:
        CPDR2: a sympy.polys.polytools.Poly object
               where the symbols defined in the values dict
               are replaced with their values
    """
    # Let's now replace the symbols in CPDR with actual values
    CPDR2 = CPDR.subs(values)

    return CPDR2


def calc_lorentz_factor(E, m, c):
    """
    Calculate the Lorentz factor gamma given the relativistic kinetic energy,
    rest mass and speed of light.
    Relativistic kinetic energy = Total relativistic energy - Rest mass energy
    RKE = TRE - RME = (gamma - 1) * m_0 * c^2
    Inputs:
        E: Joule (Relativistic kinetic energy)
        m: kg    (Rest mass)
        c: m/s   (Speed of light in vacuum)
    Returns:
        gamma: unitless (Lorentz factor)
    """
    return (E / (m * c**2)) + 1


def get_valid_roots(values, tol=1e-8):
    """
    Filter roots based on a condition (e.g real and >tol)

    Note: check default tolerance in np.isclose()
    """
    real_part = np.real(values)
    real_part_greater_zero = np.greater(real_part, tol)

    imag_part = np.imag(values)
    imag_part_almost_zero = np.isclose(imag_part, 0.0)

    vals_where_both_true = values[
        np.logical_and(real_part_greater_zero, imag_part_almost_zero)
    ]

    # Return only the real part (the imaginary part is close to zero anyways)
    return np.real(vals_where_both_true)


def poly_solver(poly):
    # roots = sym.nroots(poly)  # returns a list of sympy Float objects
    roots = np.roots(poly.as_poly().all_coeffs())  # returns a numpy ndarray with floats

    return roots


def main():
    # Constants
    # We can put those in a module or use the constants from astropy
    # https://docs.astropy.org/en/stable/constants/index.html
    c = 299_792_458  # m / (s), Speed of light in vacuum
    # R_earth = 6_378_100  # m,       Nominal Earth equatorial radius
    m_e = 9.1093837015e-31  # kg,      Electron mass (rest)

    # Conversion factors (multiply)
    # Again, either put those in a separate module
    # or a better solution is to use quantities with units
    # as in astropy.units
    # https://docs.astropy.org/en/stable/units/
    # keV_to_J = 1.6021766339999e-16
    MeV_to_J = 1.6021766339999e-13

    # Trying to reproduce Figure 5a from [Glauert & Horne, 2005]
    # Define input parameters
    RKE = 1 * MeV_to_J  # Relativistic kinetic energy
    psi = math.pi * 45 / 180  # wave normal angle
    alpha = math.pi * 5 / 180  # pitch angle
    gamma = calc_lorentz_factor(RKE, m_e, c)
    v = c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha)  # Is this correct?

    X_min = 0
    X_max = 1
    X_range = np.linspace(X_min, X_max, 101)

    # Get the cold plasma dispersion relation as a
    # polynomial. Everything is still a symbol here.
    CPDR_omega = get_cpdr_poly_omega()  # in omega
    CPDR_k = get_cpdr_poly_k()  # in k

    # We can pass a dict of key:value pairs
    # to the sympy polynomial where
    # the key is a string with the same name
    # as the symbol we want to replace with the corresponding
    # value. For the IndexedBase we need to pass a tuple with
    # the same number of elements as the number of species.
    values_dict = {
        "psi": psi,
        "v_par": v_par,
        "c": c,
        "gamma": gamma,
        "n": 0,  # FIXME
        "Omega_Base": (1, 1),  # FIXME
        "omega_p": (1, 1),  # FIXME
    }

    # X and omega are still symbols after this
    CPDR_omega2 = replace_cpdr_symbols(CPDR_omega, values_dict)

    # X, k and omega are still symbols after this
    CPDR_k2 = replace_cpdr_symbols(CPDR_k, values_dict)

    for X in X_range:
        # Only omega is a symbol after this
        CPDR_omega3 = replace_cpdr_symbols(CPDR_omega2, {"X": X})

        # Only k and omega are symbols after this
        CPDR_k3 = replace_cpdr_symbols(CPDR_k2, {"X": X})

        # Solve modified CPDR to obtain omega roots for given X
        omega_l = poly_solver(CPDR_omega3)

        # Categorise roots
        valid_omega_l = get_valid_roots(omega_l)

        # Find values of k for each valid omega root
        # yielding some kind of nested dict of X, omega, k values
        # for later use in numerical integration.
        for valid_omega in valid_omega_l:
            # Substitute omega into CPDR
            CPDR_k4 = replace_cpdr_symbols(CPDR_k3, {"omega": valid_omega})
            print(CPDR_k4.free_symbols)
            print(CPDR_k4)

            # Solve unmodified CPDR to obtain k roots for given X, omega
            k_l = poly_solver(CPDR_k4)
            print(f"{X=}")
            print(f"{valid_omega=}")
            print(f"{k_l=}")

    # Tests for get_valid_roots()
    # test_array_1 = np.array([0.0e+00 + 0.0e+00j,
    #                          1.24900090e-16 - 1.0e+0j,
    #                          3.33066907e-16 + 1.0e+0j,
    #                          1.0e+00 + 3.10894466e-17j])
    # test_array_2 = np.array([-1 + 0j, 1.1 + 0.00000001j, 100 + 2j])
    # test_array_3 = np.array([-1, 0, 1])
    # test_array_4 = np.array([-1.0, 0.0, 0.00001, 1.0])

    # print(get_valid_roots(test_array_1))
    # print(get_valid_roots(test_array_2))
    # print(get_valid_roots(test_array_3))  # Works with integers only
    # print(get_valid_roots(test_array_4))  # Works with floats only too


if __name__ == "__main__":
    main()
