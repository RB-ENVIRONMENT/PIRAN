import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

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


def plot_figure5(
    resonance_conditions,
    dispersion_relation,
    energy_mev,
    psi,
    alpha,
    omega_lc,
    omega_uc,
    Omega_e_abs,
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    # Plot resonance conditions
    for n in resonance_conditions.keys():
        x = [val[0] for val in resonance_conditions[n]]
        y = [val[1] for val in resonance_conditions[n]]
        plt.semilogy(x, y, linestyle="--", label=f"Resonance condition n={n}")

    # Plot dispersion relation
    disp_x = [val[0] for val in dispersion_relation]
    disp_y = [val[1] for val in dispersion_relation]
    plt.semilogy(disp_x, disp_y, "k", label="Dispersion relation")

    # Plot upper and lower
    lower_upper_x = np.arange(-1, 25, 1)
    lower_y = [omega_lc / Omega_e_abs for val in lower_upper_x]
    upper_y = [omega_uc / Omega_e_abs for val in lower_upper_x]
    plt.semilogy(lower_upper_x, lower_y, "k:")
    plt.semilogy(lower_upper_x, upper_y, "k:")

    # Convert radians to degrees and round (for title)
    psi = round(psi * 180 / math.pi)
    alpha = round(alpha * 180 / math.pi)

    plt.minorticks_on()
    plt.xticks(range(0, 21, 5))
    plt.yticks([0.1, 1.0], ["0.1", "1.0"])
    plt.yticks(np.arange(0.2, 1.0, 0.1), [], minor=True)
    plt.xlim(0.0, 20.0)
    plt.ylim(0.1, 1.0)
    plt.xlabel(r"$k \frac{c}{| \Omega_e |}$")
    plt.ylabel(r"$\frac{\omega}{| \Omega_e |}$")
    plt.legend(loc="lower right")
    plt.title(rf"$E={energy_mev}MeV, \psi={psi}^\circ, \alpha={alpha}^\circ, $")
    plt.tight_layout()
    # plt.savefig("figure5a.png", dpi=150)
    plt.show()


def main():
    # Constants
    # We can put those in a module or use the constants from astropy
    # https://docs.astropy.org/en/stable/constants/index.html
    c = 299_792_458  # m / (s), Speed of light in vacuum
    R_earth = 6_370_000  # m,       Mean Earth radius
    m_e = 9.1093837015e-31  # kg,      Electron mass (rest)
    m_p = 1.67262192369e-27  # kg,      Proton mass (rest)
    e = 1.60217663e-19  # Coulombs, Elementary charge
    q_e = -e  # Electron charge
    q_p = e  # Proton charge
    # epsilon_0 = 8.8541878128e-12  # ?? F/m, Vaccum permittivity

    # Conversion factors (multiply)
    # Again, either put those in a separate module
    # or a better solution is to use quantities with units
    # as in astropy.units
    # https://docs.astropy.org/en/stable/units/
    # keV_to_J = 1.6021766339999e-16
    MeV_to_J = 1.6021766339999e-13

    # Trying to reproduce Figure 5a from [Glauert & Horne, 2005]
    # Define input parameters
    energy_mev = 1  # Relativistic kinetic energy MeV
    RKE = energy_mev * MeV_to_J  # Relativistic kinetic energy (Joule)
    psi = math.pi * 45 / 180  # wave normal angle
    X = math.tan(psi)
    alpha = math.pi * 5 / 180  # pitch angle
    gamma = calc_lorentz_factor(RKE, m_e, c)
    v = c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha)  # Is this correct?

    # Compute and plot the resonance conditions from Figure 5
    M = 8.033454e15  # Tm^3
    lambd = 0
    L = 4.5
    frequency_ratio = 1.5
    B = (M * math.sqrt(1 + 3 * math.sin(lambd) ** 2)) / (
        L**3 * R_earth**3 * math.cos(lambd) ** 6
    )

    # Convert the following to a function with inputs
    # electric charge, mass and B
    Omega_e = (q_e * B) / m_e  # rad/s ??
    Omega_e_abs = abs(Omega_e)  # rad/s ??
    omega_pe = Omega_e_abs * frequency_ratio  # rad/s ??

    Omega_p = (q_p * B) / m_p  # rad/s ??
    Omega_p_abs = abs(Omega_p)  # rad/s ??
    omega_pp = Omega_p_abs * frequency_ratio  # rad/s ??

    # n_e = omega_pe**2 * epsilon_0 * m_e / e**2
    # n_p = omega_pp**2 * epsilon_0 * m_p / e**2

    y_min = 0.1
    y_max = 1.0
    y_list = np.linspace(y_min, y_max, num=181)

    # Dictionary to hold key:value pairs where key is the cyclotron
    # resonance n and value is a list of (x, y) tuples,
    # where x=k*c/Omega_e_abs and y=omega/Omega_e_abs
    resonance_conditions = {}
    for n in range(-5, 1):
        resonance_conditions[n] = []

        for y in y_list:
            omega = Omega_e_abs * y
            res_cond_k = (omega - (n * Omega_e_abs / gamma)) / (math.cos(psi) * v_par)
            x = res_cond_k * c / Omega_e_abs
            resonance_conditions[n].append((x, y))
            # print(f"{n=} / {x=} / {y=}")

    # Calculate the dispersion relation from Figure 5
    CPDR_k = get_cpdr_poly_k()  # in k

    dispersion_relation = []
    for y in y_list:
        omega = Omega_e_abs * y

        values_dict = {
            "c": c,
            "Omega_Base": (Omega_e, Omega_p),  # FIXME is this signed?
            "omega_p": (omega_pe, omega_pp),  # FIXME maybe omega_pp is wrong
            "omega": omega,
            "X": X,
        }
        CPDR_k2 = replace_cpdr_symbols(CPDR_k, values_dict)

        # Solve for k
        k_l = poly_solver(CPDR_k2)

        # Keep only real and positive roots
        valid_k_roots = get_valid_roots(k_l)

        # We expect at most 1 real positive root
        if len(valid_k_roots) > 1:
            print(valid_k_roots)
            print("We have more than 1 valid roots")
            quit()

        # If valid_k_roots is not empty
        if valid_k_roots:
            x = valid_k_roots[0] * c / Omega_e_abs
            dispersion_relation.append((x, y))

    # Parameters for plotting the horizontal dotted lines in Figure 5,
    # i.e. lines with constant omega/|Omega_e|
    omega_m = 0.35 * Omega_e_abs
    delta_omega = 0.15 * Omega_e_abs
    omega_lc = omega_m - 1.5 * delta_omega
    omega_uc = omega_m + 1.5 * delta_omega

    # Plot
    plot_figure5(
        resonance_conditions,
        dispersion_relation,
        energy_mev,
        psi,
        alpha,
        omega_lc,
        omega_uc,
        Omega_e_abs,
    )

    # Stop here since the following will be needed later
    quit()

    # Define the range over X (tan of wave normal angles)
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
