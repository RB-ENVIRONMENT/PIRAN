import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import timing


@timing.timing
def get_cpdr(PARTICLE_SPECIES=2):
    ### SYMBOL DEFINITIONS

    # Use this for indexing w.r.t. particle species
    i = sym.Idx("i", PARTICLE_SPECIES)
    PS_RANGE = (i, i.lower, i.upper)

    # Particle gyrofrequency
    Omega = sym.IndexedBase("Omega")
    Omega_i = Omega[i]

    # Particle plasma frequency
    omega_p = sym.IndexedBase("omega_p")
    omega_p_i = omega_p[i]

    # These are our remaining 'top-level' symbols
    omega, X, mu = sym.symbols("omega, X, mu")

    # A quick note on indexed objects in sympy.
    # https://docs.sympy.org/latest/modules/tensor/indexed.html
    #
    # sympy offers three classes for working with an indexed object:
    #
    # - Idx         : an integer index
    # - IndexedBase : the base or stem of an indexed object
    # - Indexed     : a mathematical object with indices
    #
    # Perhaps counterintuitively, it is the combination of an Idx and and IndexedBase
    # that are used to create the 'larger' Indexed object.
    #
    # The `.base` and `.indices` methods can be used to access the IndexedBase and Idx
    # associated with an Indexed object.
    #
    # When propagating symbols outside of this function, we will keep track of the
    # Indexed object only since this contains the 'totality' of the information.

    # Add all newly-defined symbols to a dictionary to be returned by this function.
    cpdr_syms = {s.name: s for s in (omega, X, mu, Omega_i, omega_p_i)}

    print(cpdr_syms)

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
    A = sym.factor(S * (X**2) + P)
    B = sym.factor(R * L * (X**2) + P * S * (2 + (X**2)))
    C = sym.factor(P * R * L * (1 + (X**2)))

    # Return both a polynomial in mu and a dict of the 'top-level' symbols defined
    # by this function.
    return sym.Poly.from_list([A, 0, -B, 0, C], mu), cpdr_syms


@timing.timing
def get_cpdr_poly_k(PARTICLE_SPECIES=2):
    ## Call common func to retrieve cpdr as a biquadratic function in mu
    cpdr, cpdr_syms = get_cpdr(PARTICLE_SPECIES)

    # To retrieve cpdr as a biquadratic function in k, we need to sub mu = c*k/omega
    # mu and omega should already be in cpdr_syms so we just grab them.
    # k is new and will need to be added to cpdr_syms.
    # TODO: Replace symbolic c with constant c from astropy.

    mu = cpdr_syms["mu"]
    omega = cpdr_syms["omega"]

    k = sym.symbols("k")
    cpdr_syms["k"] = k

    c = sym.symbols("c")

    return cpdr.subs(mu, c * k / omega).as_poly(k), cpdr_syms


@timing.timing
def get_cpdr_poly_omega(PARTICLE_SPECIES=2):
    # Call common func to retrieve cpdr as a biquadratic function in mu
    cpdr, cpdr_syms = get_cpdr(PARTICLE_SPECIES)

    # To retrieve cpdr as a polynomial function in omega, we need to:
    # - Rewrite mu in terms of omega only (not k) by using the resonance condition
    # - Multiply by MULTIPLICATION_FACTOR to remove all traces of omega from the
    #   denominator of any coefficients.

    # Start by grabbing our symbolic variables.
    # mu, omega, and Omega_i are already defined in cpdr_syms.
    mu = cpdr_syms["mu"]
    omega = cpdr_syms["omega"]
    Omega_i = cpdr_syms["Omega[i]"]

    # Grab the base and index associated with our Indexed object Omega_i
    Omega = Omega_i.base
    i = Omega_i.indices[0]

    # We're also going to need some additional symbols for the timebeing.
    # TODO: Replace all of these with constant values / input params?
    c, v_par, psi, n, gamma = sym.symbols("c, v_par, psi, n, gamma")

    # Substitute resonance condition into mu to obtain a new expression for mu,
    # stored in new symbolic variable mu_sub
    mu_sub = (c / (v_par * sym.cos(psi))) * (1 - (n * Omega[0] / (gamma * omega)))

    # Define our MULTIPLICATION_FACTOR
    MULTIPLICATION_FACTOR = sym.Pow(omega, 6) * sym.product(
        (omega + Omega_i) * (omega - Omega_i), (i, i.lower, i.upper)
    )

    # Replace mu with mu_sub and multiply by MULTIPLICATION_FACTOR
    return (
        sym.cancel((MULTIPLICATION_FACTOR * cpdr).subs(mu, mu_sub)).as_poly(omega),
        cpdr_syms,
    )


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
                Omega: (rad/s,) (Tuple of gyrofrequencies)
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


def plot_resonance_conditions(resonance_conditions, energy_mev, psi, alpha):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 14,
        }
    )

    for n in resonance_conditions.keys():
        x = [val[0] for val in resonance_conditions[n]]
        y = [val[1] for val in resonance_conditions[n]]
        plt.semilogy(x, y, linestyle="--", label=f"{n=}")

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
    plt.legend()
    plt.title(rf"$E={energy_mev}MeV, \psi={psi}^\circ, \alpha={alpha}^\circ, $")
    plt.show()


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
    energy_mev = 1  # Relativistic kinetic energy MeV
    RKE = energy_mev * MeV_to_J  # Relativistic kinetic energy (Joule)
    psi = math.pi * 45 / 180  # wave normal angle
    alpha = math.pi * 5 / 180  # pitch angle
    gamma = calc_lorentz_factor(RKE, m_e, c)
    v = c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha)  # Is this correct?

    # Compute and plot the resonance conditions from Figure 5
    frequency_ratio = 1.5
    omega_pe = 2.902e4 * 2 * math.pi  # in rad/s
    Omega_e = omega_pe / frequency_ratio

    y_min = 0.1
    y_max = 1.0
    y_list = np.linspace(y_min, y_max, num=181)

    # Dictionary to hold key:value pairs where key is the cyclotron
    # resonance n and value is a list of (x, y) tuples,
    # where x=k*c/Omega_e and y=omega/Omega_e
    resonance_conditions = {}
    for n in range(-5, 1):
        resonance_conditions[n] = []

        for y in y_list:
            omega = Omega_e * y
            res_cond_k = (omega - (n * Omega_e / gamma)) / (math.cos(psi) * v_par)
            x = res_cond_k * c / Omega_e
            resonance_conditions[n].append((x, y))
            # print(f"{n=} / {x=} / {y=}")

    plot_resonance_conditions(resonance_conditions, energy_mev, psi, alpha)
    quit()

    # Define the range over X (tan of wave normal angles)
    X_min = 0
    X_max = 1
    X_range = np.linspace(X_min, X_max, 101)

    # Get the cold plasma dispersion relation as a
    # polynomial. Everything is still a symbol here.
    CPDR_omega, _ = get_cpdr_poly_omega()  # in omega
    CPDR_k, _ = get_cpdr_poly_k()  # in k

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
        "Omega": (1, 1),  # FIXME
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
