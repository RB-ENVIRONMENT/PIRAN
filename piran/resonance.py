import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

import timing


@timing.timing
def get_cpdr(PARTICLE_SPECIES=2):
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

    # Return both a polynomial in mu and a dict of the 'top-level' symbols defined
    # by this function.
    return sym.Poly.from_list([A, 0, -B, 0, C], mu), cpdr_syms


@timing.timing
def get_cpdr_poly_k(PARTICLE_SPECIES=2):
    """
    Input:
        PARTICLE_SPECIES: defines total number of particle species in plasma
                          e.g. for proton-electron plasma, PARTICLE_SPECIES=2
                          TESTED ONLY WITH 2 SPECIES
    Returns:
        CPDR: the cold plasma dispersion relation polynomial
              as a sympy.polys.polytools.Poly object with free symbols:
              X:          ?        (?)
              Omega_Base: (rad/s,) (Tuple of gyrofrequencies)
              omega_p:    (rad/s,) (Tuple of plasma frequencies)
              omega:      rad/s    (Wave resonant frequency)
              k:          ?        (Wavenumber)
    """
    ## Call common func to retrieve cpdr as a biquadratic function in mu
    cpdr, cpdr_syms = get_cpdr(PARTICLE_SPECIES)

    # To retrieve cpdr as a biquadratic function in k, we need to sub mu = c*k/omega
    # mu and omega should already be in cpdr_syms so we just grab them.
    # k is new and will need to be added to cpdr_syms.

    mu = cpdr_syms["mu"]
    omega = cpdr_syms["omega"]

    k = sym.symbols("k")
    cpdr_syms["k"] = k

    return cpdr.subs(mu, const.c.value * k / omega).as_poly(k), cpdr_syms


@timing.timing
def get_cpdr_poly_omega(PARTICLE_SPECIES=2):
    """
    Input:
        PARTICLE_SPECIES: defines total number of particle species in plasma
                          e.g. for proton-electron plasma, PARTICLE_SPECIES=2
                          TESTED ONLY WITH 2 SPECIES
    Returns:
        CPDR: the cold plasma dispersion relation polynomial
              as a sympy.polys.polytools.Poly object with free symbols:
              X:          ?        (?)
              Omega_Base: (rad/s,) (Tuple of gyrofrequencies)
              n:          ?        (Cyclotron resonance)
              omega_p:    (rad/s,) (Tuple of plasma frequencies)
              omega:      rad/s    (Wave resonant frequency)
              v_par:      m/s      (Parallel component of particle velocity)
              gamma:      unitless (Lorentz factor)
              psi:        rad      (Wave normal angle)
    """

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
    v_par, psi, n, gamma = sym.symbols("v_par, psi, n, gamma")

    # Substitute resonance condition into mu to obtain a new expression for mu,
    # stored in new symbolic variable mu_sub
    mu_sub = (const.c.value / (v_par * sym.cos(psi))) * (
        1 - (n * Omega[0] / (gamma * omega))
    )

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
                Omega:      (rad/s,) (Tuple of gyrofrequencies)
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


def calc_lorentz_factor(E, m):
    """
    Calculate the Lorentz factor gamma for a given particle species given the
    relativistic kinetic energy and rest mass.
    Relativistic kinetic energy = Total relativistic energy - Rest mass energy
    RKE = TRE - RME = (gamma - 1) * m_0 * c^2
    Inputs:
        E: Joule (Relativistic kinetic energy)
        m: kg    (Rest mass)
    Returns:
        gamma: unitless (Lorentz factor)

    Note that this is different from plasmapy's `Lorentz_factor` which provides the
    'standard' way of calculating the Lorentz factor using the relative velocity `v`.
    """
    return (E / (m * const.c**2)) + 1


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
    for n, values in resonance_conditions.items():
        x = [val[0] for val in values]
        y = [val[1] for val in values]
        plt.semilogy(x, y, linestyle="--", label=f"Resonance condition n={n}")

    # Plot dispersion relation
    disp_x = [val[0].value for val in dispersion_relation]
    disp_y = [val[1] for val in dispersion_relation]
    plt.semilogy(disp_x, disp_y, "k", label="Dispersion relation")

    # Plot upper and lower
    lower_upper_x = np.arange(-1, 25, 1)
    lower_y = [(omega_lc / Omega_e_abs).value for val in lower_upper_x]
    upper_y = [(omega_uc / Omega_e_abs).value for val in lower_upper_x]
    plt.semilogy(lower_upper_x, lower_y, "k:")
    plt.semilogy(lower_upper_x, upper_y, "k:")

    plt.minorticks_on()
    plt.xticks(range(0, 21, 5))
    plt.yticks([0.1, 1.0], ["0.1", "1.0"])
    plt.yticks(np.arange(0.2, 1.0, 0.1), [], minor=True)
    plt.xlim(0.0, 20.0)
    plt.ylim(0.1, 1.0)
    plt.xlabel(r"$k \frac{c}{| \Omega_e |}$")
    plt.ylabel(r"$\frac{\omega}{| \Omega_e |}$")
    plt.legend(loc="lower right")
    plt.title(rf"$E={energy_mev}MeV, \psi={psi.deg}^\circ, \alpha={alpha.deg}^\circ, $")
    plt.tight_layout()
    # plt.savefig("figure5a.png", dpi=150)
    plt.show()


def main():
    ### CONSTANTS

    # We use the following from astropy.constants in various places below:
    # https://docs.astropy.org/en/stable/constants/index.html
    #
    # | Name    | Value          | Unit    | Description                     |
    # | ------- | -------------- | ------- | ------------------------------- |
    # | c       | 299_792_458    | m / (s) | Speed of light in vacuum        |
    # | m_e     | 9.1093837e-31  | kg      | Electron mass                   |
    # | m_p     | 1.67262192e-27 | kg      | Proton mass                     |
    # | e       | 1.60217663e-19 | C       | Electron charge                 |
    # | R_earth | 6378100        | m       | Nominal Earth equatorial radius |

    q_e = -const.e.si  # Signed electron charge
    q_p = const.e.si  # Signed proton charge
    # epsilon_0 = 8.8541878128e-12  # ?? F/m, Vaccum permittivity

    ### INPUT PARAMETERS

    # Energy
    energy_mev = 1.0
    RKE = energy_mev * u.MeV  # Relativistic kinetic energy (Mega-electronvolts)

    # Angles
    psi = Angle(45, u.deg)  # wave normal
    X = math.tan(psi.rad)  # tan(wave normal)
    alpha = Angle(5, u.deg)  # pitch

    # Magnetic field
    M = 8.033454e15  # Tm^3
    mlat = 0
    L = 4.5
    B = (M * math.sqrt(1 + 3 * math.sin(mlat) ** 2)) / (
        L**3 * const.R_earth**3 * math.cos(mlat) ** 6
    )

    # Particle number densities
    #
    # These will be inputs used to determine plasma frequencies,
    # likely replacing the use of the frequency ratio.
    # Or perhaps we can produce code to support both input types?
    #
    # n_e = omega_pe**2 * epsilon_0 * m_e / e**2
    # n_p = omega_pp**2 * epsilon_0 * m_p / e**2

    # Gyro- and plasma-frequencies
    #
    # Convert the following to a function with inputs
    # electric charge, mass and B
    frequency_ratio = 1.5

    Omega_e = (q_e * B) / const.m_e  # rad/s ??
    Omega_e_abs = abs(Omega_e)  # rad/s ??
    omega_pe = Omega_e_abs * frequency_ratio  # rad/s ??

    Omega_p = (q_p * B) / const.m_p  # rad/s ??
    Omega_p_abs = abs(Omega_p)  # rad/s ??
    omega_pp = Omega_p_abs * frequency_ratio  # rad/s ??

    # Dimensionless frequency range
    # (scaled by 1/Omega_e)
    y_min = 0.1
    y_max = 1.0
    y_list = np.linspace(y_min, y_max, num=181)

    ### PROCEDURE

    # Trying to reproduce Figure 5a from [Glauert & Horne, 2005]

    # Calculate the Lorentz factor and particle velocity using input params
    gamma = calc_lorentz_factor(RKE, const.m_e)
    v = const.c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha.rad)  # Is this correct?

    # Dictionary to hold key:value pairs where key is the cyclotron
    # resonance n and value is a list of (x, y) tuples,
    # where x=k*c/Omega_e_abs and y=omega/Omega_e_abs
    resonance_conditions = {}
    for n in range(-5, 1):
        resonance_conditions[n] = []

        for y in y_list:
            omega = Omega_e_abs * y
            res_cond_k = (omega - (n * Omega_e_abs / gamma)) / (
                math.cos(psi.rad) * v_par
            )
            x = res_cond_k * const.c / Omega_e_abs
            resonance_conditions[n].append((x, y))
            # print(f"{n=} / {x=} / {y=}")

    # Calculate the dispersion relation from Figure 5
    CPDR_k, _ = get_cpdr_poly_k()  # in k

    dispersion_relation = []
    for y in y_list:
        omega = Omega_e_abs * y

        values_dict = {
            "Omega": (Omega_e.value, Omega_p.value),  # FIXME is this signed?
            "omega_p": (
                omega_pe.value,
                omega_pp.value,
            ),  # FIXME maybe omega_pp is wrong
            "omega": omega.value,
            "X": X,
        }
        CPDR_k2 = replace_cpdr_symbols(CPDR_k, values_dict)

        # Solve for k
        k_l = poly_solver(CPDR_k2)

        # Keep only real and positive roots
        valid_k_roots = get_valid_roots(k_l)

        # If valid_k_roots is not empty
        if valid_k_roots.size > 0:
            # We expect at most 1 real positive root
            if valid_k_roots.size > 1:
                print(valid_k_roots)
                print("We have more than 1 valid roots")
                quit()
            x = valid_k_roots[0] * const.c / Omega_e_abs
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
    CPDR_omega, _ = get_cpdr_poly_omega()  # in omega
    CPDR_k, _ = get_cpdr_poly_k()  # in k

    # We can pass a dict of key:value pairs
    # to the sympy polynomial where
    # the key is a string with the same name
    # as the symbol we want to replace with the corresponding
    # value. For the IndexedBase we need to pass a tuple with
    # the same number of elements as the number of species.
    values_dict = {
        "psi": psi.rad,
        "v_par": v_par.value,
        "gamma": gamma.value,
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
