import math
import json

import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran import timing
from piran import cpdr


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


@timing.timing
def compute_root_pairs(
    dispersion,
    n_range,
    X_range,
    v_par,
    gamma,
    Omega_e,
    Omega_p,
    omega_pe,
    omega_pp,
    omega_lc,
    omega_uc,
):
    """
    Simultaneously solve the resonance condition and the dispersion relation
    to get root pairs of wave frequency omega and wave number k for each
    resonance n and tangent of wave normal angle X=tan(psi).

    Returns
    -------
    root_pairs : Dictionary where keys are resonances n and
      values are lists that contains tuples of (X, omega, k)
    """

    # Get the cold plasma dispersion relation as a
    # polynomial. Everything is still a symbol here.
    CPDR_omega, _ = dispersion.as_resonant_poly_in_omega()
    CPDR_k, _ = dispersion.as_poly_in_k()

    # We can pass a dict of key:value pairs
    # to the sympy expression where
    # the key is a string with the same name
    # as the symbol we want to replace with the corresponding
    # value. For the IndexedBase we need to pass a tuple with
    # the same number of elements as the number of species.
    values_dict = {
        "v_par": v_par.value,
        "gamma": gamma.value,
        "Omega": (Omega_e.value, Omega_p.value),
        "omega_p": (omega_pe.value, omega_pp.value),
    }

    # X, psi, omega and n are still symbols after this
    CPDR_omega2 = replace_cpdr_symbols(CPDR_omega, values_dict)

    # X, k and omega are still symbols after this
    CPDR_k2 = replace_cpdr_symbols(CPDR_k, values_dict)

    root_pairs = {}
    for n in n_range:
        root_pairs[int(n.value)] = []  # initialize list

        for X in X_range:
            psi = math.atan(X) * u.rad

            values_dict2 = {
                "X": X.value,
                "psi": psi.value,
                "n": n.value,
            }

            # Only omega is a symbol after this
            CPDR_omega3 = replace_cpdr_symbols(CPDR_omega2, values_dict2)

            # Only k and omega are symbols after this
            CPDR_k3 = replace_cpdr_symbols(CPDR_k2, values_dict2)

            # Solve modified CPDR to obtain omega roots for given X
            omega_l = poly_solver(CPDR_omega3)

            # Categorise roots
            # Keep only real, positive and within bounds
            valid_omega_l = get_valid_roots(omega_l)
            valid_omega_l = [
                x for x in valid_omega_l if omega_lc.value <= x <= omega_uc.value
            ]

            # If valid_omega_l is empty continue
            if len(valid_omega_l) == 0:
                continue

            # We expect at most 1 real positive root
            if len(valid_omega_l) > 1:
                msg = "We got more than one real positive root for omega"
                raise ValueError(msg)

            # Find values of k for each valid omega root
            # yielding some kind of nested dict of X, omega, k values
            # for later use in numerical integration.
            # Note: At this point valid_omega_l will contain only one element
            for valid_omega in valid_omega_l:
                # Substitute omega into CPDR
                CPDR_k4 = replace_cpdr_symbols(CPDR_k3, {"omega": valid_omega})

                # Solve unmodified CPDR to obtain k roots for given X, omega
                k_l = poly_solver(CPDR_k4)

                # Keep only real and positive roots
                valid_k_l = get_valid_roots(k_l)

                # If valid_k_l is empty continue
                if valid_k_l.size == 0:
                    continue

                # We expect at most 1 real positive root
                if valid_k_l.size > 1:
                    msg = "We got more than one real positive root for k"
                    raise ValueError(msg)

                # Note: At this point valid_k_l will contain only one element
                valid_k = valid_k_l[0]

                # Store a tuple into the dictionary
                root_pairs[int(n.value)].append((X.value, valid_omega, valid_k))

    return root_pairs


def plot_figure5(
    resonance_conditions,
    dispersion_relation,
    root_pairs,
    RKE,
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
        x = [val[0].value for val in values]
        y = [val[1].value for val in values]
        plt.semilogy(x, y, linestyle="--", label=f"Resonance condition n={n}")

    # Plot dispersion relation
    disp_x = [val[0].value for val in dispersion_relation]
    disp_y = [val[1].value for val in dispersion_relation]
    plt.semilogy(disp_x, disp_y, "k", label="Dispersion relation")

    for values in root_pairs.values():
        if values is []:
            continue
        x = [val[2] * (const.c.value / Omega_e_abs.value) for val in values]
        y = [val[1] / Omega_e_abs.value for val in values]
        plt.semilogy(x, y, "ro")

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
    plt.title(rf"$E={RKE}, \psi={psi.deg}^\circ, \alpha={alpha.deg}^\circ, $")
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
    # | eps0    | 8.85418781e-12 | F/m     | Vacuum electric permittivity    |

    q_e = -const.e.si  # Signed electron charge
    q_p = const.e.si  # Signed proton charge

    ### INPUT PARAMETERS

    # Energy
    RKE = 1.0 * u.MeV  # Relativistic kinetic energy (Mega-electronvolts)

    # Angles
    psi = Angle(45, u.deg)  # wave normal
    X = math.tan(psi.rad)  # tan(wave normal)
    alpha = Angle(5, u.deg)  # pitch

    # Magnetic field
    M = 8.033454e15 * (u.tesla * u.m**3)
    mlat = Angle(0, u.deg)
    l_shell = 4.5 * u.dimensionless_unscaled  # Could absorb const.R_earth into this?
    B = (M * math.sqrt(1 + 3 * math.sin(mlat.rad) ** 2)) / (
        l_shell**3 * const.R_earth**3 * math.cos(mlat.rad) ** 6
    )

    # ELectron gyro- and plasma- frequencies

    # Glauert provides the following frequency ratio for
    # electron 'plasma-:gyro-' frequency
    frequency_ratio = 1.5 * u.dimensionless_unscaled

    # Gyrofrequency can be calculated directly using electron charge, mass, and
    # the magnetic field.
    Omega_e = (q_e * B) / const.m_e

    # Application of the frequency ratio yields the electron plasma frequency.
    Omega_e_abs = abs(Omega_e)
    omega_pe = Omega_e_abs * frequency_ratio

    # Proton gyro- and plasma- frequencies
    # (including particle number densities)

    # We assume that the number density of electrons and protons are equal.
    # These can be derived from the electron plasma frequency.
    n_ = omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2

    Omega_p = (q_p * B) / const.m_p
    omega_pp = np.sqrt((n_ * q_p**2) / (const.eps0 * const.m_p))

    # Calculate the Lorentz factor and particle velocity using input params
    gamma = calc_lorentz_factor(RKE, const.m_e)
    v = const.c * math.sqrt(1 - (1 / gamma**2))  # relative velocity
    v_par = v * math.cos(alpha.rad)  # Is this correct?

    # Lower and upper cut-off frequencies
    omega_m = 0.35 * Omega_e_abs
    delta_omega = 0.15 * Omega_e_abs
    omega_lc = omega_m - 1.5 * delta_omega
    omega_uc = omega_m + 1.5 * delta_omega

    # Resonances
    n_min = -5
    n_max = 5
    n_range = u.Quantity(
        range(n_min, n_max + 1), unit=u.dimensionless_unscaled, dtype=np.int32
    )

    # Tangent of wave normal angles psi (X = tan(psi))
    #   X_min = 0.0
    #   X_max = 1.0
    #   X_npoints = 11
    #   X_range = u.Quantity(np.linspace(X_min, X_max, X_npoints))  # FIXME Unit?
    X_range = [1.0] * u.dimensionless_unscaled

    dispersion = cpdr.Cpdr(2)

    # For each resonance n and tangent of wave normal angle psi,
    # solve simultaneously the dispersion relation and the
    # resonance condition to get valid root pairs for omega and k.
    root_pairs = compute_root_pairs(
        dispersion,
        n_range,
        X_range,
        v_par,
        gamma,
        Omega_e,
        Omega_p,
        omega_pe,
        omega_pp,
        omega_lc,
        omega_uc,
    )

    print(json.dumps(root_pairs, indent=4))

    ### PROCEDURE
    # Trying to reproduce Figure 5a from [Glauert & Horne, 2005]

    # Dimensionless frequency range
    # To be scaled up by Omega_e_abs when used.
    # We could just use u.Unit(Omega_e_abs) directly here, but:
    # - This isn't an SI unit
    # - It mostly overcomplicates things (particularly debugging) in my experience.
    y_min = 0.1 * u.dimensionless_unscaled
    y_max = 1.0 * u.dimensionless_unscaled
    y_list = np.linspace(y_min, y_max, num=181)

    # Dictionary to hold key:value pairs where key is the cyclotron
    # resonance n and value is a list of (x, y) tuples,
    # where x=k*c/Omega_e_abs and y=omega/Omega_e_abs
    resonance_conditions = {}
    for n in range(n_min, n_max + 1):
        resonance_conditions[n] = []

        for y in y_list:
            omega = Omega_e_abs * y
            res_cond_k = (omega - (n * Omega_e / gamma)) / (math.cos(psi.rad) * v_par)
            x = res_cond_k * const.c / Omega_e_abs
            resonance_conditions[n].append((x, y))
            # print(f"{n=} / {x=} / {y=}")

    # Calculate the dispersion relation from Figure 5
    CPDR_k, _ = dispersion.as_poly_in_k()

    dispersion_relation = []
    for y in y_list:
        omega = Omega_e_abs * y

        values_dict = {
            "Omega": (Omega_e.value, Omega_p.value),  # FIXME is this signed?
            "omega_p": (omega_pe.value, omega_pp.value),
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

    # Plot
    plot_figure5(
        resonance_conditions,
        dispersion_relation,
        root_pairs,
        RKE,
        psi,
        alpha,
        omega_lc,
        omega_uc,
        Omega_e_abs,
    )

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
