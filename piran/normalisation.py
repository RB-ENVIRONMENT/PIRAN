import numpy as np
import sympy as sym
from scipy.integrate import trapezoid, simpson

from piran.resonance import replace_cpdr_symbols, poly_solver, get_valid_roots


def solve_dispersion_relation(
    dispersion,
    omega_c,
    omega_p,
    omega,
    X_range,
):
    """
    Given wave frequency omega, solve the dispersion relation for each
    wave normal angle X=tan(psi) in X_range to get wave number k.

    Parameters
    ----------
    dispersion : piran.cpdr.Cpdr
        Cold plasma dispersion relation.
    omega_c : tuple of astropy.units.quantity.Quantity
        Cyclotron (gyro) frequencies.
    omega_p : tuple of astropy.units.quantity.Quantity
        Plasma frequencies.
    omega : astropy.units.quantity.Quantity
        Wave frequency.
    X_range : array_like of astropy.units.quantity.Quantity
        Wave normal angles.

    Returns
    -------
    root_pairs : list of tuples (X, omega, k)
        Solutions to the cold plasma dispersion relation.
    """
    pairs = []
    for i, X in enumerate(X_range):
        values_dict = {
            "X": X.value,
            "omega": omega.value,
            "Omega": (omega_c[0].value, omega_c[1].value),
            "omega_p": (omega_p[0].value, omega_p[1].value),
        }
        CPDR_k2 = replace_cpdr_symbols(dispersion._poly_k, values_dict)
        k_l = poly_solver(CPDR_k2)
        valid_k_l = get_valid_roots(k_l)

        if valid_k_l.size == 0:
            continue

        if valid_k_l.size > 1:
            msg = "We got more than one real positive root for k."
            raise ValueError(msg)

        pairs.append((X.value, omega.value, valid_k_l[0]))

    return pairs


def compute_wave_norm_angle_distribution(x, mean, sd):
    """
    For now we suppose that it is Gaussian.
    """
    dist_wave_norm = np.exp(-1.0 * ((x - mean) / sd) ** 2)

    return dist_wave_norm


def compute_glauert_normalisation_factor(
    dispersion,
    root_pairs,
    method="simpson",
):
    """
    Calculate the normalisation factor from
    Glauert & Horne 2005 (equation 15).

    Parameters
    ----------
    dispersion : piran.cpdr.Cpdr
        Cold plasma dispersion relation.
    root_pairs : list of tuples (X, omega, k)
        Solutions to the cold plasma dispersion relation.
    method : str, default="simpson"

    Returns
    -------
    norm_factor : np.floating
    """
    # Derivative in omega
    dispersion_deriv_omega = dispersion.diff("omega")

    # Derivative in k
    dispersion_deriv_k = dispersion.diff("k")

    # It has a better performance to substitute omega here since it is the
    # same for all root pairs/triplets and then lambdify the expression outside the loop,
    # and use the lambdified object within the loop to replace X and k.
    omega = root_pairs[0][1]  # All root pairs have the same omega
    values_dict = {
        "omega": omega,
    }
    dispersion_deriv_omega_eval = sym.lambdify(
        ["X", "k"], replace_cpdr_symbols(dispersion_deriv_omega, values_dict), "numpy"
    )
    dispersion_deriv_k_eval = sym.lambdify(
        ["X", "k"], replace_cpdr_symbols(dispersion_deriv_k, values_dict), "numpy"
    )

    X_range = [pair[0] for pair in root_pairs]
    evaluated_integrand = np.empty(len(root_pairs), dtype=np.float64)
    for i, pair in enumerate(root_pairs):
        X = pair[0]
        k = pair[2]

        # g(X)
        dist_wave_norm = compute_wave_norm_angle_distribution(X, 0.0, 0.577)

        evaluated_integrand[i] = (
            dist_wave_norm * k**2 * dispersion_deriv_omega_eval(X, k) * X
        ) / ((1 + X**2) ** (3 / 2) * dispersion_deriv_k_eval(X, k))

    if method == "trapezoid":
        integral = trapezoid(evaluated_integrand, x=X_range)
    elif method == "simpson":
        integral = simpson(evaluated_integrand, x=X_range)
    else:
        raise ValueError(f"Wrong integration rule: {method}")

    norm_factor = integral * (1 / (2 * np.pi**2))

    return norm_factor
