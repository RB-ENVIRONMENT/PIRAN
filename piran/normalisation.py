import numpy as np
import sympy as sym
from scipy.integrate import simpson, trapezoid

from piran.resonance import get_valid_roots, poly_solver, replace_cpdr_symbols


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
        Cold plasma dispersion relation object.
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
    values_dict = {
        "omega": omega.value,
        "Omega": (omega_c[0].value, omega_c[1].value),
        "omega_p": (omega_p[0].value, omega_p[1].value),
    }
    cpdr_k_eval = sym.lambdify(
        ["X"], replace_cpdr_symbols(dispersion._poly_k, values_dict), "numpy"
    )

    pairs = []
    for i, X in enumerate(X_range):
        # We've lambidifed `X` but `k` is still a symbol. When we call it with an
        # argument it substitutes `X` with the value and returns a `sympy.core.add.Add`
        # object, that's why calling `poly_solver(CPDR_k2)` still works.
        CPDR_k = cpdr_k_eval(X.value)
        k_l = poly_solver(CPDR_k)
        valid_k_l = get_valid_roots(k_l)

        if valid_k_l.size == 0:
            continue

        if valid_k_l.size > 1:
            msg = "We got more than one real positive root for k."
            raise ValueError(msg)

        pairs.append((X.value, omega.value, valid_k_l[0]))

    return pairs


def compute_glauert_normalisation_factor(
    dispersion,
    dispersion_poly_k,
    root_pairs,
    method="simpson",
):
    """
    Calculate the normalisation factor from
    Glauert & Horne 2005 (equation 15).

    Parameters
    ----------
    dispersion : piran.cpdr.Cpdr
        Cold plasma dispersion relation object.
    dispersion_poly_k : sympy.core.expr.Expr
        Cold plasma dispersion relation as polynomial in k.
    root_pairs : list of tuples (X, omega, k)
        Solutions to the cold plasma dispersion relation.
    method : str, default="simpson"

    Returns
    -------
    norm_factor : np.floating
    """
    # Derivative in omega
    dispersion_deriv_omega = dispersion_poly_k.diff("omega")

    # Derivative in k
    dispersion_deriv_k = dispersion_poly_k.diff("k")

    # It has a better performance to substitute omega here since it is the
    # same for all root pairs/triplets and then lambdify the expression outside the
    # loop, and use the lambdified object within the loop to replace X and k.
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
    wave_norm_angle_distribution = dispersion._wave_angles.eval(np.array(X_range))

    evaluated_integrand = np.empty(len(root_pairs), dtype=np.float64)
    for i, pair in enumerate(root_pairs):
        X = pair[0]
        k = pair[2]

        evaluated_integrand[i] = (
            wave_norm_angle_distribution[i]
            * k**2
            * dispersion_deriv_omega_eval(X, k)
            * X
        ) / ((1 + X**2) ** (3 / 2) * dispersion_deriv_k_eval(X, k))

    if method == "trapezoid":
        integral = trapezoid(evaluated_integrand, x=X_range)
    elif method == "simpson":
        integral = simpson(evaluated_integrand, x=X_range)
    else:
        raise ValueError(f"Wrong integration rule: {method}")

    norm_factor = integral * (1 / (2 * np.pi**2))

    return norm_factor


def compute_cunningham_normalisation_factor(
    dispersion_poly_k,
    root_pairs,
):
    """
    Calculate the normalisation factor from
    Cunningham 2023 (denominator of equation 4b).

    Parameters
    ----------
    dispersion_poly_k : sympy.core.expr.Expr
        Cold plasma dispersion relation as polynomial in k.
    root_pairs : list of tuples (X, omega, k)
        Solutions to the cold plasma dispersion relation.

    Returns
    -------
    norm_factor : numpy.ndarray[numpy.float64]
    """
    # Derivative in omega
    dispersion_deriv_omega = dispersion_poly_k.diff("omega")

    # Derivative in k
    dispersion_deriv_k = dispersion_poly_k.diff("k")

    # Since all root pairs have the same omega,
    # replace it outside the loop and lamdify.
    omega = root_pairs[0][1]
    dd_domega = sym.lambdify(
        ["X", "k"],
        replace_cpdr_symbols(dispersion_deriv_omega, {"omega": omega}),
        "numpy",
    )
    dd_dk = sym.lambdify(
        ["X", "k"], replace_cpdr_symbols(dispersion_deriv_k, {"omega": omega}), "numpy"
    )

    norm_factor = np.empty(len(root_pairs), dtype=np.float64)
    for ii, pair in enumerate(root_pairs):
        X = pair[0]
        k = pair[2]

        norm_factor[ii] = (k**2 * dd_domega(X, k) * X) / (
            (1 + X**2) ** (3 / 2) * dd_dk(X, k)
        )
    norm_factor /= 2 * np.pi**2

    return norm_factor
