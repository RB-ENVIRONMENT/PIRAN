import numpy as np
import sympy as sym
from scipy.integrate import trapezoid, simpson

from piran import cpdr
from piran.resonance import replace_cpdr_symbols, poly_solver, get_valid_roots


def compute_root_pairs(
    omega,
    X_range,
    Omega_e,
    Omega_p,
    omega_pe,
    omega_pp,
):
    dispersion = cpdr.Cpdr(2)

    CPDR_k, _ = dispersion.as_poly_in_k()

    pairs = []
    for i, X in enumerate(X_range):
        values_dict = {
            "X": X.value,
            "omega": omega.value,
            "Omega": (Omega_e.value, Omega_p.value),
            "omega_p": (omega_pe.value, omega_pp.value),
        }
        CPDR_k2 = replace_cpdr_symbols(CPDR_k, values_dict)
        k_l = poly_solver(CPDR_k2)
        valid_k_l = get_valid_roots(k_l)

        if valid_k_l.size == 0:
            continue

        if valid_k_l.size > 1:
            msg = "We got more than one real positive root for k"
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
    method,
):
    """
    Placeholder
    """
    # Derivative in omega
    dispersion_deriv_omega = dispersion.diff("omega")

    # Derivative in k
    dispersion_deriv_k = dispersion.diff("k")

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

        dispersion_deriv_omega_eval2 = dispersion_deriv_omega_eval(X, k)
        dispersion_deriv_k_eval2 = dispersion_deriv_k_eval(X, k)

        evaluated_integrand[i] = (
            dist_wave_norm * k**2 * dispersion_deriv_omega_eval2 * X
        ) / ((1 + X**2) ** (3 / 2) * dispersion_deriv_k_eval2)

    if method == "trapezoid":
        integral = trapezoid(evaluated_integrand, x=X_range)
    elif method == "simpson":
        integral = simpson(evaluated_integrand, x=X_range)

    norm_factor = integral * (1 / (2 * np.pi**2))

    return norm_factor
