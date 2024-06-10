import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


def main():
    """
    We plot three things in this script, each relevant to singularity-finding:

    1. Resonant CPDR (i.e. the resonance condition substituted into the
       dispersion relation, yielding a function in X and omega).
    2. The derivative of the Resonant CPDR w.r.t. omega
    3. The derivative of the Resonant CPDR w.r.r. X

    Singularities occur when both (1) and (2) are 0. This is accompanied by a change in
    the number of real roots of the Resonant CPDR as X varies, UNLESS (3) is also 0 (we
    think). It may not be possible for the latter condition to occur in practice, but we
    haven't done the analysis to prove this one way or another.

    With this in mind, we have three+ possible routes to locating singularities:

    a. Solve (1) for some subset of X in [X_min, X_max] and look for (unexpected)
       changes in the number of real roots. We can already define regions in
       [X_min, X_max] where we expect the number of real roots to be otherwise fixed.
       Having found a rough location where the roots change, perform binary search to
       isolate the singularity.

    b. Produce a minimiser for minimising the distance between any two roots of (1).
       If this is ever zero, we've found a root! Note that this *could* include complex
       roots which might produce negative distance and aid our minimiser?
       This would likely need to be a pairwise operation on every pair of roots within
       a given range of X, which would potentially be computationally intensive.

    c. Solve (2) for some subset of X. Plug (X, omega) values back into (1) and look for
       a change of sign in (1) (indicating an intersection of (1) and (2)).
       This will potentially require 'following' multiple branches of solutions to (2).
       This will not necessarily hold if (3) is also 0 when (1) and (2) are 0!

    d. (Attempted & failed) minimise `abs(1) + C * abs(2)` for some appropriate scale
        factor C. I tried this and didn't get very far: choosing C is a bit of a fudge
        and we're unlikely to ever precisely hit the singularity because of this.
        Out-of-the-box scipy optimisers offered mixed results, and none of them liked
        the scale of the numbers here (e.g. 10^20 is about as close as I could get to 0)
    """
    # ================ Parameters =====================

    # Plotting selections
    PLOT_RESO_CPDR = True
    PLOT_RESO_CPDR_DOMEGA = True
    PLOT_RESO_CPDR_DX = True

    PLOT_TOTAL = PLOT_RESO_CPDR + PLOT_RESO_CPDR_DOMEGA + PLOT_RESO_CPDR_DX

    # Plasma parameters
    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 * u.MeV
    alpha = Angle(70.8, u.deg)
    resonance = 0
    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    # Wave angle discretisation
    MESH_GRANULARITY = 1001

    X_min = 0.0
    X_max = 1
    X_range = u.Quantity(
        np.linspace(X_min, X_max, MESH_GRANULARITY), unit=u.dimensionless_unscaled
    )

    # =================================================

    # CPDR object + constituents
    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr_sym = CpdrSymbolic(len(particles))
    cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

    ## Wave frequency discretisation (must come after creation of cpdr)
    omega_range = u.Quantity(
        np.linspace(
            cpdr.omega_lc.value,
            cpdr.omega_uc.value,
            MESH_GRANULARITY,
        ),
        u.rad / u.s,
    )

    X_sym = cpdr.symbolic.syms.get("X")
    psi_sym = cpdr.symbolic.syms.get("psi")
    omega_sym = cpdr.symbolic.syms.get("omega")

    reso_cpdr = cpdr.resonant_poly_in_omega.subs(X_sym, sym.tan(psi_sym))
    reso_cpdr_domega = sym.diff(reso_cpdr, "omega")
    reso_cpdr_dx = sym.diff(reso_cpdr, "psi")

    reso_cpdr_lambda = sym.lambdify([(psi_sym, omega_sym)], sym.Abs(reso_cpdr))
    reso_cpdr_domega_lambda = sym.lambdify(
        [(psi_sym, omega_sym)], sym.Abs(reso_cpdr_domega)
    )
    reso_cpdr_dx_lambda = sym.lambdify([(psi_sym, omega_sym)], sym.Abs(reso_cpdr_dx))

    xgrid, ygrid = np.meshgrid(X_range.value, omega_range.value)
    xy = np.stack([xgrid, ygrid])

    idx = 0
    fig, axs = plt.subplots(1, PLOT_TOTAL)

    # axs not subscriptable if PLOT_TOTAL is 1... grrr...
    if PLOT_TOTAL == 1:
        if PLOT_RESO_CPDR:
            axs.imshow(
                np.log(reso_cpdr_lambda(xy)),
                interpolation="bilinear",
                origin="lower",
                cmap="gray",
            )
            axs.set_title("Resonant CPDR")
        elif PLOT_RESO_CPDR_DOMEGA:
            axs.imshow(
                np.log(reso_cpdr_domega_lambda(xy)),
                interpolation="bilinear",
                origin="lower",
                cmap="gray",
            )
            axs.set_title("d(Resonant CPDR)/d$\omega$")
        elif PLOT_RESO_CPDR_DX:
            axs.imshow(
                np.log(reso_cpdr_dx_lambda(xy)),
                interpolation="bilinear",
                origin="lower",
                cmap="gray",
            )
            axs.set_title("d(Resonant CPDR)/d$X$")

        axs.set_xlabel("$X$")
        axs.set_ylabel("$\omega$")
        axs.set_xticks(
            range(0, MESH_GRANULARITY, int(MESH_GRANULARITY / 10)),
            [f"{X:.2f}" for X in X_range[:: int(MESH_GRANULARITY / 10)]],
        )
        axs.set_yticks(
            range(0, MESH_GRANULARITY, int(MESH_GRANULARITY / 10)),
            [f"{w:.2E}" for w in omega_range[:: int(MESH_GRANULARITY / 10)].value],
        )

    else:
        if PLOT_RESO_CPDR:
            axs[idx].imshow(
                np.log(reso_cpdr_lambda(xy)),
                interpolation="bilinear",
                origin="lower",
                cmap="gray",
            )
            axs[idx].set_title("Resonant CPDR")
            idx = idx + 1
        if PLOT_RESO_CPDR_DOMEGA:
            axs[idx].imshow(
                np.log(reso_cpdr_domega_lambda(xy)),
                interpolation="bilinear",
                origin="lower",
                cmap="gray",
            )
            axs[idx].set_title("d(Resonant CPDR)/d$\omega$")
            idx = idx + 1
        if PLOT_RESO_CPDR_DX:
            axs[idx].imshow(
                np.log(reso_cpdr_dx_lambda(xy)),
                interpolation="bilinear",
                origin="lower",
                cmap="gray",
            )
            axs[idx].set_title("d(Resonant CPDR)/d$X$")
            idx = idx + 1

        for ax in axs:
            ax.set_xlabel("$X$")
            ax.set_ylabel("$\omega$")
            ax.set_xticks(
                range(0, MESH_GRANULARITY, int(MESH_GRANULARITY / 10)),
                [f"{X:.2f}" for X in X_range[:: int(MESH_GRANULARITY / 10)]],
            )
            ax.set_yticks(
                range(0, MESH_GRANULARITY, int(MESH_GRANULARITY / 10)),
                [f"{w:.2E}" for w in omega_range[:: int(MESH_GRANULARITY / 10)].value],
            )

    plt.show()


if __name__ == "__main__":
    main()
