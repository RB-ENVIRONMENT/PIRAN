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

    # ================ Parameters =====================

    MESH_GRANULARITY = 1001

    # Plotting selections
    PLOT_RESO_CPDR = True
    PLOT_RESO_CPDR_DOMEGA = True
    PLOT_RESO_CPDR_DX = True

    PLOT_TOTAL = PLOT_RESO_CPDR + PLOT_RESO_CPDR_DOMEGA + PLOT_RESO_CPDR_DX

    mlat_deg = Angle(0 * u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    energy = 1.0 * u.MeV
    alpha = Angle(70.8, u.deg)
    resonance = 0
    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    X_min = 0.0
    X_max = 1
    X_npoints = MESH_GRANULARITY
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )
    # =================================================

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr_sym = CpdrSymbolic(len(particles))
    cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

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

    omega_range = u.Quantity(
        np.linspace(
            cpdr.omega_lc.value,
            cpdr.omega_uc.value,
            MESH_GRANULARITY,
        ),
        u.rad / u.s,
    )

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
