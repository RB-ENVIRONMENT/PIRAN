# python src/scripts/landau_singularities.py --rke 1.0 \
#                                            --alpha 5.0 \
#                                            --resonance 0
#                                           [--save]
# where rke in MeV and alpha in degrees.
import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr, ResonantRoot
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


def plot(
    x,
    y,
    rke,
    alpha,
    resonance,
    save,
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    plt.plot(x, y, "b.", markersize=2)
    plt.axhline(y=0.0, color="k", linestyle="--", alpha=0.4)
    plt.yscale("symlog")
    plt.xlim(0.0, 1.0)
    plt.ylim(-1e8, 3e8)
    plt.xlabel(r"X")
    plt.ylabel(r"$v_{||} - \partial \omega / \partial k_{||}$")
    plt.title(rf"$E$={rke} MeV, $\alpha$={alpha}$^{{\circ}}, $n$={resonance}$")
    plt.tight_layout()

    filestem = f"resonance({resonance})E({rke})alpha({float(alpha):05.2f})"
    if save:
        plt.savefig(f"{filestem}.png", dpi=150)
    else:
        plt.show()


def compute_landau_term(cpdr: Cpdr, resonant_roots: Sequence[Sequence[ResonantRoot]]):
    x_axis = []
    y_axis = []
    for row in resonant_roots:
        for root in row:
            X = root.X
            omega = root.omega
            k = root.k

            dD_dk = cpdr.stix.dD_dk(omega, X, k)
            dD_dw = cpdr.stix.dD_dw(omega, X, k)

            if root.k_par >= 0.0:
                y = cpdr.v_par + (dD_dk / dD_dw) * np.sqrt(1 + X**2)
            else:
                y = cpdr.v_par - (dD_dk / dD_dw) * np.sqrt(1 + X**2)

            y_axis.append(y.value)
            x_axis.append(X.value)

    return x_axis, y_axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rke", required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--resonance", required=True)
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    energy = args.rke << u.MeV  # Relativistic kinetic energy (Mega-electronvolts)
    alpha = Angle(args.alpha, u.deg)  # pitch angle in degrees
    resonance = int(args.resonance)  # resonance

    mlat_deg = Angle(0, u.deg)
    l_shell = 4.5

    particles = ("e", "p+")
    plasma_over_gyro_ratio = 1.5

    freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    X_min = 0.0 << u.dimensionless_unscaled
    X_max = 1.0 << u.dimensionless_unscaled
    X_npoints = 301
    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr_sym = CpdrSymbolic(len(particles))
    cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

    resonant_roots = cpdr.solve_resonant(X_range)
    # for row in resonant_roots:
    #     for root in row:
    #         print(f"X={root.X.value:.4f}, k_par={root.k_par.value:.6e}")
    #     print()

    x, y = compute_landau_term(cpdr, resonant_roots)
    plot(x, y, args.rke, args.alpha, args.resonance, args.save)


if __name__ == "__main__":
    main()
