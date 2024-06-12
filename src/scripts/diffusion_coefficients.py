import argparse
import json
from importlib import metadata
import pathlib

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.diffusion import (
    get_DnX_single_root,
    get_normalised_intensity,
    get_phi_squared,
    get_power_spectral_density,
    get_singular_term,
)
from piran.gauss import Gaussian
from piran.magpoint import MagPoint
from piran.normalisation import (
    compute_cunningham_norm_factor,
    compute_glauert_norm_factor,
)
from piran.plasmapoint import PlasmaPoint


def main():
    parser = argparse.ArgumentParser(
        prog="Diffusion_coefficints",
        description="Calculate diffusion coefficients",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="Specify the input JSON file",
    )
    args = parser.parse_args()

    # Load input json file as dict
    with open(args.input) as infile:
        parameters = json.load(infile)

    # ================ Parameters ===================================

    particles = tuple(parameters["particles"])
    energy = float(parameters["energy"]) << u.MeV
    alpha = Angle(parameters["pitch_angle"], u.deg)
    plasma_over_gyro_ratio = float(parameters["plasma_over_gyro_ratio"])
    mlat_deg = Angle(parameters["mlat"], u.deg)
    l_shell = float(parameters["l_shell"])
    resonances = list(parameters["resonances"])
    X_min = float(parameters["X_min"]) << u.dimensionless_unscaled
    X_max = float(parameters["X_max"]) << u.dimensionless_unscaled
    X_npoints = int(parameters["X_npoints"])
    X_m = float(parameters["X_m"]) << u.dimensionless_unscaled
    X_w = float(parameters["X_w"]) << u.dimensionless_unscaled
    freq_cutoff_params = tuple(parameters["freq_cutoff_params"])
    wave_amplitude = float(parameters["wave_amplitude"]) << u.T
    method = int(parameters["method"])

    # ===============================================================

    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    if method == 0:
        epsilon = 0.9999  # Glauert & Horne 2005 paragraph 23
    elif method == 1:
        epsilon = 1.0
    else:
        raise Exception("Wrong method")

    mag_point = MagPoint(mlat_deg, l_shell)
    plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
    cpdr_sym = CpdrSymbolic(len(particles))
    wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)

    # Calculate integral of g(X) to normalise it.
    # Cunningham used left endpoint integration rule in his paper.
    if method == 1:
        integral_gx = simpson(wave_norm_angle_dist.eval(X_range), x=X_range)

    container = {}
    container["version"] = metadata.version("piran")
    container["particles"] = particles
    container["rel_kin_energy_MeV"] = energy.to(u.MeV).value
    container["pitch_angle"] = alpha.deg
    container["plasma_over_gyro_ratio"] = plasma_over_gyro_ratio
    container["mlat"] = mlat_deg.value
    container["l_shell"] = l_shell
    container["resonances"] = resonances
    container["X_min"] = X_min.value
    container["X_max"] = X_max.value
    container["X_npoints"] = X_npoints
    container["X_range"] = X_range.value.tolist()
    container["X_m"] = X_m.value
    container["X_w"] = X_w.value
    container["freq_cutoff_params"] = freq_cutoff_params
    container["wave_amplitude"] = wave_amplitude.value
    container["method"] = method
    container["DnXaa"] = []
    container["DnXap"] = []
    container["DnXpp"] = []

    for resonance in resonances:
        DnXaa_this_res = []
        DnXap_this_res = []
        DnXpp_this_res = []

        cpdr = Cpdr(cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params)

        # Depends only on energy and mass. Will be the same for different resonances.
        container["momentum"] = cpdr.momentum.value
        container["rest_mass_energy_Joule"] = (cpdr.plasma.particles[0].mass.to(u.kg) * const.c**2).to(u.J).value

        resonant_roots = cpdr.solve_resonant(X_range)
        for roots_this_x in resonant_roots:

            DnXaa_this_X = 0.0
            DnXap_this_X = 0.0
            DnXpp_this_X = 0.0
            for root in roots_this_x:

                if np.isnan(root.omega) or np.isnan(root.k):
                    continue

                # See par.23 in Glauert & Horne 2005
                resonance_cone_angle = -cpdr.stix.P(root.omega) / cpdr.stix.S(root.omega)
                X_upper = min(X_max, epsilon * np.sqrt(resonance_cone_angle))
                if root.X.value > X_upper.value:
                    continue

                if method == 0:
                    eval_gx = wave_norm_angle_dist.eval(root.X)

                    # For Glauert's norm factor we limit the range in X
                    # between X_min and X_upper=min(X_max, epsilon*sqrt(-P/S)).
                    X_range_glauert = X_range[X_range <= X_upper]

                    norm_factor = compute_glauert_norm_factor(
                        cpdr,
                        root.omega,
                        X_range_glauert,
                        wave_norm_angle_dist,
                        method="simpson"
                    )
                elif method == 1:
                    if root.X.value == 0.0:
                        # Avoid division by zero
                        continue
                    eval_gx = wave_norm_angle_dist.eval(root.X) / integral_gx
                    norm_factor = compute_cunningham_norm_factor(
                        cpdr,
                        root.omega,
                        [root.X] << root.X.unit,
                    )[0]

                power_spectral_density = get_power_spectral_density(cpdr, wave_amplitude, root.omega)
                normalised_intensity = get_normalised_intensity(
                    power_spectral_density, eval_gx, norm_factor
                )
                phi_squared = get_phi_squared(cpdr, root)
                singular_term = get_singular_term(cpdr, root)

                DnXaa_this_root, DnXap_this_root, DnXpp_this_root = get_DnX_single_root(
                    cpdr,
                    root,
                    normalised_intensity,
                    phi_squared,
                    singular_term,
                )

                DnXaa_this_X += DnXaa_this_root.value
                DnXap_this_X += DnXap_this_root.value
                DnXpp_this_X += DnXpp_this_root.value

            DnXaa_this_res.append(DnXaa_this_X)
            DnXap_this_res.append(DnXap_this_X)
            DnXpp_this_res.append(DnXpp_this_X)

        container["DnXaa"].append(DnXaa_this_res)
        container["DnXap"].append(DnXap_this_res)
        container["DnXpp"].append(DnXpp_this_res)

    with open(f"results_{alpha.deg:.2f}deg.json", "w") as outfile:
        json.dump(container, outfile, indent=4)


if __name__ == "__main__":
    main()
