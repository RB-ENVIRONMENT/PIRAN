# Copyright (C) 2025 The University of Birmingham, United Kingdom /
#   Dr Oliver Allanson, ORCiD: 0000-0003-2353-8586, School Of Engineering, University of Birmingham /
#   Dr Thomas Kappas, ORCiD: 0009-0003-5888-2093, Advanced Research Computing, University of Birmingham /
#   Dr James Tyrrell, ORCiD: 0000-0002-2344-737X, Advanced Research Computing, University of Birmingham /
#   Dr Adrian Garcia, ORCiD: 0009-0007-4450-324X, Advanced Research Computing, University of Birmingham
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
Create an `input.json` file with the following:
| Name                     | Data type        | Example                 | Unit | Info                                                |
|--------------------------|------------------|-------------------------|------|-----------------------------------------------------|
| "particles"              | Array of Strings | ["e", "p+"]             |      |                                                     |
| "energy"                 | Number           | 1.0                     | MeV  | Relativistic kinetic energy                         |
| "equatorial_pitch_angle" | Number           | 60.0                    | deg  |                                                     |
| "plasma_over_gyro_ratio" | Number           | 1.5                     |      |                                                     |
| "mlat_npoints"           | Number           | 30                      |      |                                                     |
| "mlat_cutoff"            | Number           | 15.0                    | deg  | Magnetic latitude cutoff (use "inf" if not desired) |
| "l_shell"                | Number           | 4.5                     |      |                                                     |
| "resonances"             | Array of Numbers | [-2, -1, 0, 1]          |      |                                                     |
| "X_min"                  | Number           | 0.0                     |      |                                                     |
| "X_max"                  | Number           | 1.0                     |      |                                                     |
| "X_npoints"              | Number           | 1000                    |      |                                                     |
| "X_m"                    | Number           | 0.0                     |      | Peak                                                |
| "X_w"                    | Number           | 0.577                   |      | Angular width                                       |
| "freq_cutoff_params"     | Array of Numbers | [0.35, 0.15, -1.5, 1.5] |      | See par.30 Glauert and Horne 2005                   |
| "wave_amplitude"         | Number           | 1e-10                   | T    |                                                     |
| "method"                 | Number           | 0                       |      | 0 for Glauert, 1 for Cunningham                     |

for example:
{
    "particles": ["e", "p+"],
    "energy": 1.0,
    "equatorial_pitch_angle": 60.0,
    "plasma_over_gyro_ratio": 1.5,
    "mlat_npoints": 30,
    "mlat_cutoff": 15.0,
    "l_shell": 4.5,
    "resonances": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "X_min": 0.0,
    "X_max": 1.0,
    "X_npoints": 600,
    "X_m": 0.0,
    "X_w": 0.577,
    "freq_cutoff_params": [0.35, 0.15, -1.5, 1.5],
    "wave_amplitude": 1e-10,
    "method": 0
}

and run the script as: `python path/to/bounce_averaged_diffusion_coefficients.py -i path/to/input.json`

The mlat_cutoff setting is used to impose a maximum magnetic latitude at which waves are present
(disregarding any solutions beyond this point). If this is not desired, set it to "inf".

The script will create a file `results_[ANGLE]deg_[ENERGY]MeV.json` in the
current working directory, where [ANGLE] is the equatorial pitch angle in degrees
and [ENERGY] the relativistic kinetic energy in MeV for this run.
"""

import argparse
import json
import pathlib
from importlib import metadata

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.bounce import Bounce
from piran.cpdr import Cpdr
from piran.diffusion import (
    UNIT_DIFF,
    get_diffusion_coefficients,
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


def get_DnX_per_X(
    cpdr,
    X_range,
    X_max,
    epsilon,
    wave_norm_angle_dist,
    integral_gx,
    wave_amplitude,
    method,
):
    """
    Given a cpdr object (which means that resonance is fixed) calculate
    equations 11, 12 and 13 from Glauert & Horne 2005.
    It returns three lists one for each DnXaa, DnXap and DnXpp.
    Each individual list contains one value per X.
    """
    DnXaa_this_res = []
    DnXap_this_res = []
    DnXpp_this_res = []

    resonant_roots = cpdr.solve_resonant(X_range)

    for roots_this_x in resonant_roots:
        DnXaa_this_X = 0.0
        DnXap_this_X = 0.0
        DnXpp_this_X = 0.0
        for root in roots_this_x:
            if np.isnan(root.omega) or np.isnan(root.k):
                continue

            # Cap the value of X according to the resonance cone angle
            # See par.23 in Glauert & Horne 2005
            rca_squared = -cpdr.stix.P(root.omega) / cpdr.stix.S(root.omega)

            if rca_squared < 0:
                print(f"Warning: imaginary resonance cone angle for omega={root.omega}")

            X_upper = (
                min(X_max, epsilon * np.sqrt(rca_squared))
                if rca_squared >= 0
                else X_max
            )
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
                    method="simpson",
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

            power_spectral_density = get_power_spectral_density(
                cpdr, wave_amplitude, root.omega
            )
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

    return DnXaa_this_res, DnXap_this_res, DnXpp_this_res


def main():
    parser = argparse.ArgumentParser(
        prog="Bounce-averaged diffusion coefficints",
        description="Calculate bounce-averaged diffusion coefficients",
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
    equatorial_pitch_angle = Angle(float(parameters["equatorial_pitch_angle"]), u.deg)
    plasma_over_gyro_ratio = float(parameters["plasma_over_gyro_ratio"])
    mlat_npoints = int(parameters["mlat_npoints"])
    mlat_cutoff = Angle(float(parameters["mlat_cutoff"]), u.deg)
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

    container = {}
    container["version"] = metadata.version("piran")
    container["particles"] = particles
    container["rel_kin_energy_MeV"] = energy.to(u.MeV).value
    container["equatorial_pitch_angle"] = equatorial_pitch_angle.deg
    container["plasma_over_gyro_ratio"] = plasma_over_gyro_ratio
    container["mlat_npoints"] = mlat_npoints
    container["mlat_cutoff"] = mlat_cutoff.deg
    container["l_shell"] = l_shell
    container["resonances"] = resonances
    container["X_min"] = X_min.value
    container["X_max"] = X_max.value
    container["X_npoints"] = X_npoints
    container["X_m"] = X_m.value
    container["X_w"] = X_w.value
    container["freq_cutoff_params"] = freq_cutoff_params
    container["wave_amplitude"] = wave_amplitude.value
    container["method"] = method

    X_range = u.Quantity(
        np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
    )

    if method == 0:
        epsilon = 0.9999  # Glauert & Horne 2005 paragraph 23
    elif method == 1:
        epsilon = 1.0
    else:
        raise Exception("Wrong method")

    wave_norm_angle_dist = Gaussian(X_min, X_max, X_m, X_w)
    bounce = Bounce(equatorial_pitch_angle, MagPoint(0.0 << u.rad, l_shell))

    # Calculate integral of g(X) to normalise it.
    # Cunningham used left endpoint integration rule in his paper.
    if method == 1:
        integral_gx = simpson(wave_norm_angle_dist.eval(X_range), x=X_range)
    elif method == 0:
        integral_gx = None  # Unused for Glauert's norm factor
    else:
        raise ValueError("Invalid normalisation method specified.")

    # Create integration range for equations 24, 25 and 26 in Glauert 2005
    lambda_min = 0.0 << u.rad
    lambda_max = (
        0.9999 * bounce.mirror_latitude << u.rad
    )  # Cap this at 0.9999 to avoid (integrable) singularity
    lambda_range = Angle(np.linspace(lambda_min, lambda_max, mlat_npoints), unit=u.rad)
    container["mlat_range"] = lambda_range.value.tolist()

    # Start the main loop over the magnetic latitudes.
    # Given magnetic latitude we will calculate the pitch angle
    # and then the Daa, Dap, Dpp and the three integrands for the
    # respective bounce-averaged diffusion coefficients.
    # It's important to initialise with zeros as if pitch angle
    # is invalid, we will continue with the next latitude.
    baDaa_integrand = u.Quantity(np.zeros(mlat_npoints, dtype=np.float64), UNIT_DIFF)
    baDap_integrand = u.Quantity(np.zeros(mlat_npoints, dtype=np.float64), UNIT_DIFF)
    baDpp_integrand = u.Quantity(np.zeros(mlat_npoints, dtype=np.float64), UNIT_DIFF)

    for ii, mlat in enumerate(lambda_range):
        if mlat >= mlat_cutoff:
            continue

        pitch_angle = bounce.get_bounce_pitch_angle(mlat)
        if (
            np.isnan(pitch_angle)
            or pitch_angle <= 0.0 << u.rad
            or pitch_angle >= np.pi / 2 << u.rad
        ):
            continue

        mag_point = MagPoint(mlat, l_shell)
        if mlat == lambda_min: 
            plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
            number_density_at_equator = plasma_point.number_density
        else: 
            plasma_point = PlasmaPoint(
                mag_point,
                particles,
                number_density=number_density_at_equator
            )

        Dnaa = []
        Dnap = []
        Dnpp = []
        for resonance in resonances:
            cpdr = Cpdr(
                plasma_point,
                energy,
                pitch_angle,
                resonance,
                freq_cutoff_params,
            )

            # Depends only on energy and mass. Will be the same for different
            # resonances and latitudes.
            container["momentum"] = cpdr.momentum.value
            container["rest_mass_energy_Joule"] = (
                (cpdr.plasma.particles[0].mass.to(u.kg) * const.c**2).to(u.J).value
            )

            # Calculate equations 11, 12 and 13 from
            # Glauert & Horne 2005, for this resonance.
            DnXaa_this_res, DnXap_this_res, DnXpp_this_res = get_DnX_per_X(
                cpdr,
                X_range,
                X_max,
                epsilon,
                wave_norm_angle_dist,
                integral_gx,
                wave_amplitude,
                method,
            )

            # Calculate the integrals from equations 8, 9 and 10 in
            # Glauert & Horne 2005, only for this resonance.
            Dnaa_this_res = get_diffusion_coefficients(
                X_range, DnXaa_this_res << UNIT_DIFF
            )
            Dnap_this_res = get_diffusion_coefficients(
                X_range, DnXap_this_res << UNIT_DIFF
            )
            Dnpp_this_res = get_diffusion_coefficients(
                X_range, DnXpp_this_res << UNIT_DIFF
            )
            Dnaa.append(Dnaa_this_res)
            Dnap.append(Dnap_this_res)
            Dnpp.append(Dnpp_this_res)

        # Sum the diffusion coefficients for all the resonances.
        # This is essentially what is calculated in equations 8, 9
        # and 10 in Glauert & Horne 2005.
        Daa = np.sum([v.value for v in Dnaa]) << UNIT_DIFF
        Dap = np.sum([v.value for v in Dnap]) << UNIT_DIFF
        Dpp = np.sum([v.value for v in Dnpp]) << UNIT_DIFF

        baDaa_integrand[ii] = Daa * bounce.get_pitch_angle_factor(mlat)
        baDap_integrand[ii] = Dap * bounce.get_mixed_factor(mlat)
        baDpp_integrand[ii] = Dpp * bounce.get_momentum_factor(mlat)

    # Scipy's simpson strips the units so we might want to re-add them here manually
    # If we don't, the unit will be "dimensionless_unscaled" which is incorrect.
    baDaa = simpson(baDaa_integrand, x=lambda_range) / bounce.particle_bounce_period
    baDap = simpson(baDap_integrand, x=lambda_range) / bounce.particle_bounce_period
    baDpp = simpson(baDpp_integrand, x=lambda_range) / bounce.particle_bounce_period
    container["baDaa"] = baDaa.value
    container["baDap"] = baDap.value
    container["baDpp"] = baDpp.value

    # Write the results to disk as a JSON formatted file
    formatted_angle = f"{equatorial_pitch_angle.deg:.3f}"
    formatted_energy = f"{energy.to(u.MeV).value:.10f}"
    filename = f"results_{formatted_angle}deg_{formatted_energy}MeV.json"
    with open(filename, "w") as outfile:
        json.dump(container, outfile, indent=4)


if __name__ == "__main__":
    main()
