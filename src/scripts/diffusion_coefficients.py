import argparse
import json
import pathlib

from astropy import units as u
from astropy.coordinates import Angle


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

if __name__ == "__main__":
    main()
