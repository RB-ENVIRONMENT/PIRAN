import argparse
import json
from pathlib import Path

import numpy as np

from piran.diffusion import get_diffusion_coefficients


def calc_Daa_over_p_squared(pathname):
    """
    `pathname` must contain the `results*.json` files produced by the
    `diffusion_coefficients.py` script.
    """
    pitch_angle = []
    Daa_over_p_squared = []

    for file in Path(pathname).glob("results*.json"):
        with open(file, "r") as f:
            results = json.load(f)

        X_range = np.array(results["X_range"])
        alpha = results["pitch_angle"]
        resonances = results["resonances"]
        DnXaa = results["DnXaa"]
        momentum = results["momentum"]

        Daa = 0.0
        for i, resonance in enumerate(resonances):
            DnXaa_this_res = np.array(DnXaa[i])
            integral = get_diffusion_coefficients(X_range, DnXaa_this_res)
            Daa += integral

        pitch_angle.append(alpha)
        Daa_over_p_squared.append(Daa / momentum**2)

    # Sort by pitch angle
    sorted_vals = sorted(zip(pitch_angle, Daa_over_p_squared), key=lambda z: z[0])
    xx = [z[0] for z in sorted_vals]
    yy = [z[1] for z in sorted_vals]

    return (xx, yy)

def main():
    parser = argparse.ArgumentParser(
        prog="Cunningham_2023_Figure4",
        description="Reproduce Figure 4 from Cunningham, 2023",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        help="Path to Cunningham's dat file for Figure4.",
    )
    parser.add_argument(
        "--c1",
        default=None,
        help="Path to directory with results*.json files for Cunningham 1.5 ratio",
    )
    parser.add_argument(
        "--c2",
        default=None,
        help="Path to directory with results*.json files for Cunningham 10 ratio",
    )
    parser.add_argument(
        "--g1",
        default=None,
        help="Path to directory with results*.json files for Glauert 1.5 ratio",
    )
    parser.add_argument(
        "--g2",
        default=None,
        help="Path to directory with results*.json files for Glauert 10 ratio",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=False,
        help="Pass this argument to save the figure on disk.",
    )
    args = parser.parse_args()

    # Load data from Cunningham paper
    if args.path is not None:
        cunningham_dat_filepath = Path(args.path)
        if not cunningham_dat_filepath.is_file():
            msg = f"Incorrect path for Cunningham's dat file: {cunningham_dat_filepath}"
            raise Exception(msg)

        cunningham_figure_data = np.loadtxt(
            cunningham_dat_filepath,
            dtype=np.float64,
            delimiter=None,
            comments=";",
        )
    else:
        cunningham_figure_data = None

    # Our results
    if args.c1 is not None:
        piran_cunn_1 = calc_Daa_over_p_squared(args.c1)  # Cunningham ratio 1.5
    else:
        piran_cunn_1 = None

    if args.c2 is not None:
        piran_cunn_2 = calc_Daa_over_p_squared(args.c2)  # Cunningham ratio 10
    else:
        piran_cunn_2 = None

    if args.g1 is not None:
        piran_glau_1 = calc_Daa_over_p_squared(args.g1)  # Glauert & Horne ratio 1.5
    else:
        piran_glau_1 = None

    if args.g2 is not None:
        piran_glau_2 = calc_Daa_over_p_squared(args.g2)  # Glauert & Horne ratio 10
    else:
        piran_glau_2 = None


if __name__ == "__main__":
    main()
