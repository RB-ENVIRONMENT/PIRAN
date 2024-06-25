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
