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

import json
from pathlib import Path

from astropy import units as u

from piran.diffusion import (
    UNIT_DIFF,
    get_energy_diffusion_coefficient,
    get_mixed_aE_diffusion_coefficient,
)


def load_and_post_process(pathname, bounce=False, fixed_energy=True):
    """
    `pathname` must contain the `results*.json` files produced by the
    `diffusion_coefficients.py` (`bounce=False`) or the
    `bounce_averaged_diffusion_coefficients.py` (`bounce=True`) script.
    `fixed_energy` must be True if all the output JSON files have the
    same energy or False if they have the same pitch angle.
    """

    x_values = []
    Daa_over_p_squared = []
    DaE_over_E_squared_abs = []
    DEE_over_E_squared = []

    files = list(Path(pathname).glob("results*.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"'results*.json' files not found in {pathname}")

    for file in files:
        with open(file, "r") as f:
            results = json.load(f)

        momentum = results["momentum"]
        rest_mass_energy_J = results["rest_mass_energy_Joule"] << u.J
        rel_kin_energy_MeV = results["rel_kin_energy_MeV"] << u.MeV
        rel_kin_energy_J = (rel_kin_energy_MeV).to(u.J)

        if fixed_energy:
            fixed_quantity_value = rel_kin_energy_MeV
            if bounce:
                xval = results["equatorial_pitch_angle"]
            else:
                xval = results["pitch_angle"]
        else:
            xval = rel_kin_energy_MeV.value
            if bounce:
                fixed_quantity_value = results["equatorial_pitch_angle"]
            else:
                fixed_quantity_value = results["pitch_angle"]

        method = results["method"]
        ratio = results["plasma_over_gyro_ratio"]
        X_max = results["X_max"]

        if bounce:
            Daa = results["baDaa"] << UNIT_DIFF
            Dap = results["baDap"] << UNIT_DIFF
            Dpp = results["baDpp"] << UNIT_DIFF
        else:
            Daa = results["Daa"] << UNIT_DIFF
            Dap = results["Dap"] << UNIT_DIFF
            Dpp = results["Dpp"] << UNIT_DIFF

        DaE = get_mixed_aE_diffusion_coefficient(
            rel_kin_energy_J, rest_mass_energy_J, Dap
        )
        DEE = get_energy_diffusion_coefficient(
            rel_kin_energy_J, rest_mass_energy_J, Dpp
        )

        x_values.append(xval)
        Daa_over_p_squared.append(Daa.value / momentum**2)
        DaE_over_E_squared_abs.append(abs(DaE.value) / rel_kin_energy_J.value**2)
        DEE_over_E_squared.append(DEE.value / rel_kin_energy_J.value**2)

    # Sort the data, using the x_values (pitch angle or energy) as the sorting key
    sorted_vals = sorted(
        zip(x_values, Daa_over_p_squared, DaE_over_E_squared_abs, DEE_over_E_squared),
        key=lambda z: z[0],
    )
    xx = [z[0] for z in sorted_vals]
    yy_aa = [z[1] for z in sorted_vals]
    yy_aE = [z[2] for z in sorted_vals]
    yy_EE = [z[3] for z in sorted_vals]

    return xx, yy_aa, yy_aE, yy_EE, method, ratio, X_max, fixed_quantity_value
