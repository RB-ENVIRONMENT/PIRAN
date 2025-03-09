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


def lists_are_identical(list1, list2, list3, list4):
    """
    Checks if four lists are identical (same elements in the same order).

    Returns:
        True if all four lists are identical, False otherwise.
    """

    if len(list1) != len(list2) or len(list1) != len(list3) or len(list1) != len(list4):
        return False  # Lists must have the same length

    for i in range(len(list1)):
        if list1[i] != list2[i] or list1[i] != list3[i] or list1[i] != list4[i]:
            return False  # Elements at the same index must be equal

    return True  # All checks passed, lists are identical
