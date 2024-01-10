# This script compares the results from Cunningham's .dat file
# and the ones produced by the PIRAN code and plots the ratio,
# which in the best case, where the results from both codes match,
# should be 1.
# To run you need to pass a path to the .dat file from Cunningham
# and the .txt file from PIRAN with the `--cunningham` and `--piran`
# arguments respectively.
# By default the plot will be displayed on screen, but you can pass
# the `-s` argument and it will be saved on disk, in the current
# working directory, as "Figure2[abcd]_compare.png".
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="Compare_Cunningham_Figure2",
        description="Cunningham / PIRAN results for Figure 2 from Cunningham, 2023",
    )
    parser.add_argument("--cunningham", required=True)
    parser.add_argument("--piran", required=True)
    parser.add_argument("-s", "--save", action="store_true", default=False)
    args = parser.parse_args()

    cunningham_file = Path(args.cunningham)
    piran_file = Path(args.piran)
    if not (cunningham_file.is_file() and piran_file.is_file()):
        raise Exception("Incorrect files.")

    filestem = cunningham_file.stem

    cunningham_data = np.loadtxt(
        cunningham_file,
        dtype=np.float64,
        delimiter=None,
        comments=";",
    )

    piran_data = np.loadtxt(
        piran_file,
        dtype=np.float64,
        delimiter=",",
        comments="#",
    )

    ratio1 = np.empty((piran_data.shape[0], 2), dtype=np.float64)
    ratio2 = np.empty((piran_data.shape[0], 2), dtype=np.float64)
    for i, (row1, row2) in enumerate(zip(cunningham_data, piran_data)):
        # Perform a basic check that the X values match (columns 1 and 3)
        if row1[0] != row2[0] or row1[2] != row2[2]:
            raise Exception(f"X values do not match for index: {i}")

        ratio1[i, 0] = row1[0]
        ratio1[i, 1] = (
            row1[1] / row2[1]
        )  # Cunningham / Our results for omega ratio 0.1225 (black)

        ratio2[i, 0] = row1[2]
        ratio2[i, 1] = (
            row1[3] / row2[3]
        )  # Cunningham / Our results for omega ratio 0.5725 (red)

    # Plot
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
        }
    )

    plt.plot(ratio1[:, 0], ratio1[:, 1], "k", label=r"$\omega$/$\omega_{ce}$=0.1225")
    plt.plot(ratio2[:, 0], ratio2[:, 1], "r", label=r"$\omega$/$\omega_{ce}$=0.5725")

    plt.xlim(-0.2, 8.2)
    plt.ylim(-0.2, 3.2)

    x_ticks = [float(x) for x in range(9)]
    y_ticks = [float(x) for x in range(4)]
    plt.xticks(x_ticks, [str(v) for v in x_ticks])
    plt.yticks(y_ticks, [str(v) for v in y_ticks])
    plt.tick_params("x", which="both", top=True, labeltop=False)
    plt.tick_params("y", which="both", right=True, labelright=False)
    plt.minorticks_on()

    plt.xlabel(r"X ($\tan{\theta}$)")
    plt.ylabel("Ratio")
    plt.legend(loc="lower right")
    plt.title(f"Cunningham / PIRAN results for {filestem}")

    if args.save:
        plt.savefig(f"{filestem}_compare.png", dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
