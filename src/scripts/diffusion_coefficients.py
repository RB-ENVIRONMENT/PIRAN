import argparse
import json
import pathlib


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


if __name__ == "__main__":
    main()
