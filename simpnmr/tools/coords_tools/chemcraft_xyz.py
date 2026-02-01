# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Extract ChemCraft atom labels from annotated XYZ files.

Reads ChemCraft-style XYZ files and writes per-atom chemical labels to CSV format.
"""

import argparse
import csv
import logging

from . import xyz_format as xyzf

logger = logging.getLogger(__name__)


# TODO: detect_xyz_formatting????
def load_chemcraft_xyz(file_name: str):
    """
    Load a Chemcraft-annotated XYZ file and extract per-atom captions.

    Chemcraft XYZ files follow the standard XYZ format but may include an
    optional 5th column containing an atomic caption/label in double quotes.

    Args:
        file_name (str): Path to the input XYZ file.

    Returns:
        tuple[dict[str, str], list[str], np.ndarray]:
            - chem_dict (dict[str, str]): Mapping from indexed atom labels
              (e.g. "C1") to Chemcraft captions. Missing captions are returned
              as empty strings.
            - indexed_labels (list[str]): Indexed atom labels in file order.
            - coords (np.ndarray): Cartesian coordinates with shape (N, 3).

    Raises:
        SystemExit: If the file cannot be parsed as an XYZ file or contains
            invalid formatting.
    """

    # Read labels and coordinates
    try:
        _labels, coords = xyzf.load_xyz(file_name, check=False)
    except (ValueError, xyzf.XYZError) as vxe:
        raise ValueError(str(vxe))

    if all(label.isdigit() for label in _labels):
        indexed_labels = xyzf.add_label_indices(xyzf.num_to_lab(_labels))
    else:
        indexed_labels = xyzf.add_label_indices(_labels)

    chem_dict = dict.fromkeys(indexed_labels, "")

    # Read chemical labels
    with open(file_name, "r") as f:
        for it, line in enumerate(f):
            if it < 2:
                continue
            spl = line.split()
            if len(spl) > 4:
                chem_dict[indexed_labels[it]] = spl[4].replace('"', "")

    return chem_dict, indexed_labels, coords


def main():
    # Configure logging for standalone entry-point usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "This script converts annotated chemcraft .xyz files into a\n"
            "chemlabels csv file for use with SimpNMR"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        type=str,
        help=".xyz file containing atomic captions in Chemcraft style",
    )

    parser.add_argument(
        "-m",
        "--math_placeholder",
        action="store_true",
        help="Add placeholder column for chem_math_labels",
    )

    uargs = parser.parse_args()

    # Get chemical labels from chemcraft xyz file
    chem_dict, _, _ = load_chemcraft_xyz(uargs.input_file)

    # Remove entries with no chemlabel
    chem_dict = {k: v for k, v in chem_dict.items() if v}

    # Create placeholder math labels
    if uargs.math_placeholder:
        math_dict = {k: f"${v}$" for k, v in chem_dict.items()}

    # Save new chemlabels file
    with open("chemlabels.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if uargs.math_placeholder:
            writer.writerow(["atom_label", "chem_label", "chem_math_label"])
        else:
            writer.writerow(["atom_label", "chem_label"])

        for k, v in chem_dict.items():
            row = [k, v]
            if uargs.math_placeholder:
                row.append(math_dict[k])
            writer.writerow(row)

    logger.info("Chemical labels written to chemlabels.csv")
