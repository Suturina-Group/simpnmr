# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write molecular structures to XYZ files.

Provides helpers to export atomic labels and Cartesian coordinates to standard
XYZ format, with optional metadata and ChemCraft-compatible labels.
"""

import datetime
import logging

import numpy as np
from numpy.typing import ArrayLike

from simpnmr.__version__ import __version__
from simpnmr.tools.coords import xyz_fmt
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)


def save_xyz(
    file_name: str,
    labels: ArrayLike,
    coords: ArrayLike,
    with_numbers: bool = False,
    verbose: bool = True,
    mask: list | None = None,
    atomic_numbers: bool = False,
    comment: str = "",
) -> None:
    """Write an XYZ file from labels and coordinates.

    Args:
        file_name: Output file name.
        labels: Atomic labels.
        coords: Cartesian coordinates with shape (n_atoms, 3).
        with_numbers: If True, overwrite labels with freshly assigned indices.
        verbose: If True, print a confirmation message after writing.
        mask: Indices of atoms to exclude from output.
        atomic_numbers: If True, write atomic numbers instead of element symbols.
        comment: User comment appended after an auto-generated provenance string
        on the second line of the XYZ file.

    Returns:
        None.

    Raises:
        OSError: For underlying I/O errors.
    """

    coords = np.asarray(coords)
    mask = mask or []

    _comment = f"This file was generated with SimpNMR v{__version__} at {{}}. ".format(
        datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    )
    _comment += comment
    if not _comment.endswith("\n"):
        _comment += "\n"

    # Option to have numbers added
    if with_numbers:
        # Remove and re-add numbers to be safe
        _labels = xyz_fmt.remove_label_indices(labels)
        _labels = xyz_fmt.add_label_indices(_labels)
    else:
        _labels = labels

    # Set up masks
    if mask:
        coords = np.delete(coords, mask, axis=0)
        _labels = np.delete(_labels, mask, axis=0).tolist()

    n_atoms = len(_labels)

    if atomic_numbers:
        _labels = xyz_fmt.remove_label_indices(_labels)
        _numbers = xyz_fmt.lab_to_num(_labels)
        _identifier = _numbers
    else:
        _identifier = _labels

    with open(file_name, "w") as f:
        f.write(f"{n_atoms:d}\n")
        f.write(_comment)

        for i, (ident, trio) in enumerate(zip(_identifier, coords)):
            if i == 0:
                f.write("{:5} {:15.7f} {:15.7f} {:15.7f}".format(ident, *trio))
            else:
                f.write("\n{:5} {:15.7f} {:15.7f} {:15.7f}".format(ident, *trio))

    if verbose:
        logger.info("New XYZ file written to %s", file_name)

    return


def save_chemcraft_xyz(
    file_name: str,
    labels: ArrayLike,
    coords: ArrayLike,
    chem_labels: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Save an XYZ file with ChemCraft-compatible chemical labels.

    ChemCraft can display per-atom labels if an extra quoted string is appended to
    each coordinate line. This writer appends per-atom chemical labels when provided.
    """
    coords = np.asarray(coords)
    chem_labels = chem_labels or {}

    with open(file_name, "w") as f:
        for lab, trio in zip(labels, coords):
            f.write(
                "{:5} {:15.7f} {:15.7f} {:15.7f}".format(xyzf.lab_to_num(lab), *trio)
            )
            if lab in chem_labels:
                f.write(f'      "{chem_labels[lab]}"\n')
            else:
                f.write("\n")

    if verbose:
        logger.info("ChemCraft XYZ file written to %s", file_name)
