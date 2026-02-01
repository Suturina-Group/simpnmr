# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read, validate, and write XYZ coordinate files.

Provides lightweight helpers for XYZ I/O, atom label normalization, and basic
geometry alignment utilities.
"""

import logging

import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike, NDArray

from ...core.constants import periodic_table

logger = logging.getLogger(__name__)


def add_label_indices(
    labels: ArrayLike, style: str = "per_element", start_index: int = 1
) -> list[str]:
    """Append numeric indices to atomic labels.

    Args:
        labels: Atomic symbols to index.
        style: Indexing scheme.
            - "per_element": Restart numbering for each element (e.g. C1, C2, N1).
            - "sequential": Number atoms consecutively regardless of element.
        start_index: First index value to use.

    Returns:
        A list of labels with indices appended.

    Raises:
        ValueError: If `style` is not a supported option.
    """

    # Normalize labels by stripping any existing indices.
    labels_nn = remove_label_indices(labels)

    # Number atoms consecutively regardless of element.
    if style == "sequential":
        labels_wn = [
            "{}{:d}".format(lab, it + start_index) for (it, lab) in enumerate(labels_nn)
        ]

    # Number atoms separately within each element type.
    elif style == "per_element":
        # Get list of unique elements
        unique_elements = set(labels_nn)
        # Create dict to keep track of index of current atom of each element
        atom_count = {el: start_index for el in unique_elements}
        # Create labelled list of elements
        labels_wn = []

        for lab in labels_nn:
            # Index according to dictionary
            labels_wn.append("{}{:d}".format(lab, atom_count[lab]))
            # Then add one to dictionary
            atom_count[lab] += 1
    else:
        raise ValueError("Unknown label style requested")

    return labels_wn


def remove_label_indices(labels: ArrayLike | str, debug: bool = False) -> list[str]:
    """Remove index suffixes from atomic labels.

    Examples of supported inputs include "H1", "H2a", and "C12". The function
    removes trailing digits (and any trailing letters after those digits).

    Args:
        labels: A label or sequence of labels.
        debug: Reserved for diagnostics.

    Returns:
        Labels with index suffixes removed. The return type mirrors the input type.
    """

    if isinstance(labels, str):
        _labels = [labels]
    else:
        _labels = labels

    labels_nn = []
    for label in _labels:
        no_digits = []
        for i in label:
            if not i.isdigit():
                no_digits.append(i)
            elif i.isdigit():
                break
        result = "".join(no_digits)
        labels_nn.append(result)

    if isinstance(labels, str):
        return labels_nn[0]
    else:
        return labels_nn


def num_to_lab(numbers: ArrayLike, numbered: bool = True) -> list[str]:
    """Convert atomic numbers to element symbols.

    Args:
        numbers: Atomic numbers.
        numbered: If True, append indices to the resulting labels.

    Returns:
        Element symbols for the provided atomic numbers.
    """

    labels = [periodic_table.num_lab[int(num)] for num in numbers]

    if numbered:
        labels_wn = add_label_indices(labels)
    else:
        labels_wn = labels

    return labels_wn


def lab_to_num(labels: ArrayLike | str) -> list[int]:
    """Convert element symbols to atomic numbers.

    Args:
        labels: A label or sequence of labels. Index suffixes are allowed.

    Returns:
        Atomic numbers corresponding to the provided labels. The return type mirrors
        the input type.
    """

    if isinstance(labels, str):
        _labels = [labels]
    else:
        _labels = labels

    labels_nn = remove_label_indices(_labels)

    numbers = [periodic_table.lab_num[lab] for lab in labels_nn]

    if isinstance(labels, str):
        return numbers[0]
    else:
        return numbers


def load_xyz(
    f_name: str,
    atomic_numbers: bool = False,
    add_indices: bool = False,
    capitalise: bool = True,
    check: bool = True,
    missing_headers: bool = False,
) -> tuple[list, NDArray]:
    """Load labels and Cartesian coordinates from an XYZ file.

    Supports standard XYZ files (NATOMS + optional comment line) and files with
    missing headers.

    Args:
        f_name: Path to the XYZ file.
        atomic_numbers: If True, interpret the first column as atomic numbers.
        add_indices: If True, (re)assign indices on atomic labels.
        capitalise: If True, capitalise element symbols (e.g. "fe" -> "Fe").
        check: If True, run lightweight validation before reading.
        missing_headers: If True, assume no NATOMS/comment header lines are present.

    Returns:
        A tuple of:
            - labels: List of atomic labels.
            - coords: (n_atoms, 3) float array of Cartesian coordinates.

    Raises:
        FileNotFoundError: If `f_name` does not exist.
        ValueError: If numeric parsing fails (propagated from NumPy).
        OSError: For underlying I/O errors.
    """

    # Optional preflight validation.
    if check:
        logger.info("Checking XYZ file: %s", f_name)
        check_xyz(
            f_name,
            allow_nonelements=atomic_numbers,
            allow_missing_headers=missing_headers,
        )

    # Choose the header offset (0/1/2) unless the caller forces missing headers.
    if missing_headers:
        skiprows = 0
    else:
        # default to 2, but inspect file to decide between 0,1,2
        skiprows = 2
        with open(f_name, "r") as _f:
            first = _f.readline()
            parts = first.split()
            # If first line looks like coordinates (label + 3 floats) -> no headers
            if len(parts) >= 4:
                try:
                    float(parts[-1])
                    float(parts[-2])
                    float(parts[-3])
                    skiprows = 0
                except Exception:
                    pass
            # If first line is single integer, inspect second line:
            elif len(parts) == 1:
                second = _f.readline()
                if second:
                    sp = second.split()
                    if len(sp) >= 4:
                        try:
                            float(sp[-1])
                            float(sp[-2])
                            float(sp[-3])
                            # second line looks like coordinates -> only natoms header
                            skiprows = 1
                        except Exception:
                            skiprows = 2

    if atomic_numbers:
        _numbers = np.loadtxt(f_name, skiprows=skiprows, usecols=0, dtype=int, ndmin=1)
        _labels = num_to_lab(_numbers.tolist())
    else:
        _labels = np.loadtxt(f_name, skiprows=skiprows, usecols=0, dtype=str, ndmin=1)
        _labels = _labels.tolist()

    # Canonicalize label casing.
    if capitalise:
        _labels = [lab.capitalize() for lab in _labels]

    if add_indices:
        _labels = remove_label_indices(_labels)
        _labels = add_label_indices(_labels)

    _coords = np.loadtxt(f_name, skiprows=skiprows, usecols=(1, 2, 3), ndmin=2)

    return _labels, _coords


def check_xyz(
    f_name: str,
    allow_indices: bool = True,
    allow_nonelements: bool = False,
    allow_missing_headers: bool = False,
) -> None:
    """Validate basic structure and label content of an XYZ file.

    This function is intentionally non-throwing: on validation failure it logs an
    error and returns.

    Args:
        f_name: Path to the XYZ file.
        allow_indices: If False, disallow index suffixes on labels.
        allow_nonelements: If True, allow labels that are not chemical elements.
        allow_missing_headers: If True, permit files without NATOMS/comment headers.

    Returns:
        None.
    """

    # Check file contains number of atoms on first line and comment on second
    if not allow_missing_headers:
        _check_xyz_headers(f_name)

    try:
        _labels, _ = load_xyz(f_name, capitalise=False, check=False)
    except Exception as exc:
        logger.error("Failed to read XYZ file: %s", exc)
        return

    # Compare labels with indices removed.
    _labels_nn = remove_label_indices(_labels)

    # Check all entries are real elements
    if not allow_nonelements:
        if any([lab not in periodic_table.elements for lab in _labels_nn]):
            logger.error("XYZ file contains non-elements")
            return

    # Check if indices are present
    if not allow_indices:
        if any([labnn != lab for labnn, lab in zip(_labels_nn, _labels)]):
            logger.error("XYZ file contains elements with indices")
            return

    return


def _check_xyz_headers(f_name: str):
    """Validate XYZ header lines and atom count consistency.

    This helper is intentionally non-throwing: on failure it logs an error and returns.

    Args:
        f_name: Path to the XYZ file.

    Returns:
        None.
    """

    with open(f_name, "r") as f:
        line = next(f)
        parts = line.split()
        # if first line isn't a single token, check if it looks like coordinates
        if len(parts) != 1:
            if len(parts) >= 4:
                try:
                    float(parts[-1])
                    float(parts[-2])
                    float(parts[-3])
                    # first line appears to be coordinates -> treat as missing headers
                    return
                except Exception:
                    logger.error("XYZ file does not contain number of atoms")
                    return
            else:
                logger.error("XYZ file does not contain number of atoms")
                return
        try:
            n_atoms = int(line)
        except ValueError:
            logger.error("XYZ file number of atoms is malformed")
            return

        n_lines = len(f.readlines()) + 1
        # Accept either NATOMS+comment or NATOMS-only headers.
        if not (n_lines == n_atoms + 2 or n_lines == n_atoms + 1):
            logger.error("XYZ file length/format is incorrect")
            return

    return


def find_rotation(coords_1: ArrayLike, coords_2: ArrayLike) -> tuple[NDArray, float]:
    """Compute the optimal rotation aligning `coords_2` onto `coords_1`.

    Uses an SVD-based Kabsch alignment and returns the rotation matrix and RMSD.

    Args:
        coords_1: Reference coordinates with shape (n_atoms, 3).
        coords_2: Coordinates to rotate, shape (n_atoms, 3).

    Returns:
        A tuple of:
            - R: (3, 3) rotation matrix.
            - rmsd: RMSD after alignment.
    """

    # Calculate B matrix
    coords_1 = np.asarray(coords_1)
    coords_2 = np.asarray(coords_2)

    assert coords_1.shape == coords_2.shape, (
        f"Coordinate shapes must match, got {coords_1.shape} and {coords_2.shape}"
    )

    # Cross-covariance matrix for Kabsch alignment.
    B = coords_1.T @ coords_2

    # Calculate SVD of B matrix
    U, _, Vt = la.svd(B)

    # Calculate M matrix
    M = np.diag([1, 1, la.det(U) * la.det(Vt)])

    # Calculate rotation matrix
    R = U @ M @ Vt

    # Apply rotation matrix to coords_2
    coords_2_rotated = (R @ coords_2.T).T

    # Calculate rmsd
    rmsd = _calculate_rmsd(coords_1, coords_2_rotated)

    return R, rmsd


def _calculate_rmsd(coords_1: ArrayLike, coords_2: ArrayLike) -> float:
    """Compute RMSD between two coordinate arrays.

    Args:
        coords_1: First coordinate array, shape (n_atoms, 3).
        coords_2: Second coordinate array, shape (n_atoms, 3).

    Returns:
        RMSD between the two structures.
    """

    coords_1 = np.asarray(coords_1)
    coords_2 = np.asarray(coords_2)

    # Check there are the same number of coordinates
    assert len(coords_1) == len(coords_2)

    diff = coords_1 - coords_2
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

    return rmsd
