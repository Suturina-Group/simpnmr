# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Utility helpers for SimpNMR.

This module provides constants, parsing helpers, formatting utilities, and
relaxation-rate helper functions used across the package.
"""

import math

import numpy as np
import scipy.constants as consts

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
HBAR = consts.hbar  # [J s radian-1]
H = consts.h  # [J s radian-1]
KB = consts.physical_constants["Boltzmann constant"][0]  # Boltzmann constant k [J·K⁻¹]
GE = abs(consts.physical_constants["electron g factor"][0])  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]


def flatten(biglist: list) -> list:
    """Flattens a nested list by one level.

    Args:
        biglist: A list of lists.

    Returns:
        A single list containing the concatenated elements.
    """
    return [item for sublist in biglist for item in sublist]


def find_mean_values(values: list[float], thresh: float = 0.1) -> list[int]:
    """Finds indices where a 1D sequence changes by at least a threshold.

    Args:
        values: Values to analyze.
        thresh: Threshold for identifying a step change.

    Returns:
        Indices at which the step size ``abs(diff(values))`` is greater than or equal
        to `thresh`.
    """

    # Find values for which step size is >= thresh
    mask = np.abs(np.diff(values)) >= thresh
    # and mark indices at which to split
    split_indices = np.where(mask)[0] + 1

    return [int(i) for i in split_indices]


def comp2ind(comp_str: str) -> list[int]:
    """Converts a tensor component label into matrix indices.

    Args:
        comp_str: Component string, e.g. ``"xy"``.

    Returns:
        A tuple ``(row, col)`` for the corresponding element of a ``(3, 3)`` tensor.
    """

    _c2i = {
        "xx": [0, 0],
        "xy": [0, 1],
        "xz": [0, 2],
        "yx": [1, 0],
        "yy": [1, 1],
        "yz": [1, 2],
        "zx": [2, 0],
        "zy": [2, 1],
        "zz": [2, 2],
    }

    return _c2i[comp_str][0], _c2i[comp_str][1]


def find_index_of_nearest(array, value):
    """Returns the index of the nearest value in a sorted array."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def isotope_format(isotope_string: str) -> str:
    r"""Formats an isotope label as Matplotlib mathtext.

    Args:
        isotope_string: Isotope label, e.g. ``"1H"`` or ``"13C"``.

    Returns:
        A mathtext string, e.g. ``$^\mathregular{13} \mathregular{C}$``.
    """

    # Split at number letter boundary
    for it, char in enumerate(isotope_string):
        if char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            split_at = it
            break
    nums = isotope_string[:split_at]
    lets = isotope_string[split_at:]

    return r"$^\mathregular{{{}}} \mathregular{{{}}}$".format(nums, lets)
