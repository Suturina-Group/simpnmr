# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define small array and list utility helpers.

Provides simple helpers for flattening lists and locating indices in numeric arrays.
"""

import math

import numpy as np


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
