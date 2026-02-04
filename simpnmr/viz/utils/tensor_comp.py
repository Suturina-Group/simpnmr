# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Map tensor component labels to matrix indices.

Provides helpers to convert string tensor components (e.g. "xy") into
row and column indices of 3×3 tensors.
"""


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
