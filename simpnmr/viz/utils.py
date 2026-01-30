# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO"""


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
