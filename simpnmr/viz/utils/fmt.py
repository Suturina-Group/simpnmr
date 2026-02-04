# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Format labels for Matplotlib rendering.

Provides helpers to format isotope labels using Matplotlib mathtext.
"""


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
