# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Violin plot styling helpers.

Provides utilities to apply consistent coloring to Matplotlib violin plots.
"""


def set_violin_colours(violin: dict, color: str) -> None:
    """Sets violin-plot colours.

    Args:
        violin: Dictionary returned by ``Axes.violinplot``.
        color: Matplotlib color string.

    Returns:
        None.
    """
    for name, pc in violin.items():
        if name == "bodies":
            for part in pc:
                part.set_facecolor(color)
                part.set_edgecolor(color)
        else:
            pc.set_edgecolor(color)
    return
