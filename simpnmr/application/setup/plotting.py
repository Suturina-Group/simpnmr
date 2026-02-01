# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define plot mode options and helpers.

Provides utilities to map plot modes to show/save flags used by pipelines.
"""

from __future__ import annotations

from typing import Literal

PlotMode = Literal["on", "save", "show", "off"]

# Plot-mode helpers used across pipelines.
SHOW_CONV: dict[PlotMode, bool] = {
    "on": True,
    "save": False,
    "show": True,
    "off": False,
}

SAVE_CONV: dict[PlotMode, bool] = {
    "on": True,
    "save": True,
    "show": False,
    "off": False,
}

PLOT_ACTIVE: set[PlotMode] = {"on", "show", "save"}


def mode_to_show_save(mode: PlotMode) -> tuple[bool, bool]:
    """Convert a PlotMode into (show, save) flags.

    Args:
        mode: Plot mode.

    Returns:
        Tuple of (show, save).

    Raises:
        ValueError: If mode is not a valid PlotMode.
    """
    try:
        return SHOW_CONV[mode], SAVE_CONV[mode]
    except KeyError as exc:
        raise ValueError(f"Invalid plot mode: {mode!r}") from exc
