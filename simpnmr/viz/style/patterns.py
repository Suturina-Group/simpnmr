# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Accessibility patterns for Matplotlib plots.

This module defines secondary visual encodings (e.g., hatch patterns) that can
be used in addition to colour to improve accessibility (e.g., colour vision
deficiency).

Patterns are intentionally defined by *semantic role* and keyed by the global
plotting profile and accessibility mode. Plotting code must consume patterns
through the PlotSpec contract (constructed by :mod:`simpnmr.viz.style.theme`).

At present, only hatch patterns for shift components are provided. Future
extensions can add linestyles/markers for line/scatter artists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from simpnmr.app.params.plot_cfg import PlotProfile

AccessibilityMode = Literal["default", "colorblind"]


@dataclass(frozen=True, slots=True)
class ShiftPatterns:
    """Secondary encodings for pNMR shift components.

    Hatch patterns are primarily intended for bar/patch artists (``ax.bar``,
    legend patches, etc.). For violin/line/scatter artists, consider adding a
    separate channel such as linestyle/marker in the plotting layer.

    Attributes:
        total_hatch: Hatch for total shift.
        fc_hatch: Hatch for Fermi contact contribution.
        pc_hatch: Hatch for pseudocontact contribution.
        dia_hatch: Hatch for diamagnetic contribution.
    """

    total_hatch: str
    fc_hatch: str
    pc_hatch: str
    dia_hatch: str


# Default hatch choices. Keep them sparse and visually distinct.
# - total: none (usually black points/lines)
# - fc: forward slashes
# - pc: cross hatch
# - dia: dotted
_SHIFT_PATTERNS: dict[AccessibilityMode, ShiftPatterns] = {
    "default": ShiftPatterns(
        total_hatch="",
        fc_hatch="",
        pc_hatch="",
        dia_hatch="",
    ),
    "colorblind": ShiftPatterns(
        total_hatch="",
        fc_hatch="///",
        pc_hatch="xxx",
        dia_hatch="...",
    ),
}


def get_shift_patterns(
    accessibility: AccessibilityMode = "default",
    profile: PlotProfile = "paper",
) -> ShiftPatterns:
    """Return shift-component patterns for the given accessibility mode.

    Args:
        accessibility: Accessibility mode.
        profile: Plot profile. Currently unused (patterns are profile-invariant),
            but kept for future expansion and API symmetry with other style
            subsystems.

    Returns:
        ShiftPatterns for the requested accessibility mode.

    Raises:
        ValueError: If an unknown accessibility mode is provided.
    """

    _ = profile  # reserved for future use

    try:
        return _SHIFT_PATTERNS[accessibility]
    except KeyError as exc:
        raise ValueError(f"Unknown accessibility mode: {accessibility!r}") from exc
