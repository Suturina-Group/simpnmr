# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define a safe color palette for plots.

Provides a list of Matplotlib-compatible color specifications.
"""

from __future__ import annotations

from dataclasses import dataclass

from simpnmr.app.params.plot_cfg import PlotProfile


@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic colour roles for plots.

    Colours are defined by role, not by index. Plotting code must not
    access colours positionally.
    """

    # Core semantic roles
    primary: str
    secondary: str
    highlight: str
    auxiliary: str

    # UI / reference elements
    reference: str
    grid: str
    annotation_bg: str


PALETTES: dict[PlotProfile, Palette] = {
    "paper": Palette(
        primary="#111111",  # black
        secondary="#28A228",  # green
        highlight="#F22613",  # red
        auxiliary="#3455DB",  # dark blue
        reference="#AA2E00",  # dark orange
        grid="#D9D9D9",  # light grey
        annotation_bg="#FFFFFF",  # white
    ),
    "poster": Palette(
        primary="#111111",  # black
        secondary="#28A228",  # green
        highlight="#F22613",  # red
        auxiliary="#3455DB",  # dark blue
        reference="#AA2E00",  # dark orange
        grid="#D9D9D9",  # light grey
        annotation_bg="#FFFFFF",  # white
    ),
}


@dataclass(frozen=True, slots=True)
class ShiftColours:
    """Domain-specific colours for shift component plots.

    These colours encode physical meaning for pNMR shift decomposition.
    They are intentionally kept separate from the generic UI palette.
    """

    total: str
    fc: str
    pc: str
    dia: str


SHIFT_COLOURS: dict[PlotProfile, ShiftColours] = {
    "paper": ShiftColours(
        total="#111111",  # total pNMR shift (black)
        fc="#3455DB",  # Fermi contact contribution (dark blue)
        pc="#F22613",  # pseudocontact contribution (red)
        dia="#28A228",  # diamagnetic contribution (green)
    ),
    "poster": ShiftColours(
        total="#111111",  # total pNMR shift (black)
        fc="#3455DB",  # Fermi contact contribution (dark blue)
        pc="#F22613",  # pseudocontact contribution (red)
        dia="#28A228",  # diamagnetic contribution (green)
    ),
}


def get_palette(profile: PlotProfile) -> Palette:
    """Return the colour palette for a plotting profile.

    Args:
        profile: Plot profile (e.g. ``"paper"`` or ``"poster"``).

    Returns:
        Palette for the requested profile.
    """

    try:
        return PALETTES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown plot profile: {profile!r}") from exc


def get_shift_colours(profile: PlotProfile) -> ShiftColours:
    """Return the shift-component colours for a plotting profile.

    Args:
        profile: Plot profile (e.g. ``"paper"`` or ``"poster"``).

    Returns:
        ShiftColours for the requested profile.
    """

    try:
        return SHIFT_COLOURS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown plot profile: {profile!r}") from exc
