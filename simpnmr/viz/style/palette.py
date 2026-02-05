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
    primary: str  # main theory / primary series
    secondary: str  # secondary theory / comparison
    experimental: str  # experimental data
    highlight: str  # emphasis / selected series
    muted: str  # de-emphasised / background series

    # UI / reference elements
    reference: str  # zero-lines, diagonals
    grid: str  # grid lines
    legend_edge: str  # legend frame edge
    annotation_bg: str  # annotation / caption background


PALETTES: dict[PlotProfile, Palette] = {
    "paper": Palette(
        primary="#1F4E79",  # deep navy blue (theory / main series)
        secondary="#1E8449",  # muted green (secondary theory / comparison)
        experimental="#111111",  # charcoal black (measured data)
        highlight="#C0392B",  # vermillion red (emphasis / selected component)
        muted="#808080",  # neutral grey (de-emphasised background data)
        reference="#111111",  # charcoal black (zero lines, diagonals)
        grid="#D9D9D9",  # light grey (non-dominant grid)
        legend_edge="#111111",  # charcoal black (legend frame)
        annotation_bg="#FFFFFF",  # white (annotation background)
    ),
    "poster": Palette(
        primary="#1F4E79",  # deep navy blue (theory / main series)
        secondary="#1E8449",  # muted green (secondary theory / comparison)
        experimental="#111111",  # charcoal black (measured data)
        highlight="#C0392B",  # vermillion red (emphasis / selected component)
        muted="#6E6E6E",  # neutral grey (de-emphasised background data)
        reference="#111111",  # charcoal black (zero lines, diagonals)
        grid="#CCCCCC",  # light grey (non-dominant grid)
        legend_edge="#111111",  # charcoal black (legend frame)
        annotation_bg="#FFFFFF",  # white (annotation background)
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
        total="#111111",  # total pNMR shift (charcoal black)
        fc="#1F4E79",  # Fermi contact contribution (deep navy blue)
        pc="#C0392B",  # pseudocontact contribution (vermillion red)
        dia="#1E8449",  # diamagnetic contribution (muted green)
    ),
    "poster": ShiftColours(
        total="#111111",  # total pNMR shift (charcoal black)
        fc="#1F4E79",  # Fermi contact contribution (deep navy blue)
        pc="#C0392B",  # pseudocontact contribution (vermillion red)
        dia="#1E8449",  # diamagnetic contribution (muted green)
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
