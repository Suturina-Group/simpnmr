# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Glyph (markers/lines/errorbars/fills) scales and helpers for Matplotlib plots.

This module centralises the numeric "ink" parameters that determine how data is
rendered (marker sizes, line widths, errorbar thickness, band alpha, etc.).

Plot functions should avoid hardcoding these numbers and instead request a
small set of size classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SizeClass = Literal["small", "standard", "large"]


@dataclass(frozen=True, slots=True)
class GlyphScale:
    """Numeric rendering parameters for a figure size class.

    Attributes:
        marker: Default marker symbol for point series.
        ms: Marker size.
        mec: Marker edge colour.
        mew: Marker edge width.
        line_lw: Default line width for generic lines.
        fit_lw: Line width for model/fit curves.
        elinewidth: Errorbar line width.
        capsize: Errorbar cap size.
        band_alpha: Alpha for uncertainty bands.
        band_lw: Line width for band edges.
        series_alpha_muted: Alpha for muted/secondary series.
    """

    marker: str
    ms: float
    mec: str
    mew: float
    line_lw: float
    fit_lw: float
    elinewidth: float
    capsize: float
    band_alpha: float
    band_lw: float
    series_alpha_muted: float


SCALES: dict[SizeClass, GlyphScale] = {
    "standard": GlyphScale(
        marker="s",
        ms=6.5,
        mec="none",
        mew=0.0,
        line_lw=1.5,
        fit_lw=2.0,
        elinewidth=1.5,
        capsize=2.5,
        band_alpha=0.15,
        band_lw=0.0,
        series_alpha_muted=0.65,
    ),
    "small": GlyphScale(
        marker="s",
        ms=5.8,
        mec="none",
        mew=0.0,
        line_lw=1.3,
        fit_lw=1.8,
        elinewidth=1.3,
        capsize=2.0,
        band_alpha=0.15,
        band_lw=0.0,
        series_alpha_muted=0.65,
    ),
    "large": GlyphScale(
        marker="s",
        ms=7.2,
        mec="none",
        mew=0.0,
        line_lw=1.8,
        fit_lw=2.2,
        elinewidth=1.8,
        capsize=3.0,
        band_alpha=0.15,
        band_lw=0.0,
        series_alpha_muted=0.65,
    ),
}


def get_glyphs(size: SizeClass = "standard") -> GlyphScale:
    """Return the glyph scale for a given size class.

    Args:
        size: Size class name.

    Returns:
        GlyphScale for the requested size class.

    Raises:
        KeyError: If an unknown size class is provided.
    """

    return SCALES[size]
