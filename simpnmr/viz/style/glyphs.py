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

from simpnmr.app.params.plot_cfg import PlotProfile


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
    aux_lw: float
    fit_lw: float
    elinewidth: float
    capsize: float
    band_alpha: float
    band_lw: float
    series_alpha_muted: float


SCALES: dict[PlotProfile, GlyphScale] = {
    "paper": GlyphScale(
        marker="s",
        ms=5.5,
        mec="none",
        mew=0.0,
        line_lw=1.2,
        aux_lw=1.5,
        fit_lw=1.6,
        elinewidth=1.2,
        capsize=2.0,
        band_alpha=0.15,
        band_lw=0.0,
        series_alpha_muted=0.65,
    ),
    "poster": GlyphScale(
        marker="s",
        ms=7.0,
        mec="none",
        mew=0.0,
        line_lw=1.8,
        aux_lw=2.1,
        fit_lw=2.3,
        elinewidth=1.9,
        capsize=3.2,
        band_alpha=0.15,
        band_lw=0.0,
        series_alpha_muted=0.65,
    ),
}


def get_glyphs(profile: PlotProfile = "paper") -> GlyphScale:
    """Return the glyph scale for a given plotting profile.

    Args:
        profile: Plotting profile name ("paper" or "poster").

    Returns:
        GlyphScale for the requested profile.

    Raises:
        KeyError: If an unknown profile is provided.
    """

    return SCALES[profile]
