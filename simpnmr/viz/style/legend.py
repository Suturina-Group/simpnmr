# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Legend style scales and helpers for Matplotlib plots.

This module centralises non-typographic legend appearance (handle geometry,
spacing, and frame styling) to keep all figures visually consistent across the
library.

Font sizes for legends are owned by `viz/style/typography.py` via global
Matplotlib rcParams (e.g. `legend.fontsize`). This module therefore does not
encode any `fontsize` values.
"""

from __future__ import annotations

from dataclasses import dataclass

from simpnmr.app.params.plot_cfg import PlotProfile


@dataclass(frozen=True, slots=True)
class LegendStyle:
    """Legend appearance parameters independent of legend layout.

    This class defines legend styling independently of layout decisions
    (location, number of columns, anchoring). It is purely a style contract
    and is resolved via PlotProfile (e.g. paper vs poster).

    Attributes:
        handlelength: Length of legend handles in font-size units.
        handletextpad: Spacing between the handle and the label text.
        columnspacing: Spacing between legend columns.
        borderpad: Padding between the legend content and its border.
        labelspacing: Vertical spacing between legend entries.
        frameon: Whether to draw a legend frame.
        fancybox: Whether to draw a rounded legend frame.
        framealpha: Legend frame transparency.
        markerscale: Relative scaling of legend markers. If None, Matplotlib
            default is used.
    """

    handlelength: float
    handletextpad: float
    columnspacing: float
    borderpad: float
    labelspacing: float
    frameon: bool
    fancybox: bool
    framealpha: float
    markerscale: float | None = None


def get_legend_style(profile: PlotProfile) -> LegendStyle:
    """Return the legend style for a given plotting profile."""

    if profile == "paper":
        return LegendStyle(
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.0,
            borderpad=0.3,
            labelspacing=0.4,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            markerscale=1.0,
        )

    if profile == "poster":
        return LegendStyle(
            handlelength=2.4,
            handletextpad=0.9,
            columnspacing=1.6,
            borderpad=0.5,
            labelspacing=0.8,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            markerscale=1.4,
        )

    raise ValueError(f"Unsupported PlotProfile: {profile}")


def apply_global_legend_style(profile: PlotProfile) -> None:
    """Apply global Matplotlib legend styling for the given plotting profile.

    This function sets rcParams controlling legend geometry and frame
    appearance. Font sizes are intentionally not handled here and are
    owned by typography via rcParams (e.g. ``legend.fontsize``).

    It is expected to be called once per pipeline run via
    ``viz.style.theme.apply_profile``.
    """
    import matplotlib as mpl

    style = get_legend_style(profile)

    rcparams: dict[str, object] = {
        "legend.handlelength": style.handlelength,
        "legend.handletextpad": style.handletextpad,
        "legend.columnspacing": style.columnspacing,
        "legend.borderpad": style.borderpad,
        "legend.labelspacing": style.labelspacing,
        "legend.frameon": style.frameon,
        "legend.fancybox": style.fancybox,
        "legend.framealpha": style.framealpha,
        "legend.edgecolor": "black",
    }

    if style.markerscale is not None:
        rcparams["legend.markerscale"] = style.markerscale

    mpl.rcParams.update(rcparams)
