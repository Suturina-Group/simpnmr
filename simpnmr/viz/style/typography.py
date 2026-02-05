# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Typography scales and helpers for Matplotlib plots.

This module centralises font-size decisions to keep all figures visually
consistent across the library. Plot functions should avoid hardcoding
`fontsize=...` and instead rely on a small set of size classes.

The design intentionally provides only a few size classes to prevent visual
drift across plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.axes

from simpnmr.app.params.plot_cfg import PlotProfile


@dataclass(frozen=True, slots=True)
class TypographyScale:
    """Font sizes (in points) for a figure size class.

    Attributes:
        base: Base font size.
        axis_label: Axis label font size.
        tick_label: Tick label font size.
        legend: Legend font size.
        title: Title font size.
        annotation: Annotation/caption font size.
    """

    base: int
    axis_label: int
    tick_label: int
    legend: int
    title: int
    annotation: int


SCALES: dict[PlotProfile, TypographyScale] = {
    # Default for most figures, designed for publication-style PDFs.
    "paper": TypographyScale(
        base=10,
        axis_label=11,
        tick_label=9,
        legend=9,
        title=12,
        annotation=9,
    ),
    # For wide figures, posters, and slides.
    "poster": TypographyScale(
        base=11,
        axis_label=12,
        tick_label=10,
        legend=10,
        title=13,
        annotation=10,
    ),
}


def get_scale(size: PlotProfile = "paper") -> TypographyScale:
    """Return the typography scale for a given size class.

    Args:
        size: Size class name.

    Returns:
        TypographyScale for the requested size class.

    Raises:
        KeyError: If an unknown size class is provided.
    """

    return SCALES[size]


def apply_typography(
    ax: matplotlib.axes.Axes, size: PlotProfile = "paper"
) -> TypographyScale:
    """Apply tick/axis-label typography defaults to an Axes.

    This function is intentionally conservative: it sets tick label sizes and
    axis label sizes, but does not overwrite titles or existing text content.

    Args:
        ax: Axes to configure.
        size: Size class name.

    Returns:
        The TypographyScale that was applied.
    """

    scale = get_scale(size)

    # Tick labels.
    ax.tick_params(axis="both", which="major", labelsize=scale.tick_label)
    ax.tick_params(axis="both", which="minor", labelsize=scale.tick_label)

    # Axis labels.
    ax.xaxis.label.set_size(scale.axis_label)
    ax.yaxis.label.set_size(scale.axis_label)

    return scale


def apply_typography_many(
    axes: list[matplotlib.axes.Axes],
    size: PlotProfile = "paper",
) -> TypographyScale:
    """Apply typography defaults to multiple Axes.

    Args:
        axes: List of axes to configure.
        size: Size class name.

    Returns:
        The TypographyScale that was applied.
    """

    scale = get_scale(size)
    for ax in axes:
        apply_typography(ax=ax, size=size)
    return scale


def apply_global_typography(profile: PlotProfile = "paper") -> TypographyScale:
    """Apply global Matplotlib typography defaults for a plotting profile.

    This function sets rcParams that are safe to apply globally and should be
    called once per pipeline run via theme.apply_profile().
    """

    scale = get_scale(profile)

    mpl.rcParams.update(
        {
            # Base font size
            "font.size": scale.base,
            # Axis titles and labels
            "axes.labelsize": scale.axis_label,
            "axes.titlesize": scale.title,
            # Tick labels
            "xtick.labelsize": scale.tick_label,
            "ytick.labelsize": scale.tick_label,
            # Legend
            "legend.fontsize": scale.legend,
            "legend.title_fontsize": scale.legend,
            # Math text (critical for NMR / physics plots)
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
        }
    )

    return scale
