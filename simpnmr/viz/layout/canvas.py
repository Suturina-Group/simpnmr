# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Canvas creation helpers for visualization layouts."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simpnmr.app.params.plot_cfg import PlotProfile
from simpnmr.viz.layout.figure import FigureVariant, get_figsize


def create_canvas(
    profile: PlotProfile,
    *,
    variant: FigureVariant = "standard",
    window_title: str | None = None,
    layout: str = "constrained",
    width_scale: float = 1.0,
    height_scale: float = 1.0,
) -> tuple[Figure, Axes]:
    """Create a single-axis Matplotlib canvas from canonical layout tokens.

    Args:
        profile: Plotting profile that selects the publication context.
        variant: Canonical figure geometry variant within the selected profile.
        window_title: Optional Matplotlib figure identifier.
        layout: Matplotlib layout engine passed to ``plt.subplots``.
        width_scale: Multiplier applied to the canonical figure width.
            Defaults to 1.0.
        height_scale: Multiplier applied to the canonical figure height.
            Defaults to 1.0.

    Returns:
        Tuple of ``(fig, ax)`` for a single-axis plotting canvas.
    """
    _w, _h = get_figsize(profile, variant)
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(_w * width_scale, _h * height_scale),
        num=window_title,
        layout=layout,
    )
    return fig, ax


def create_stacked_canvas(
    profile: PlotProfile,
    *,
    nrows: int,
    variant: FigureVariant = "standard",
    window_title: str | None = None,
    layout: str = "constrained",
    sharex: bool = False,
    sharey: bool = False,
    hspace: float = 0.05,
    height_ratios: list[float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Create a vertically stacked canvas from canonical layout tokens.

    Args:
        profile: Plotting profile that selects the publication context.
        nrows: Number of vertically stacked subplot rows.
        variant: Canonical figure geometry variant within the selected profile.
        window_title: Optional Matplotlib figure identifier.
        layout: Matplotlib layout engine passed to ``plt.subplots``.
        sharex: Whether stacked axes should share the x-axis.
        sharey: Whether stacked axes should share the y-axis.
        hspace: Vertical spacing between rows as a fraction of the average
            subplot height (passed to ``gridspec_kw``).
        height_ratios: Relative heights of each row. When supplied, the list
            length must equal ``nrows``.

    Returns:
        Tuple of ``(fig, axes)`` for a vertically stacked plotting canvas.

    Raises:
        ValueError: If ``nrows`` is smaller than 1.
    """
    if nrows < 1:
        raise ValueError("nrows must be at least 1 for a stacked canvas.")

    gridspec_kw: dict = {"hspace": hspace}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=get_figsize(profile, variant),
        num=window_title,
        layout=layout,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
    )
    return fig, axes


def create_header_plot_canvas(
    profile: PlotProfile,
    *,
    variant: FigureVariant = "vertical",
    window_title: str | None = None,
    layout: str = "constrained",
    header_ratio: float = 1.35,
    plot_ratio: float = 5.05,
    hspace: float = 0.02,
) -> tuple[Figure, Axes, Axes]:
    """Create a two-row canvas with a header axis above a plot axis.

    Args:
        profile: Plotting profile that selects the publication context.
        variant: Canonical figure geometry variant within the selected profile.
        window_title: Optional Matplotlib figure identifier.
        layout: Matplotlib layout engine passed to ``plt.figure``.
        header_ratio: Relative height ratio of the header axis.
        plot_ratio: Relative height ratio of the plot axis.
        hspace: Vertical spacing between the two rows in the grid spec.

    Returns:
        Tuple of ``(fig, header_ax, ax)`` for a header-plus-plot canvas.
    """
    fig = plt.figure(
        figsize=get_figsize(profile, variant),
        num=window_title,
        layout=layout,
    )
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[header_ratio, plot_ratio],
        hspace=hspace,
    )
    header_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])
    header_ax.axis("off")
    return fig, header_ax, ax
