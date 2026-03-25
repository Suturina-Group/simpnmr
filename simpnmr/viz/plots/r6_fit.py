# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot r^-6 distance-model fit results."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)

_OBS_LABELS = {
    "r1": r"$R_1$ (s$^{-1}$)",
    "width": "Linewidth (ppm)",
}


def plot_r6_fit(
    fit_result: dict,
    observable: str = "r1",
    spec: PlotSpec = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "r6_fit",
    verbose: bool = True,
    window_title: str = "r⁻⁶ Fit",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot observed vs fitted values from a :func:`fit_r6` result.

    Draws the data points as a scatter plot against ``r^-6`` and overlays
    the smooth fitted line ``p1 * r^-6 + p2``.

    Args:
        fit_result: Dict returned by :func:`~simpnmr.core.fitting.r6_fit.fit_r6`.
        observable: ``"r1"`` or ``"width"`` — used for axis label only.
        spec: Plot style specification.
        save: If ``True``, saves the figure.
        show: If ``True``, displays the figure.
        save_name: Output file base name (extension added automatically).
        verbose: If ``True``, logs the saved file path.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, ax)``.
    """
    r6_inv = 1.0 / fit_result["r_eff"] ** 6
    obs = fit_result["obs"]
    labels = fit_result["labels"]
    p1 = fit_result["p1"]
    p2 = fit_result["p2"]
    p1_err = fit_result["p1_err"]
    p2_err = fit_result["p2_err"]
    rmse = fit_result["rmse"]

    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )

    palette = spec.palette
    glyphs = spec.glyphs
    spec.skin_axes(ax)

    fig.patch.set_facecolor(palette.annotation_bg)
    ax.set_facecolor(palette.annotation_bg)

    ax.grid(True, which="major", color=palette.grid, linewidth=1.0)
    ax.grid(True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    # Scatter: data points
    ax.plot(
        r6_inv,
        obs,
        lw=0,
        marker="o",
        color=palette.primary,
        markersize=glyphs.ms,
        markerfacecolor=(0, 0, 0, 0.55),
        markeredgecolor=palette.primary,
        markeredgewidth=0.8,
        zorder=3,
    )

    # Label each point
    for x, y, lbl in zip(r6_inv, obs, labels):
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=glyphs.annotation_size
            if hasattr(glyphs, "annotation_size")
            else 7,
            color=palette.primary,
        )

    # Smooth fitted curve
    x_smooth = np.linspace(r6_inv.min() * 0.9, r6_inv.max() * 1.1, 300)
    y_smooth = p1 * x_smooth + p2
    ax.plot(
        x_smooth,
        y_smooth,
        color=palette.primary,
        lw=1.2,
        label="Fit",
        zorder=2,
    )

    obs_label = _OBS_LABELS.get(observable, observable)
    ax.set_xlabel(r"$r^{-6}$ (Å$^{-6}$)")
    ax.set_ylabel(obs_label)

    # Annotation box
    ann = (
        f"$p_1$ = {p1:.3g} ± {p1_err:.2g}\n"
        f"$p_2$ = {p2:.3g} ± {p2_err:.2g}\n"
        f"RMSE = {rmse:.3g}"
    )
    ax.text(
        0.97,
        0.97,
        ann,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=palette.primary,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=palette.annotation_bg,
            edgecolor=palette.grid,
            alpha=0.8,
        ),
    )

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save and verbose:
        logger.info("r^-6 fit plot saved to %s", f"{save_name}.pdf")

    return fig, ax
