# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Correlation-time fit diagnostic plots."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import to_rgba

from simpnmr.viz.layout.canvas import create_canvas, create_header_plot_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.label import resolve_label_layout
from simpnmr.viz.layout.table import render_compact_table
from simpnmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)


def plot_corr_time_scatter(
    *,
    theory_r1: np.ndarray,
    exp_r1: np.ndarray,
    chem_labels: list[str],
    rsquared: float,
    fix_param: str | None = None,
    tau_R_fit: float | None = None,
    tau_E_fit: float | None = None,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "experimental_vs_fitted_R1.pdf",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot experimental versus fitted ``R1`` values.

    This helper renders a scatter plot of fitted theoretical ``R1`` values
    against experimental ``R1`` values and overlays an ``x = y`` reference
    line. Optional fit diagnostics are shown in the plot annotation area.

    Args:
        theory_r1: Fitted theoretical ``R1`` values.
        exp_r1: Experimental ``R1`` values.
        chem_labels: Chemical labels corresponding to the plotted points.
        rsquared: Coefficient of determination for the fit.
        fix_param: Optional fit mode. Supported values are ``"tau_r"``,
            ``"tau_e"``, ``"none"``, ``""``, or ``None``.
        tau_R_fit: Fitted rotational correlation time.
        tau_E_fit: Fitted electronic correlation time.
        spec: Optional plot specification used for styling.
        save: Whether to save the figure.
        show: Whether to display the figure interactively.
        save_name: Output filename for the figure.
        verbose: Whether to emit an info log when the figure is saved.

    Returns:
        Tuple of ``(figure, axes)`` for the rendered scatter plot.
    """

    glyphs = spec.glyphs
    palette = spec.palette

    fig, summary_ax, ax = create_header_plot_canvas(
        spec.profile,
        variant="vertical",
        layout="constrained",
        header_ratio=0.55,
        plot_ratio=3.05,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    scale = spec.skin_axes(ax)
    ax.minorticks_on()
    ax.grid(True, which="major", color=palette.grid, linewidth=0.3, alpha=0.8)
    ax.set_axisbelow(True)
    marker_size = glyphs.ms
    annotation_size = scale.annotation
    axis_label_size = scale.axis_label
    fit_lines = [f"R²: {rsquared:.4f}"]

    model_lines = ["—"]
    if fix_param == "tau_r" and tau_E_fit is not None:
        model_lines = [f"τE: {tau_E_fit:.3e} s"]
    elif fix_param == "tau_e" and tau_R_fit is not None:
        model_lines = [f"τR: {tau_R_fit:.3e} s"]
    elif fix_param in {None, "", "none"}:
        tau_lines: list[str] = []
        if tau_R_fit is not None:
            tau_lines.append(f"τR: {tau_R_fit:.3e} s")
        if tau_E_fit is not None:
            tau_lines.append(f"τE: {tau_E_fit:.3e} s")
        if tau_lines:
            model_lines = tau_lines

    blocks = [
        ("Fit Statistics", fit_lines),
        ("Corr. Time", model_lines),
    ]

    render_compact_table(
        summary_ax,
        blocks,
        spec,
        bbox=[0.0, 0.0, 1.0, 1.0],
        cell_align="center",
        remove_outer_frame=True,
    )

    scatter_color = palette.primary

    ax.scatter(
        theory_r1,
        exp_r1,
        marker="o",
        facecolors=to_rgba(scatter_color, alpha=0.55),
        edgecolors=scatter_color,
        linewidths=0.8,
        s=(marker_size**2),
    )

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    shared_min = min(x_limits[0], y_limits[0])
    shared_max = max(x_limits[1], y_limits[1])
    ax.set_xlim(shared_min, shared_max)
    ax.set_ylim(shared_min, shared_max)
    ax.set_aspect("equal", adjustable="box")

    diag_line = ax.plot(
        [shared_min, shared_max],
        [shared_min, shared_max],
        linestyle="-",
        color="black",
        lw=1.0,
    )[0]

    label_entries = [
        (label, float(theory_val), float(exp_val))
        for label, theory_val, exp_val in zip(chem_labels, theory_r1, exp_r1)
    ]
    resolve_label_layout(
        ax,
        label_entries,
        fontsize=annotation_size,
        marker_size=marker_size,
        diag_line=diag_line,
    )

    ax.set_xlabel("Theoretical $R_1$ (s$^{-1}$)", fontsize=axis_label_size)
    ax.set_ylabel("Experimental $R_1$ (s$^{-1}$)", fontsize=axis_label_size)

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if save and verbose:
        logger.info("Saved correlation-time scatter plot to %s", save_name)
    return fig, ax


def plot_corr_time_contrib(
    *,
    theory_r1: np.ndarray,
    theory_r1_dipolar: np.ndarray | None = None,
    theory_r1_contact: np.ndarray | None = None,
    theory_r1_curie: np.ndarray | None = None,
    exp_r1: np.ndarray | None = None,
    chem_labels: list[str],
    spec: PlotSpec,
    ylabel: str = r"$R_1$ (s$^{-1}$)",
    save: bool = True,
    show: bool = True,
    save_name: str = "r1_fit_contributions.pdf",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot decomposed fitted ``R1`` contributions by chemical label.

    This helper renders a component-style comparison plot for fitted ``R1``
    values, analogous to the shift-contribution visualisation used elsewhere in
    the codebase. Experimental and total fitted ``R1`` values are shown as
    markers, while individual fitted ``R1`` contributions are shown as separate
    component series by chemical label.

    Args:
        theory_r1: Fitted theoretical ``R1`` values.
        theory_r1_dipolar: Optional fitted dipolar contribution to ``R1``.
        theory_r1_contact: Optional fitted contact contribution to ``R1``.
        theory_r1_curie: Optional fitted Curie contribution to ``R1``.
        exp_r1: Experimental ``R1`` values.
        chem_labels: Chemical labels corresponding to the plotted points.
        spec: Optional plot specification used for styling.
        save: Whether to save the figure.
        show: Whether to display the figure interactively.
        save_name: Output filename for the figure.
        verbose: Whether to emit an info log when the figure is saved.

    Returns:
        Tuple of ``(figure, axes)`` for the rendered contribution plot.
    """

    glyphs = spec.glyphs
    palette = spec.palette

    xvals = np.arange(len(chem_labels), dtype=float)

    order_idx = np.argsort(exp_r1 if exp_r1 is not None else theory_r1)[::-1]
    chem_labels_ordered = [chem_labels[idx] for idx in order_idx]
    theory_r1_ordered = theory_r1[order_idx]
    exp_r1_ordered = exp_r1[order_idx] if exp_r1 is not None else None

    component_series: list[tuple[str, np.ndarray, str]] = []

    if theory_r1_dipolar is not None:
        component_series.append(
            ("Dipolar", theory_r1_dipolar[order_idx], palette.highlight)
        )
    if theory_r1_contact is not None:
        component_series.append(
            ("Contact", theory_r1_contact[order_idx], palette.secondary)
        )
    if theory_r1_curie is not None:
        component_series.append(
            ("Curie", theory_r1_curie[order_idx], palette.auxiliary)
        )

    width = 1 / (len(component_series) + 1) if component_series else 0.8

    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        layout="constrained",
    )

    scale = spec.skin_axes(ax)
    widthscaler = 1

    ax.plot(
        (xvals + 0.5),
        theory_r1_ordered,
        label="Total",
        color=palette.primary,
        lw=0,
        marker="x",
        markersize=(glyphs.ms if glyphs is not None else 7),
        zorder=4,
    )

    for label, values, color in component_series:
        ax.bar(
            (xvals + width * widthscaler),
            values,
            width,
            label=label,
            color=color,
            zorder=2,
        )
        widthscaler += 1

    y_arrays = [theory_r1_ordered]
    if exp_r1_ordered is not None:
        y_arrays.append(exp_r1_ordered)
    y_arrays.extend(values for _, values, _ in component_series)
    y_concat = np.concatenate(y_arrays)
    y_min = float(np.min(y_concat))
    y_max = float(np.max(y_concat))
    y_span = y_max - y_min
    y_pad = 0.1 * y_span if y_span > 0 else 0.1 * max(abs(y_max), 1.0)

    if exp_r1_ordered is not None:
        ax.plot(
            (xvals + 0.5),
            exp_r1_ordered,
            label="Exp.",
            color=palette.primary,
            lw=0,
            marker="o",
            fillstyle="none",
            markersize=(glyphs.ms if glyphs is not None else 7),
            zorder=5,
        )

    ax.hlines(
        0.0,
        0,
        len(chem_labels_ordered),
        color=palette.primary,
        lw=(glyphs.line_lw if glyphs is not None else 0.5),
    )
    ax.grid(axis="x", ls="--", which="minor", color=palette.grid)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim([-0.5, xvals[-1] + 1.5])

    ax.set_xticks(xvals + 0.5)
    ax.set_xticklabels(chem_labels_ordered, rotation=90)
    ax.tick_params(axis="x", labelsize=scale.axis_label)

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_tick_params("major", length=0)

    ax.legend(loc="best")

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Correlation-time contribution plot saved to %s", save_name)

    return fig, ax
