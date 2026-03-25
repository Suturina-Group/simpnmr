# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Grouped scatter plot of experimental shift vs linewidth or R1.

Signals are grouped into subplots by the number of equivalent nuclei sharing
the same chemical label (group size). All markers are uniform in size. An
optional overlay shows values predicted by the r^-6 distance model.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)

_MARKER_SIZE = 60  # uniform scatter marker area (points²)

_Y_LABELS = {
    "width": "Linewidth (ppm)",
    "r1": r"$R_1$ (s$^{-1}$)",
}


def plot_shift_width_bubble(
    experiment: Experiment,
    molecule: Molecule,
    spec: PlotSpec,
    observable: str = "width",
    fit_result: dict | None = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_width_bubble",
    verbose: bool = True,
    window_title: str = "Shift vs Linewidth",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Grouped scatter plot of shift (x) vs linewidth or R1 (y).

    Signals are split into subplots by the group size of their chemical
    label (number of equivalent nuclei in the molecule). All markers are
    drawn at the same size. If ``fit_result`` is supplied, predicted values
    from the r^-6 model are overlaid as open circles with dashed connectors.

    Args:
        experiment: Experiment supplying shift, width, r1, area, and
            assignment for each signal.
        molecule: Molecule used to determine the group size for each
            chemical label.
        spec: Plot style specification.
        observable: ``"width"`` or ``"r1"`` — selects the y-axis quantity.
        fit_result: Optional r^-6 fit result dict (from
            :func:`~simpnmr.core.fitting.r6_fit.fit_r6`). When provided,
            predicted values are shown as open circles.
        save: If ``True``, saves the figure.
        show: If ``True``, displays the figure.
        save_name: Output file base name (extension appended automatically).
        verbose: If ``True``, logs the saved file path.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, axes)`` where ``axes`` is the list of subplots.
    """
    # --- Build group-size lookup from molecule nuclei ---
    cl_to_size: dict[str, int] = {}
    for nuc in molecule.nuclei:
        cl_to_size[nuc.chem_label] = cl_to_size.get(nuc.chem_label, 0) + 1

    # --- Collect per-signal data ---
    label_to_pred = (
        dict(zip(fit_result["labels"], fit_result["pred"]))
        if fit_result is not None
        else {}
    )

    records = []
    for sig in experiment.signals:
        y = sig.r1 if observable == "r1" else sig.width
        if y is None or (observable == "r1" and np.isnan(float(y))):
            continue
        cl = sig.assignment
        records.append(
            {
                "shift": sig.shift,
                "y": float(y),
                "label": cl,
                "group_size": cl_to_size.get(cl, 0),
                "pred": label_to_pred.get(cl),
            }
        )

    if not records:
        logger.warning("plot_shift_width_bubble: no valid data for %s", observable)
        return None, []

    # --- Group by group size, sorted ascending ---
    group_sizes = sorted({r["group_size"] for r in records})
    n_cols = len(group_sizes)

    palette = spec.palette
    fontsize = 7

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4.0 * n_cols, 4.0),
        squeeze=False,
    )
    axes = axes[0]  # flatten to 1-D list

    if hasattr(fig, "canvas") and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    fig.patch.set_facecolor(palette.annotation_bg)

    y_label = _Y_LABELS.get(observable, observable)

    for ax, gs in zip(axes, group_sizes):
        spec.skin_axes(ax)
        ax.set_facecolor(palette.annotation_bg)
        ax.grid(True, which="major", color=palette.grid, linewidth=1.0)
        ax.grid(True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

        subset = [r for r in records if r["group_size"] == gs]
        shifts = [r["shift"] for r in subset]
        ys = [r["y"] for r in subset]
        labels = [r["label"] for r in subset]
        preds = [r["pred"] for r in subset]

        # Experimental points
        ax.scatter(
            shifts,
            ys,
            s=_MARKER_SIZE,
            color=palette.primary,
            alpha=0.7,
            edgecolors=palette.primary,
            linewidths=0.8,
            zorder=3,
            label="Experiment",
        )

        # Labels
        for x, y, lbl in zip(shifts, ys, labels):
            ax.annotate(
                lbl,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=fontsize,
                color=palette.primary,
            )

        # Predicted overlay
        has_pred = [p is not None for p in preds]
        if any(has_pred):
            pred_shifts = [x for x, hp in zip(shifts, has_pred) if hp]
            pred_ys = [p for p in preds if p is not None]
            exp_ys = [y for y, hp in zip(ys, has_pred) if hp]

            ax.scatter(
                pred_shifts,
                pred_ys,
                s=_MARKER_SIZE,
                facecolors="none",
                edgecolors=palette.primary,
                linewidths=1.2,
                zorder=4,
                label=r"Predicted ($r^{-6}$)",
            )

            for xs, ye, yp in zip(pred_shifts, exp_ys, pred_ys):
                ax.plot(
                    [xs, xs],
                    [ye, yp],
                    color=palette.primary,
                    lw=0.5,
                    ls="--",
                    alpha=0.4,
                    zorder=2,
                )

        ax.set_title(f"N = {gs}", fontsize=8)
        ax.set_xlabel("Shift (ppm)")
        ax.invert_xaxis()

        # Only label y-axis on the leftmost subplot
        if ax is axes[0]:
            ax.set_ylabel(y_label)
            ax.legend(fontsize=fontsize, framealpha=0.8)

    fig.suptitle(window_title, fontsize=9)
    fig.tight_layout()

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save and verbose:
        logger.info(
            "Shift–%s grouped bubble plot saved to %s", observable, f"{save_name}.pdf"
        )

    return fig, axes
