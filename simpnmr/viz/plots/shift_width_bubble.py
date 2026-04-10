# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Scatter plot of experimental shift vs linewidth or R1.

Two subplots are produced side by side:

* **Matched** (left): signals whose assignment exists in both the experiment
  and the molecule. Experimental points are area-scaled; predicted overlay
  uses a fixed marker size.
* **Unmatched** (right): signals present in only one dataset. Experimental
  unmatched markers are area-scaled; predicted unmatched markers are scaled
  by the molecule group size (number of equivalent nuclei).
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)

_MARKER_SIZE = 60   # reference marker area (points²)
_MAX_MARKER  = 300  # cap on scaled marker area

_Y_LABELS = {
    "width": "Linewidth (ppm)",
    "r1": r"$R_1$ (s$^{-1}$)",
}

_UNMATCHED_COLOR = "#e05c3a"


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
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel scatter: matched signals (left) and unmatched (right).

    Args:
        experiment: Experiment supplying shift, width, r1, area, assignment.
        molecule: Molecule used for predicted shifts and group sizes.
        spec: Plot style specification.
        observable: ``"width"`` or ``"r1"`` — selects the y-axis quantity.
        fit_result: Optional r^-6 fit result dict. When provided, predicted
            values are shown as open circles on the matched panel.
        save: If ``True``, saves the figure.
        show: If ``True``, displays the figure.
        save_name: Output file base name (extension appended automatically).
        verbose: If ``True``, logs the saved file path.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, axes)`` where ``axes`` is a length-2 ndarray.
    """
    # ------------------------------------------------------------------
    # Build per-label lookups from molecule
    # ------------------------------------------------------------------
    cl_to_size: dict[str, int] = {}
    cl_to_pred_shift: dict[str, float] = {}
    for nuc in molecule.nuclei:
        cl_to_size[nuc.chem_label] = (
            cl_to_size.get(nuc.chem_label, 0) + 1
        )
        prev = cl_to_pred_shift.get(nuc.chem_label)
        cl_to_pred_shift[nuc.chem_label] = (
            float(nuc.shift.avg)
            if prev is None
            else (prev + float(nuc.shift.avg)) / 2
        )

    # Predicted observable y via p1 * mean(1/r^6) + p2
    cl_to_pred_y: dict[str, float] = {}
    if (
        fit_result is not None
        and getattr(molecule, "paramagnetic_centre", None) is not None
    ):
        p1 = fit_result["p1"]
        p2 = fit_result["p2"]
        centre = np.asarray(molecule.paramagnetic_centre, dtype=float)
        cl_r6: dict[str, list] = {}
        for nuc in molecule.nuclei:
            r = float(np.linalg.norm(nuc.coord - centre))
            cl_r6.setdefault(nuc.chem_label, []).append(
                1.0 / max(r, 1e-6) ** 6
            )
        for cl, vals in cl_r6.items():
            cl_to_pred_y[cl] = p1 * float(np.mean(vals)) + p2

    label_to_pred = (
        dict(zip(fit_result["labels"], fit_result["pred"]))
        if fit_result is not None
        else {}
    )

    # ------------------------------------------------------------------
    # Partition experimental signals
    # ------------------------------------------------------------------
    matched: list[dict] = []
    unmatched_exp: list[dict] = []
    exp_assignments: set[str] = set()

    for sig in experiment.signals:
        cl = sig.assignment
        exp_assignments.add(cl)
        y = sig.r1 if observable == "r1" else sig.width
        if y is None or (observable == "r1" and np.isnan(float(y))):
            continue
        record = {
            "shift": sig.shift,
            "pred_shift": cl_to_pred_shift.get(cl, sig.shift),
            "y": float(y),
            "label": cl,
            "pred": label_to_pred.get(cl),
            "area": float(sig.area),
        }
        if cl in cl_to_size:
            matched.append(record)
        else:
            unmatched_exp.append(record)

    # Unmatched predicted: in molecule but not in experiment
    unmatched_pred: list[dict] = []
    for cl, pred_shift in cl_to_pred_shift.items():
        if cl in exp_assignments:
            continue
        pred_y = cl_to_pred_y.get(cl)
        if pred_y is None:
            continue
        unmatched_pred.append(
            {
                "pred_shift": pred_shift,
                "pred_y": pred_y,
                "label": cl,
                "group_size": cl_to_size[cl],
            }
        )

    if not matched and not unmatched_exp and not unmatched_pred:
        logger.warning(
            "plot_shift_width_bubble: no valid data for %s", observable
        )
        return None, None

    # ------------------------------------------------------------------
    # Scale marker sizes
    # ------------------------------------------------------------------
    _scale_by_area(matched)
    _scale_by_area(unmatched_exp)
    _scale_by_group(unmatched_pred)

    # ------------------------------------------------------------------
    # Figure with two subplots
    # ------------------------------------------------------------------
    palette = spec.palette
    fontsize = 7
    y_label = _Y_LABELS.get(observable, observable)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.0, 4.5),
        squeeze=False,
    )
    axes = axes[0]

    if hasattr(fig, "canvas") and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    fig.patch.set_facecolor(palette.annotation_bg)

    # ------------------------------------------------------------------
    # Left panel — matched
    # ------------------------------------------------------------------
    ax_m = axes[0]
    spec.skin_axes(ax_m)
    ax_m.set_facecolor(palette.annotation_bg)
    ax_m.grid(True, which="major", color=palette.grid, linewidth=1.0)
    ax_m.grid(
        True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8
    )
    ax_m.set_axisbelow(True)

    if matched:
        ax_m.scatter(
            [r["shift"] for r in matched],
            [r["y"] for r in matched],
            s=[r["marker_size"] for r in matched],
            color=palette.primary,
            alpha=0.7,
            edgecolors=palette.primary,
            linewidths=0.8,
            zorder=3,
            label="Experiment",
        )
        for r in matched:
            ax_m.annotate(
                r["label"], xy=(r["shift"], r["y"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=fontsize, color=palette.primary,
            )

    has_pred = [r for r in matched if r["pred"] is not None]
    if has_pred:
        ax_m.scatter(
            [r["pred_shift"] for r in has_pred],
            [r["pred"] for r in has_pred],
            s=_MARKER_SIZE,
            facecolors="none",
            edgecolors=palette.primary,
            linewidths=1.2,
            zorder=4,
            label=r"Predicted ($r^{-6}$)",
        )
        for r in has_pred:
            ax_m.plot(
                [r["shift"], r["pred_shift"]],
                [r["y"], r["pred"]],
                color=palette.primary,
                lw=0.5, ls="--", alpha=0.4, zorder=2,
            )

    ax_m.set_title("Matched", fontsize=8)
    ax_m.set_xlabel("Shift (ppm)")
    ax_m.set_ylabel(y_label)
    ax_m.invert_xaxis()
    ax_m.legend(fontsize=fontsize, framealpha=0.8)

    # ------------------------------------------------------------------
    # Right panel — unmatched
    # ------------------------------------------------------------------
    ax_u = axes[1]
    spec.skin_axes(ax_u)
    ax_u.set_facecolor(palette.annotation_bg)
    ax_u.grid(True, which="major", color=palette.grid, linewidth=1.0)
    ax_u.grid(
        True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8
    )
    ax_u.set_axisbelow(True)

    if unmatched_exp:
        ax_u.scatter(
            [r["shift"] for r in unmatched_exp],
            [r["y"] for r in unmatched_exp],
            s=[r["marker_size"] for r in unmatched_exp],
            color=_UNMATCHED_COLOR,
            alpha=0.7,
            edgecolors=_UNMATCHED_COLOR,
            linewidths=0.8,
            zorder=3,
            label="Exp. (unmatched)",
        )
        for r in unmatched_exp:
            ax_u.annotate(
                r["label"], xy=(r["shift"], r["y"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=fontsize, color=_UNMATCHED_COLOR,
            )

    if unmatched_pred:
        ax_u.scatter(
            [r["pred_shift"] for r in unmatched_pred],
            [r["pred_y"] for r in unmatched_pred],
            s=[r["marker_size"] for r in unmatched_pred],
            facecolors="none",
            edgecolors=_UNMATCHED_COLOR,
            linewidths=1.2,
            zorder=4,
            label="Predicted (unmatched)",
        )
        for r in unmatched_pred:
            ax_u.annotate(
                r["label"], xy=(r["pred_shift"], r["pred_y"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=fontsize, color=_UNMATCHED_COLOR,
            )

    ax_u.set_title("Unmatched", fontsize=8)
    ax_u.set_xlabel("Shift (ppm)")
    ax_u.invert_xaxis()
    ax_u.legend(fontsize=fontsize, framealpha=0.8)

    fig.suptitle(window_title, fontsize=9)
    fig.tight_layout()

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save and verbose:
        logger.info(
            "Shift–%s bubble plot saved to %s",
            observable, f"{save_name}.pdf",
        )

    return fig, axes


# ------------------------------------------------------------------
# Marker sizing helpers
# ------------------------------------------------------------------

def _scale_by_area(records: list[dict]) -> None:
    """Set marker_size proportional to area, median → _MARKER_SIZE."""
    if not records:
        return
    areas = np.array([r["area"] for r in records])
    ref = float(np.median(areas[areas > 0])) if np.any(areas > 0) else 1.0
    for r in records:
        r["marker_size"] = min(
            _MAX_MARKER, max(10.0, _MARKER_SIZE * r["area"] / ref)
        )


def _scale_by_group(records: list[dict]) -> None:
    """Set marker_size proportional to group_size, max group → _MARKER_SIZE."""
    if not records:
        return
    max_gs = max(r["group_size"] for r in records)
    ref = max_gs if max_gs > 0 else 1
    for r in records:
        r["marker_size"] = min(
            _MAX_MARKER,
            max(10.0, _MARKER_SIZE * r["group_size"] / ref),
        )
