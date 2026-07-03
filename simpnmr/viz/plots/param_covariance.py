# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""2-D chi-squared landscape for pairs of susceptibility fit parameters."""

import logging

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.fitting import models
from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)

# Δchi² thresholds for a 2-parameter joint confidence region
_DELTA_CHI2 = {1: 2.30, 2: 6.17, 3: 11.83}


def plot_param_covariance(
    molecule: Molecule,
    experiment: Experiment,
    susc_model: models.SusceptibilityModel,
    param_x: str,
    param_y: str,
    spec: PlotSpec,
    average_labels: list[list[str]] | None = None,
    n_sigma: float = 3.0,
    n_grid: int = 60,
    save: bool = True,
    show: bool = True,
    save_name: str = "param_covariance.pdf",
    window_title: str = "Parameter Covariance",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the chi-squared landscape for two fit parameters.

    Scans ``Σresiduals²`` over a grid of ``(param_x, param_y)`` values
    while holding all other parameters at their best-fit values.  Contour
    lines at Δχ² = 2.30 / 6.17 / 11.83 mark the 1σ / 2σ / 3σ
    joint-confidence regions for two parameters simultaneously.

    A circular or axis-aligned ellipse means the two parameters are
    uncorrelated and can be fitted independently; a tilted ellipse
    indicates correlation.

    Args:
        molecule: Molecule providing nuclei and geometric information.
        experiment: Experimental data object.
        susc_model: Fitted susceptibility model with ``final_var_values``.
        average_labels: Groups of atom labels whose predicted shifts are
            averaged before computing residuals — must match what was passed
            to ``fit_to`` so the landscape minimum aligns with the best-fit
            point.  ``None`` means no averaging.
        param_x: Model parameter name for the x-axis (in ``VARNAMES``).
        param_y: Model parameter name for the y-axis (in ``VARNAMES``).
        spec: Plot style specification.
        n_sigma: Half-width of the scan range in units of the parameter
            1σ uncertainty.  Falls back to 50 % of the best-fit value
            when the uncertainty is unavailable (fixed or NaN).
        n_grid: Number of grid points along each axis.
        save: If ``True``, saves the figure to ``save_name``.
        show: If ``True``, displays the figure.
        save_name: Output file name.
        window_title: Matplotlib window/figure title.

    Returns:
        A tuple ``(fig, ax)``.

    Raises:
        ValueError: If a parameter name is not in ``susc_model.VARNAMES``
            or if ``param_x == param_y``.
    """
    for p in (param_x, param_y):
        if p not in susc_model.VARNAMES:
            raise ValueError(
                f"Parameter {p!r} not in model VARNAMES: "
                f"{susc_model.VARNAMES}"
            )
    if param_x == param_y:
        raise ValueError("param_x and param_y must be different.")

    best = {k: float(v) for k, v in susc_model.final_var_values.items()}
    stdev = susc_model.fit_stdev

    def _half_width(p: str) -> float:
        sig = stdev.get(p)
        val = best[p]
        if sig is not None and np.isfinite(float(sig)) and float(sig) > 0:
            return n_sigma * float(sig)
        return abs(val) * 0.5 if abs(val) > 1e-12 else 1.0

    x_c, y_c = best[param_x], best[param_y]
    x_hw, y_hw = _half_width(param_x), _half_width(param_y)

    # Build nuclei list and experimental shifts (same logic as fit_to)
    _fit_nuclei = [
        nuc for nuc in molecule.nuclei if nuc.chem_label in experiment
    ]
    al_to_para_shift = {
        nuc.label: experiment[nuc.chem_label].shift - nuc.shift.dia
        for nuc in _fit_nuclei
    }
    _avg_labels = average_labels or []

    def _scan(xhw: float, yhw: float):
        xv = np.linspace(x_c - xhw, x_c + xhw, n_grid)
        yv = np.linspace(y_c - yhw, y_c + yhw, n_grid)
        grid = np.empty((n_grid, n_grid), dtype=float)
        for j, xval in enumerate(xv):
            for i, yval in enumerate(yv):
                p = {**best, param_x: xval, param_y: yval}
                res = susc_model.residuals(
                    p, _fit_nuclei, al_to_para_shift,
                    average_labels=_avg_labels,
                )
                grid[i, j] = float(np.sum(np.asarray(res) ** 2))
        return xv, yv, grid

    # Auto-expand until the 1σ contour is fully enclosed in the plot and
    # the 3σ contour is reachable.  At most 5 doublings (factor-32 max).
    _1sigma = _DELTA_CHI2[1]   # 2.30
    _3sigma = _DELTA_CHI2[3]   # 11.83
    for _attempt in range(6):
        logger.info(
            "Scanning %dx%d grid for (%s, %s), half-widths (%.4g, %.4g)…",
            n_grid, n_grid, param_x, param_y, x_hw, y_hw,
        )
        x_vals, y_vals, chi2_grid = _scan(x_hw, y_hw)
        chi2_min = chi2_grid.min()
        delta_chi2 = chi2_grid - chi2_min

        # Check if 1σ contour is enclosed (all four edges above 2.30)
        # and 3σ contour is visible somewhere in the plot.
        _edge_min = min(
            delta_chi2[0, :].min(),
            delta_chi2[-1, :].min(),
            delta_chi2[:, 0].min(),
            delta_chi2[:, -1].min(),
        )
        _enclosed = _edge_min > _1sigma
        _3sigma_visible = delta_chi2.max() > _3sigma

        if _enclosed and _3sigma_visible:
            break

        # Expand whichever axis still exits the plot, or both
        if not _enclosed:
            # Expand whichever axis still exits the 1σ boundary
            _x_exits = (
                delta_chi2[:, 0].min() <= _1sigma
                or delta_chi2[:, -1].min() <= _1sigma
            )
            _y_exits = (
                delta_chi2[0, :].min() <= _1sigma
                or delta_chi2[-1, :].min() <= _1sigma
            )
            if _x_exits:
                x_hw *= 2.0
            if _y_exits:
                y_hw *= 2.0
        else:
            # Just need more range to see 3σ
            x_hw *= 1.5
            y_hw *= 1.5

    if not _enclosed:
        logger.warning(
            "Covariance plot: 1σ contour for (%s, %s) extends beyond the "
            "plot boundary after %d expansions. The parameters may be very "
            "poorly constrained.",
            param_x, param_y, _attempt + 1,
        )

    # Principal axis of the ellipse via finite-difference Hessian at the
    # minimum.  Near the minimum Δχ² ≈ ½ δpᵀ H δp, so H⁻¹ ∝ covariance;
    # the eigenvector of H with the smallest eigenvalue is the most
    # degenerate (least-constrained) direction.
    _imin = np.unravel_index(chi2_grid.argmin(), chi2_grid.shape)
    _i0, _j0 = int(_imin[0]), int(_imin[1])
    # Use central differences if interior, else forward/backward
    _ic = np.clip(_i0, 1, n_grid - 2)
    _jc = np.clip(_j0, 1, n_grid - 2)
    _dx = x_vals[1] - x_vals[0]
    _dy = y_vals[1] - y_vals[0]
    _Hxx = (chi2_grid[_ic, _jc + 1] - 2 * chi2_grid[_ic, _jc]
            + chi2_grid[_ic, _jc - 1]) / _dx ** 2
    _Hyy = (chi2_grid[_ic + 1, _jc] - 2 * chi2_grid[_ic, _jc]
            + chi2_grid[_ic - 1, _jc]) / _dy ** 2
    _Hxy = (chi2_grid[_ic + 1, _jc + 1] - chi2_grid[_ic + 1, _jc - 1]
            - chi2_grid[_ic - 1, _jc + 1]
            + chi2_grid[_ic - 1, _jc - 1]) / (4 * _dx * _dy)
    _H = np.array([[_Hxx, _Hxy], [_Hxy, _Hyy]])
    _eigvals, _eigvecs = np.linalg.eigh(_H)
    # Smallest eigenvalue → most degenerate / longest ellipse axis
    _principal = _eigvecs[:, 0]   # (dx, dy) direction in parameter space

    # Draw the principal axis as a line through the best-fit point,
    # clipped to the plot range.
    _px, _py = _principal[0], _principal[1]
    if abs(_px) > 1e-12:
        _t_x = np.array([x_vals[0] - x_c, x_vals[-1] - x_c]) / _px
    else:
        _t_x = np.array([-1e9, 1e9])
    if abs(_py) > 1e-12:
        _t_y = np.array([y_vals[0] - y_c, y_vals[-1] - y_c]) / _py
    else:
        _t_y = np.array([-1e9, 1e9])
    _t_lo = max(min(_t_x), min(_t_y))
    _t_hi = min(max(_t_x), max(_t_y))
    _axis_x = [x_c + _t_lo * _px, x_c + _t_hi * _px]
    _axis_y = [y_c + _t_lo * _py, y_c + _t_hi * _py]

    # --- Canvas ---
    fig, ax = create_canvas(
        spec.profile,
        variant="narrow",
        window_title=window_title,
        layout="constrained",
    )
    palette = spec.palette
    spec.skin_axes(ax)
    fig.patch.set_facecolor(palette.annotation_bg)
    ax.set_facecolor(palette.annotation_bg)

    # 1σ confidence contour only
    _1sigma_level = _DELTA_CHI2[1]
    legend_handles: list[mlines.Line2D] = []
    if _1sigma_level < delta_chi2.max():
        ax.contour(
            x_vals, y_vals, delta_chi2,
            levels=[_1sigma_level],
            colors=[palette.primary],
            linewidths=1.2,
        )
        legend_handles.append(
            mlines.Line2D(
                [], [], color=palette.primary, lw=1.2, label=r"1$\sigma$"
            )
        )

    # Principal (most degenerate) axis of the ellipse
    ax.plot(
        _axis_x, _axis_y,
        color="#e05252",
        lw=1.0,
        ls="--",
        zorder=4,
    )
    legend_handles.append(
        mlines.Line2D(
            [], [],
            color="#e05252",
            lw=1.0, ls="--", label="degeneracy direction",
        )
    )

    # Best-fit cross-hair
    ax.plot(
        x_c, y_c,
        marker="+", color=palette.primary,
        markersize=10, markeredgewidth=1.5, lw=0,
        zorder=5,
    )
    legend_handles.append(
        mlines.Line2D(
            [], [],
            marker="+", color=palette.primary,
            markersize=10, markeredgewidth=1.5, lw=0,
            label="best fit",
        )
    )

    ax.legend(
        handles=legend_handles,
        fontsize=spec.typography.tick_label,
        framealpha=0.0,
        loc="upper left",
    )

    # Axis labels from model metadata
    mm = susc_model.VARNAMES_MM
    units = susc_model.UNITS_MM

    def _axis_label(p: str) -> str:
        lab = mm.get(p, p)
        u = units.get(p, "")
        return f"{lab} ({u})" if u else lab

    ax.set_xlabel(_axis_label(param_x))
    ax.set_ylabel(_axis_label(param_y))

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save:
        logger.info("Covariance plot saved to %s", save_name)

    return fig, ax
