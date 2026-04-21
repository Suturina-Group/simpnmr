# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot r^-6 distance-model fit results."""

import logging

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from simpnmr.core.fitting.r6_fit import compute_p1_theoretical
from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.uncertainty import format_compact_uncertainty

logger = logging.getLogger(__name__)


def _fmt_ps(t_s: float) -> str:
    """Format a time in seconds as a plain ps string without sci notation."""
    val = t_s * 1e12
    if val >= 10:
        return f"{val:.0f} ps"
    if val >= 1:
        return f"{val:.1f} ps"
    return f"{val:.2g} ps"


def _fmt_sci(val: float, err: float | None) -> str:
    """Format ``val ± err`` as compact scientific notation, e.g. ``4.2(3)e5``."""  # noqa: E501
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    scale = 10**exp
    scaled_err = err / scale if err is not None else None
    return f"{format_compact_uncertainty(val / scale, scaled_err)}e{exp}"


_OBS_LABELS = {
    "r1": r"$R_1$ (s$^{-1}$)",
    "width": "Linewidth (ppm)",
}

_P1_UNITS = {
    "r1": r"s$^{-1}$Å$^{6}$",
    "width": r"Hz·Å$^{6}$",
}

_P2_UNITS = {
    "r1": r"s$^{-1}$",
    "width": "Hz",
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
        fit_result: Dict returned by
            :func:`~simpnmr.core.fitting.r6_fit.fit_r6`.
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
    labels = fit_result.get("math_labels", fit_result["labels"])
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

    ax.grid(True, which="major", color=palette.grid, linewidth=0.3)
    ax.grid(True, which="minor", color=palette.grid, linewidth=0.2, alpha=0.8)
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
    _fsize = (
        glyphs.annotation_size if hasattr(glyphs, "annotation_size") else 7
    )
    for x, y, lbl in zip(r6_inv, obs, labels):
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=_fsize,
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
    p1_unit = _P1_UNITS.get(observable, "")
    p2_unit = _P2_UNITS.get(observable, "")
    ann = (
        f"$p_1$ = {_fmt_sci(p1, p1_err)} {p1_unit}\n"
        f"$p_2$ = {format_compact_uncertainty(p2, p2_err)} {p2_unit}\n"
        f"RMSE = {rmse:.3g}"
    )
    ax.text(
        0.03,
        0.97,
        ann,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=spec.typography.annotation,
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


def plot_tau_space(
    fit_result: dict,
    observable: str,
    omega_I: float,
    omega_S: float,
    gamma_I: float,
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    temperature: float,
    relaxation_model: str,
    spec: PlotSpec = None,
    tau_e_range: list[float] | None = None,
    tau_r_range: list[float] | None = None,
    n_points: int = 120,
    confidence: float = 0.95,
    tau_R_fixed: float | None = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "r6_tau_space",
    verbose: bool = True,
    window_title: str = "τ parameter space",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the (τe, τR) parameter space consistent with the fitted p1.

    Draws filled contours showing the region of (τe, τR) pairs that
    reproduce the fitted p1 within the requested confidence interval.
    The central contour corresponds to p1_fit exactly.

    Args:
        fit_result: Dict returned by
            :func:`~simpnmr.core.fitting.r6_fit.fit_r6`.
        observable: ``"r1"`` or ``"width"``.
        omega_I: Nuclear Larmor angular frequency (rad s⁻¹).
        omega_S: Electron Larmor angular frequency (rad s⁻¹).
        gamma_I: Nuclear gyromagnetic ratio (rad s⁻¹ T⁻¹).
        spin: Electron spin quantum number S.
        orbit: Orbital angular momentum L.
        total_momentum_J: Total angular momentum J, or None.
        temperature: Temperature in Kelvin.
        relaxation_model: One of ``"sbm"``, ``"curie"``,
            ``"sbm curie"``.
        spec: Plot style specification.
        n_points: Grid resolution along each axis.
        confidence: Confidence level for the shaded band (default 0.95).
        tau_R_fixed: Optional fixed τ_R value (seconds). When provided, a
            horizontal line is drawn at this τ_R, its intersection with the
            central contour (p1_calc = p1_fit) is found, and the derived τ_e
            is annotated on the figure.
        save: Save the figure if ``True``.
        show: Display the figure if ``True``.
        save_name: Output file base name.
        verbose: Log path when ``True``.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, ax)``.
    """
    p1_fit = fit_result["p1"]
    p1_err = fit_result["p1_err"]

    # z-score for the requested confidence level (two-sided)
    from scipy.stats import norm as _norm
    z = _norm.ppf(0.5 + confidence / 2.0)

    # Grid in log space — use user-supplied ranges or sensible defaults
    # Default τe: 0.01 fs to 10 ps; default τR: 1 ps to 10 µs
    _tau_e_lo = np.log10(tau_e_range[0]) if tau_e_range else -14
    _tau_e_hi = np.log10(tau_e_range[1]) if tau_e_range else -10
    _tau_r_lo = np.log10(tau_r_range[0]) if tau_r_range else -12
    _tau_r_hi = np.log10(tau_r_range[1]) if tau_r_range else -5
    # Ensure tau_R_fixed is within the computed grid
    if tau_R_fixed is not None:
        _tau_r_lo = min(_tau_r_lo, np.log10(tau_R_fixed) - 0.5)
        _tau_r_hi = max(_tau_r_hi, np.log10(tau_R_fixed) + 0.5)
    tau_e = np.logspace(_tau_e_lo, _tau_e_hi, n_points)
    tau_R = np.logspace(_tau_r_lo, _tau_r_hi, n_points)

    p1_grid = compute_p1_theoretical(
        tau_e, tau_R,
        omega_I=omega_I,
        omega_S=omega_S,
        gamma_I=gamma_I,
        spin=spin,
        orbit=orbit,
        total_momentum_J=total_momentum_J,
        temperature=temperature,
        observable=observable,
        relaxation_model=relaxation_model,
    )

    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )
    palette = spec.palette
    spec.skin_axes(ax)
    fig.patch.set_facecolor(palette.annotation_bg)
    ax.set_facecolor(palette.annotation_bg)

    # log-ratio: log10(p1_theoretical / p1_fit)
    # zero = exact match, ±log10(1 ± z*p1_err/p1_fit) = CI boundary
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(np.abs(p1_grid) / abs(p1_fit))

    log_ratio = np.clip(log_ratio, -3, 3)

    # Meshgrid for plotting — note: contourf(x, y, z) expects
    # z[row, col] where row ~ y and col ~ x.
    # With indexing="ij": axis-0 = tau_e (→ x), axis-1 = tau_R (→ y)
    # so we transpose before plotting.
    _tau_e_scale, _tau_e_unit = 1e12, "ps"
    _tau_r_scale, _tau_r_unit = 1e12, "ps"

    TAU_E_2D, TAU_R_2D = np.meshgrid(
        tau_e * _tau_e_scale, tau_R * _tau_r_scale
    )
    log_ratio_plot = log_ratio.T  # (N_R, N_e) for contourf

    # Set log scale and limits before any drawing so clip paths are correct
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(TAU_E_2D.min(), TAU_E_2D.max())
    ax.set_ylim(TAU_R_2D.min(), TAU_R_2D.max())

    # Heatmap
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    pcm = ax.pcolormesh(
        TAU_E_2D, TAU_R_2D, log_ratio_plot,
        cmap=cmap, norm=norm,
        shading="auto",
    )
    fig.colorbar(
        pcm, ax=ax,
        label=r"$\log_{10}(p_1^\mathrm{calc}/p_1^\mathrm{fit})$",
    )

    # CI boundary contours
    z = _norm.ppf(0.5 + confidence / 2.0)
    ci_pct = int(round(confidence * 100))
    if p1_err is not None and p1_fit != 0:
        rel_err = z * abs(p1_err / p1_fit)
        log_lo = np.log10(max(1.0 - rel_err, 1e-6))
        log_hi = np.log10(1.0 + rel_err)
    else:
        rel_err = None
        log_lo, log_hi = -0.1, 0.1

    logger.debug(
        "tau_space CI contours: log_lo=%.4f  log_hi=%.4f  "
        "p1_err/p1_fit=%s",
        log_lo, log_hi,
        f"{abs(p1_err / p1_fit):.4f}"
        if (p1_err is not None and p1_fit != 0) else "n/a",
    )
    logger.info(
        "p1 relative uncertainty (%.0f%% CI): %s",
        confidence * 100,
        f"±{rel_err * 100:.1f}%" if rel_err is not None else "n/a",
    )

    # Ensure contour levels are within the clipped log_ratio range
    actual_min = float(log_ratio_plot.min())
    actual_max = float(log_ratio_plot.max())
    levels_central = [0.0] if actual_min <= 0.0 <= actual_max else []
    levels_lo = [log_lo] if actual_min <= log_lo <= actual_max else []
    levels_hi = [log_hi] if actual_min <= log_hi <= actual_max else []

    if levels_central:
        ax.contour(
            TAU_E_2D, TAU_R_2D, log_ratio_plot,
            levels=levels_central,
            colors=["black"],
            linewidths=[1.6],
            linestyles=["-"],
        )
    if levels_lo or levels_hi:
        ci_levels = levels_lo + levels_hi
        ax.contour(
            TAU_E_2D, TAU_R_2D, log_ratio_plot,
            levels=sorted(ci_levels),
            colors=["white"] * len(ci_levels),
            linewidths=[1.2] * len(ci_levels),
            linestyles=["--"] * len(ci_levels),
        )
    if not levels_central:
        logger.warning(
            "Central contour (p1_calc = p1_fit) not visible: "
            "p1_fit=%.4g is outside the computed grid range "
            "[%.4g, %.4g]. Adjust tau_e/tau_R grid limits.",
            p1_fit,
            float(np.min(p1_grid)),
            float(np.max(p1_grid)),
        )

    # Clip all contour/mesh collections in one pass after drawing is done
    for coll in ax.collections:
        coll.set_clip_on(True)
        coll.set_clip_box(ax.bbox)

    obs_label = _OBS_LABELS.get(observable, observable)
    ax.set_xlabel(rf"$\tau_e$ ({_tau_e_unit})")
    ax.set_ylabel(rf"$\tau_R$ ({_tau_r_unit})")
    ax.set_title(
        f"{obs_label}  —  {ci_pct}% CI  |  "
        r"black line: $p_1^\mathrm{calc}=p_1^\mathrm{fit}$",
        fontsize=spec.typography.title,
    )

    p1_unit = _P1_UNITS.get(observable, "")
    ann = (
        f"$p_1$ = {_fmt_sci(p1_fit, p1_err)} {p1_unit}\n"
        f"model: {relaxation_model}\n"
        f"T = {temperature:.0f} K"
    )
    ax.text(
        0.97, 0.97, ann,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=spec.typography.annotation,
        color=palette.primary,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=palette.annotation_bg,
            edgecolor=palette.grid,
            alpha=0.8,
        ),
    )

    if tau_R_fixed is not None:
        tau_R_plot = tau_R_fixed * _tau_r_scale
        ax.axhline(
            tau_R_plot,
            color=palette.highlight,
            lw=1.5,
            linestyle="--",
            zorder=10,
        )

        # Find τ_e at the intersection with the central contour (log_ratio=0)
        # by interpolating log_ratio along the row closest to tau_R_fixed.
        r_idx = int(np.argmin(np.abs(tau_R - tau_R_fixed)))
        lr_row = log_ratio[:, r_idx]  # shape (N_e,), varies with tau_e
        # Find sign changes → zero crossings
        sign_changes = np.where(np.diff(np.sign(lr_row)))[0]
        tau_e_intersections: list[float] = []
        for sc in sign_changes:
            # Linear interpolation between sc and sc+1
            lr0, lr1 = lr_row[sc], lr_row[sc + 1]
            te0, te1 = tau_e[sc], tau_e[sc + 1]
            frac = -lr0 / (lr1 - lr0)
            tau_e_intersections.append(te0 * (te1 / te0) ** frac)

        for tau_e_cross in tau_e_intersections:
            tau_e_cross_plot = tau_e_cross * _tau_e_scale
            ax.axvline(
                tau_e_cross_plot,
                color=palette.highlight,
                lw=0.8,
                linestyle=":",
                zorder=5,
            )
            ax.plot(
                tau_e_cross_plot,
                tau_R_plot,
                marker="x",
                color=palette.highlight,
                markersize=8,
                markeredgewidth=1.5,
                zorder=6,
            )

        if tau_e_intersections:
            # Format τ_e values for annotation
            _fmt_tau = _fmt_ps
            _fmt_tau_r = _fmt_ps

            te_strs = ", ".join(_fmt_tau(t) for t in tau_e_intersections)
            ann_tau = (
                f"$\\tau_R$ = {_fmt_tau_r(tau_R_fixed)}\n"
                f"$\\tau_e$ = {te_strs}"
            )
            ax.text(
                0.03, 0.03, ann_tau,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=spec.typography.annotation,
                color=palette.highlight,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=palette.annotation_bg,
                    edgecolor=palette.highlight,
                    alpha=0.8,
                ),
            )
        else:
            logger.warning(
                "tau_R_fixed=%.3g s: no intersection with central contour "
                "found. Adjust tau_e_range or tau_r_range.",
                tau_R_fixed,
            )

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save and verbose:
        logger.info("τ-space plot saved to %s", f"{save_name}.pdf")

    return fig, ax


def plot_tau_space_combined(
    r1_fit_result: dict,
    width_fit_result: dict,
    omega_I: float,
    omega_S: float,
    gamma_I: float,
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    temperature: float,
    relaxation_model: str,
    spec: PlotSpec = None,
    tau_e_range: list[float] | None = None,
    tau_r_range: list[float] | None = None,
    tau_R_fixed: float | None = None,
    n_points: int = 120,
    confidence: float = 0.95,
    save: bool = True,
    show: bool = True,
    save_name: str = "r6_tau_space_combined",
    verbose: bool = True,
    window_title: str = "τ parameter space (R1 + width)",
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay R1 and linewidth τ-space contours on one plot.

    Draws the CI band for each observable in a different colour and
    the exact-match contour (p1_calc = p1_fit) as a solid line. The
    overlap region is where both observables are simultaneously
    consistent with their fitted p1 values.

    Args:
        r1_fit_result: Dict returned by ``fit_r6`` for ``observable="r1"``.
        width_fit_result: Dict returned by ``fit_r6`` for
            ``observable="width"``.
        omega_I: Nuclear Larmor angular frequency (rad s⁻¹).
        omega_S: Electron Larmor angular frequency (rad s⁻¹).
        gamma_I: Nuclear gyromagnetic ratio (rad s⁻¹ T⁻¹).
        spin: Electron spin quantum number S.
        orbit: Orbital angular momentum L.
        total_momentum_J: Total angular momentum J, or None.
        temperature: Temperature in Kelvin.
        relaxation_model: One of ``"sbm"``, ``"curie"``,
            ``"sbm curie"``.
        spec: Plot style specification.
        tau_e_range: Optional [min, max] in seconds for τe axis.
        tau_r_range: Optional [min, max] in seconds for τR axis.
        tau_R_fixed: Optional fixed τ_R value (seconds). When provided, a
            dashed horizontal line is drawn at that τ_R for both observables.
            Dotted vertical lines and cross markers are placed at each
            observable's intersection with the central contour, using that
            observable's colour.
        n_points: Grid resolution along each axis.
        confidence: Confidence level for shaded bands.
        save: Save the figure if ``True``.
        show: Display the figure if ``True``.
        save_name: Output file base name.
        verbose: Log path when ``True``.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, ax)``.
    """
    from scipy.stats import norm as _norm

    z = _norm.ppf(0.5 + confidence / 2.0)
    ci_pct = int(round(confidence * 100))

    _tau_e_lo = np.log10(tau_e_range[0]) if tau_e_range else -14
    _tau_e_hi = np.log10(tau_e_range[1]) if tau_e_range else -10
    _tau_r_lo = np.log10(tau_r_range[0]) if tau_r_range else -12
    _tau_r_hi = np.log10(tau_r_range[1]) if tau_r_range else -5

    if tau_R_fixed is not None:
        _tau_r_lo = min(_tau_r_lo, np.log10(tau_R_fixed) - 0.5)
        _tau_r_hi = max(_tau_r_hi, np.log10(tau_R_fixed) + 0.5)
    tau_e = np.logspace(_tau_e_lo, _tau_e_hi, n_points)
    tau_R = np.logspace(_tau_r_lo, _tau_r_hi, n_points)

    _tau_e_scale, _tau_e_unit = 1e12, "ps"
    _tau_r_scale, _tau_r_unit = 1e12, "ps"

    TAU_E_2D, TAU_R_2D = np.meshgrid(
        tau_e * _tau_e_scale, tau_R * _tau_r_scale
    )

    def _log_ratio_grid(fit_result, obs):
        p1_fit = fit_result["p1"]
        grid = compute_p1_theoretical(
            tau_e, tau_R,
            omega_I=omega_I, omega_S=omega_S, gamma_I=gamma_I,
            spin=spin, orbit=orbit,
            total_momentum_J=total_momentum_J,
            temperature=temperature,
            observable=obs,
            relaxation_model=relaxation_model,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log10(np.abs(grid) / abs(p1_fit))
        return np.clip(lr.T, -3, 3)

    lr_r1 = _log_ratio_grid(r1_fit_result, "r1")
    lr_lw = _log_ratio_grid(width_fit_result, "width")

    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )
    palette = spec.palette
    spec.skin_axes(ax)

    # Set log scale and explicit limits before drawing any contours so that
    # set_clip_path(ax.patch) uses the correct transform.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(TAU_E_2D.min(), TAU_E_2D.max())
    ax.set_ylim(TAU_R_2D.min(), TAU_R_2D.max())

    # CI boundary contours (dashed) and central contours (solid)
    def _ci_levels(fit_result):
        p1_fit = fit_result["p1"]
        p1_err = fit_result["p1_err"]
        if p1_err is None or p1_fit == 0:
            return -0.1, 0.1
        rel = z * abs(p1_err / p1_fit)
        return np.log10(max(1.0 - rel, 1e-6)), np.log10(1.0 + rel)

    lo_r1, hi_r1 = _ci_levels(r1_fit_result)
    lo_lw, hi_lw = _ci_levels(width_fit_result)

    legend_handles = []
    legend_labels_list = []

    for lr, color, label, lo_ci, hi_ci in [
        (lr_r1, palette.primary, r"$R_1$ fit", lo_r1, hi_r1),
        (lr_lw, palette.highlight, "Width fit", lo_lw, hi_lw),
    ]:
        lo_data = float(lr.min())
        hi_data = float(lr.max())

        # CI boundary dashed lines
        ci_levels = [
            lv for lv in [lo_ci, hi_ci]
            if lo_data <= lv <= hi_data
        ]
        if ci_levels:
            ax.contour(
                TAU_E_2D, TAU_R_2D, lr,
                levels=sorted(ci_levels),
                colors=[color] * len(ci_levels),
                linewidths=[1.0] * len(ci_levels),
                linestyles=["--"] * len(ci_levels),
            )

        # Central solid line
        if lo_data <= 0.0 <= hi_data:
            ax.contour(
                TAU_E_2D, TAU_R_2D, lr,
                levels=[0.0],
                colors=[color],
                linewidths=[1.6],
            )
            legend_handles.append(
                Line2D([0], [0], color=color, lw=1.6)
            )
            legend_labels_list.append(label)
        else:
            logger.warning(
                "Central contour for '%s' not visible in grid.", label
            )

    # Clip all line collections in one pass after drawing is complete
    for coll in ax.collections:
        coll.set_clip_on(True)
        coll.set_clip_box(ax.bbox)

    if legend_handles:
        ax.legend(
            legend_handles, legend_labels_list,
            fontsize=spec.typography.legend, framealpha=0.8,
            loc="upper left",
        )
    ax.set_xlabel(rf"$\tau_e$ ({_tau_e_unit})")
    ax.set_ylabel(rf"$\tau_R$ ({_tau_r_unit})")
    ax.set_title(
        f"Combined τ space  —  {ci_pct}% CI",
        fontsize=spec.typography.title,
    )

    _r1_str = _fmt_sci(r1_fit_result['p1'], r1_fit_result['p1_err'])
    _w_str = _fmt_sci(width_fit_result['p1'], width_fit_result['p1_err'])
    ann = (
        f"$p_1(R_1)$ = {_r1_str} {_P1_UNITS['r1']}\n"
        f"$p_1$(width) = {_w_str} {_P1_UNITS['width']}\n"
        f"model: {relaxation_model}  |  T = {temperature:.0f} K"
    )
    ax.text(
        0.97, 0.97, ann,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=spec.typography.annotation,
        color=palette.primary,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=palette.annotation_bg,
            edgecolor=palette.grid,
            alpha=0.8,
        ),
    )

    if tau_R_fixed is not None:
        tau_R_plot = tau_R_fixed * _tau_r_scale
        ax.axhline(
            tau_R_plot,
            color=palette.grid,
            lw=1.5,
            linestyle="--",
            zorder=10,
        )

        _fmt_tau = _fmt_ps
        _fmt_tau_r = _fmt_ps

        r_idx = int(np.argmin(np.abs(tau_R - tau_R_fixed)))
        ann_lines = [f"$\\tau_R$ = {_fmt_tau_r(tau_R_fixed)}"]

        for lr, color, obs_label in [
            (lr_r1, palette.primary, "$R_1$"),
            (lr_lw, palette.highlight, "width"),
        ]:
            # lr_r1 / lr_lw come from _log_ratio_grid which returns lr.T,
            # shape (n_tau_R, n_tau_e).  Row r_idx fixes tau_R, columns vary
            # with tau_e — the same slice used in plot_tau_space's log_ratio.
            lr_row = lr[r_idx, :]
            sign_changes = np.where(np.diff(np.sign(lr_row)))[0]
            tau_e_intersections = []
            for sc in sign_changes:
                lr0, lr1 = lr_row[sc], lr_row[sc + 1]
                te0, te1 = tau_e[sc], tau_e[sc + 1]
                frac = -lr0 / (lr1 - lr0)
                tau_e_intersections.append(te0 * (te1 / te0) ** frac)

            for tau_e_cross in tau_e_intersections:
                tau_e_cross_plot = tau_e_cross * _tau_e_scale
                ax.axvline(
                    tau_e_cross_plot,
                    color=color,
                    lw=0.8,
                    linestyle=":",
                    zorder=5,
                )
                ax.plot(
                    tau_e_cross_plot,
                    tau_R_plot,
                    marker="x",
                    color=color,
                    markersize=8,
                    markeredgewidth=1.5,
                    zorder=6,
                )

            if tau_e_intersections:
                te_strs = ", ".join(_fmt_tau(t) for t in tau_e_intersections)
                ann_lines.append(f"$\\tau_e$({obs_label}) = {te_strs}")
            else:
                logger.warning(
                    "tau_R_fixed=%.3g s: no intersection with %s central "
                    "contour. Adjust tau_e_range or tau_r_range.",
                    tau_R_fixed, obs_label,
                )

        if len(ann_lines) > 1:
            ax.text(
                0.03, 0.03, "\n".join(ann_lines),
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=spec.typography.annotation,
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
        logger.info(
            "Combined τ-space plot saved to %s", f"{save_name}.pdf"
        )

    return fig, ax


def plot_tau_space_multitemp(
    records: list[dict],
    observable: str,
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    relaxation_model: str,
    spec: PlotSpec = None,
    tau_e_range: list[float] | None = None,
    tau_r_range: list[float] | None = None,
    n_points: int = 120,
    confidence: float = 0.95,
    save: bool = True,
    show: bool = True,
    save_name: str = "r6_tau_space_multitemp",
    verbose: bool = True,
    window_title: str = "τ parameter space (multi-T)",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot τ-space constraint contours for multiple temperatures.

    Each temperature produces a different constraint curve in (τe, τR)
    space. The intersection of all curves indicates the (τe, τR) pair
    consistent with all experimental temperatures simultaneously.

    Args:
        records: List of dicts, one per temperature, each containing:
            ``"fit_result"`` (from ``fit_r6``),
            ``"temperature"`` (K),
            ``"omega_I"`` (rad s⁻¹),
            ``"omega_S"`` (rad s⁻¹),
            ``"gamma_I"`` (rad s⁻¹ T⁻¹).
        observable: ``"r1"`` or ``"width"``.
        spin: Electron spin quantum number S.
        orbit: Orbital angular momentum L.
        total_momentum_J: Total angular momentum J, or None.
        relaxation_model: One of ``"sbm"``, ``"curie"``,
            ``"sbm curie"``.
        spec: Plot style specification.
        tau_e_range: Optional [min, max] in seconds for τe axis.
        tau_r_range: Optional [min, max] in seconds for τR axis.
        n_points: Grid resolution along each axis.
        confidence: Confidence level for shaded CI bands.
        save: Save the figure if ``True``.
        show: Display the figure if ``True``.
        save_name: Output file base name.
        verbose: Log path when ``True``.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, ax)``.
    """
    from scipy.stats import norm as _norm
    import matplotlib.cm as cm

    if not records:
        logger.warning("No records provided to plot_tau_space_multitemp.")
        return None, None

    z = _norm.ppf(0.5 + confidence / 2.0)
    ci_pct = int(round(confidence * 100))

    _tau_e_lo = np.log10(tau_e_range[0]) if tau_e_range else -14
    _tau_e_hi = np.log10(tau_e_range[1]) if tau_e_range else -10
    _tau_r_lo = np.log10(tau_r_range[0]) if tau_r_range else -12
    _tau_r_hi = np.log10(tau_r_range[1]) if tau_r_range else -5
    tau_e = np.logspace(_tau_e_lo, _tau_e_hi, n_points)
    tau_R = np.logspace(_tau_r_lo, _tau_r_hi, n_points)

    _tau_e_scale, _tau_e_unit = 1e12, "ps"
    _tau_r_scale, _tau_r_unit = 1e12, "ps"

    TAU_E_2D, TAU_R_2D = np.meshgrid(
        tau_e * _tau_e_scale, tau_R * _tau_r_scale
    )

    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )
    palette = spec.palette
    spec.skin_axes(ax)
    ax.set_facecolor(palette.annotation_bg)

    # Set log scale and explicit limits before drawing contours
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(TAU_E_2D.min(), TAU_E_2D.max())
    ax.set_ylim(TAU_R_2D.min(), TAU_R_2D.max())

    # Colour cycle across temperatures
    colors = cm.plasma(
        np.linspace(0.1, 0.9, len(records))
    )

    legend_handles = []
    legend_labels = []

    for rec, color in zip(records, colors):
        fit_result = rec["fit_result"]
        T = rec["temperature"]
        omega_I = rec["omega_I"]
        omega_S = rec["omega_S"]
        gamma_I = rec["gamma_I"]
        p1_fit = fit_result["p1"]
        p1_err = fit_result["p1_err"]

        p1_grid = compute_p1_theoretical(
            tau_e, tau_R,
            omega_I=omega_I, omega_S=omega_S, gamma_I=gamma_I,
            spin=spin, orbit=orbit,
            total_momentum_J=total_momentum_J,
            temperature=T,
            observable=observable,
            relaxation_model=relaxation_model,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log10(np.abs(p1_grid) / abs(p1_fit))
        lr = np.clip(lr.T, -3, 3)

        lo_val = float(lr.min())
        hi_val = float(lr.max())

        # CI band
        if p1_err is not None and p1_fit != 0:
            rel = z * abs(p1_err / p1_fit)
            log_lo = np.log10(max(1.0 - rel, 1e-6))
            log_hi = np.log10(1.0 + rel)
            if lo_val <= log_lo <= hi_val or lo_val <= log_hi <= hi_val:
                ax.contourf(
                    TAU_E_2D, TAU_R_2D, lr,
                    levels=[log_lo, log_hi],
                    colors=[color],
                    alpha=0.20,
                )

        # Central contour
        if lo_val <= 0.0 <= hi_val:
            ax.contour(
                TAU_E_2D, TAU_R_2D, lr,
                levels=[0.0],
                colors=[color],
                linewidths=[1.6],
            )
            legend_handles.append(Line2D([0], [0], color=color, lw=1.6))
            legend_labels.append(f"{T:.0f} K")
        else:
            logger.warning(
                "Central contour for T=%.1f K not visible in grid "
                "(p1_fit=%.4g, grid range [%.4g, %.4g]).",
                T, p1_fit,
                float(np.min(p1_grid)), float(np.max(p1_grid)),
            )

    # Clip all contour collections to the axes patch in one pass after drawing
    for coll in ax.collections:
        coll.set_clip_on(True)
        coll.set_clip_box(ax.bbox)

    obs_label = _OBS_LABELS.get(observable, observable)
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            title="Temperature",
            fontsize=spec.typography.legend,
            framealpha=0.8,
        )
    ax.set_xlabel(rf"$\tau_e$ ({_tau_e_unit})")
    ax.set_ylabel(rf"$\tau_R$ ({_tau_r_unit})")
    ax.set_title(
        f"{obs_label}  —  {ci_pct}% CI per temperature",
        fontsize=spec.typography.title,
    )

    ann = f"model: {relaxation_model}"
    ax.text(
        0.97, 0.03, ann,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=spec.typography.annotation,
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
        logger.info(
            "Multi-T τ-space plot saved to %s", f"{save_name}.pdf"
        )

    return fig, ax
