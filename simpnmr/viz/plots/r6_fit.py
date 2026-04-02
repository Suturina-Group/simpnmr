# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot r^-6 distance-model fit results."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.core.fitting.r6_fit import compute_p1_theoretical
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
    # Choose display units based on grid range
    _tau_e_mid = np.sqrt(tau_e[0] * tau_e[-1])
    if _tau_e_mid < 1e-12:
        _tau_e_scale, _tau_e_unit = 1e15, "fs"
    elif _tau_e_mid < 1e-9:
        _tau_e_scale, _tau_e_unit = 1e12, "ps"
    else:
        _tau_e_scale, _tau_e_unit = 1e9, "ns"

    _tau_r_mid = np.sqrt(tau_R[0] * tau_R[-1])
    if _tau_r_mid < 1e-9:
        _tau_r_scale, _tau_r_unit = 1e12, "ps"
    elif _tau_r_mid < 1e-6:
        _tau_r_scale, _tau_r_unit = 1e9, "ns"
    else:
        _tau_r_scale, _tau_r_unit = 1e6, "µs"

    TAU_E_2D, TAU_R_2D = np.meshgrid(
        tau_e * _tau_e_scale, tau_R * _tau_r_scale
    )
    log_ratio_plot = log_ratio.T  # (N_R, N_e) for contourf

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
    if p1_fit != 0:
        rel_err = z * abs(p1_err / p1_fit)
        log_lo = np.log10(max(1.0 - rel_err, 1e-6))
        log_hi = np.log10(1.0 + rel_err)
    else:
        log_lo, log_hi = -0.1, 0.1

    logger.debug(
        "tau_space CI contours: log_lo=%.4f  log_hi=%.4f  "
        "p1_err/p1_fit=%.4f",
        log_lo, log_hi,
        abs(p1_err / p1_fit) if p1_fit != 0 else float("nan"),
    )
    logger.info(
        "p1 relative uncertainty (%.0f%% CI): ±%.1f%%",
        confidence * 100,
        rel_err * 100 if p1_fit != 0 else float("nan"),
    )

    # Ensure contour levels are within the clipped log_ratio range
    actual_min = float(log_ratio_plot.min())
    actual_max = float(log_ratio_plot.max())
    levels_central = [0.0] if actual_min <= 0.0 <= actual_max else []
    levels_lo = [log_lo] if actual_min <= log_lo <= actual_max else []
    levels_hi = [log_hi] if actual_min <= log_hi <= actual_max else []

    if levels_central:
        cs_c = ax.contour(
            TAU_E_2D, TAU_R_2D, log_ratio_plot,
            levels=levels_central,
            colors=["black"],
            linewidths=[1.6],
            linestyles=["-"],
        )
        for coll in cs_c.collections:
            coll.set_clip_on(True)
    if levels_lo or levels_hi:
        ci_levels = levels_lo + levels_hi
        cs_ci = ax.contour(
            TAU_E_2D, TAU_R_2D, log_ratio_plot,
            levels=sorted(ci_levels),
            colors=["white"] * len(ci_levels),
            linewidths=[1.2] * len(ci_levels),
            linestyles=["--"] * len(ci_levels),
        )
        for coll in cs_ci.collections:
            coll.set_clip_on(True)
    if not levels_central:
        logger.warning(
            "Central contour (p1_calc = p1_fit) not visible: "
            "p1_fit=%.4g is outside the computed grid range "
            "[%.4g, %.4g]. Adjust tau_e/tau_R grid limits.",
            p1_fit,
            float(np.min(p1_grid)),
            float(np.max(p1_grid)),
        )

    obs_label = _OBS_LABELS.get(observable, observable)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$\tau_e$ ({_tau_e_unit})")
    ax.set_ylabel(rf"$\tau_R$ ({_tau_r_unit})")
    ax.set_title(
        f"{obs_label}  —  {ci_pct}% CI  |  "
        r"black line: $p_1^\mathrm{calc}=p_1^\mathrm{fit}$",
        fontsize=spec.typography.title,
    )

    ann = (
        f"$p_1$ = {p1_fit:.3g} ± {p1_err:.2g}\n"
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
    tau_e = np.logspace(_tau_e_lo, _tau_e_hi, n_points)
    tau_R = np.logspace(_tau_r_lo, _tau_r_hi, n_points)

    _tau_e_mid = np.sqrt(tau_e[0] * tau_e[-1])
    if _tau_e_mid < 1e-12:
        _tau_e_scale, _tau_e_unit = 1e15, "fs"
    elif _tau_e_mid < 1e-9:
        _tau_e_scale, _tau_e_unit = 1e12, "ps"
    else:
        _tau_e_scale, _tau_e_unit = 1e9, "ns"

    _tau_r_mid = np.sqrt(tau_R[0] * tau_R[-1])
    if _tau_r_mid < 1e-9:
        _tau_r_scale, _tau_r_unit = 1e12, "ps"
    elif _tau_r_mid < 1e-6:
        _tau_r_scale, _tau_r_unit = 1e9, "ns"
    else:
        _tau_r_scale, _tau_r_unit = 1e6, "µs"

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

    # CI bands — semi-transparent fills
    def _ci_levels(fit_result):
        p1_fit = fit_result["p1"]
        p1_err = fit_result["p1_err"]
        if p1_fit == 0:
            return -0.1, 0.1
        rel = z * abs(p1_err / p1_fit)
        return np.log10(max(1.0 - rel, 1e-6)), np.log10(1.0 + rel)

    lo_r1, hi_r1 = _ci_levels(r1_fit_result)
    lo_lw, hi_lw = _ci_levels(width_fit_result)

    for _cf in [
        ax.contourf(
            TAU_E_2D, TAU_R_2D, lr_r1,
            levels=[lo_r1, hi_r1],
            colors=[palette.primary],
            alpha=0.25,
        ),
        ax.contourf(
            TAU_E_2D, TAU_R_2D, lr_lw,
            levels=[lo_lw, hi_lw],
            colors=[palette.highlight],
            alpha=0.25,
        ),
    ]:
        for coll in _cf.collections:
            coll.set_clip_on(True)

    # Central contours
    for lr, color, label in [
        (lr_r1, palette.primary, r"$R_1$ fit"),
        (lr_lw, palette.highlight, "Width fit"),
    ]:
        lo = float(lr.min())
        hi = float(lr.max())
        if lo <= 0.0 <= hi:
            cs = ax.contour(
                TAU_E_2D, TAU_R_2D, lr,
                levels=[0.0],
                colors=[color],
                linewidths=[1.6],
            )
            cs.collections[0].set_label(label)
            for coll in cs.collections:
                coll.set_clip_on(True)
        else:
            logger.warning(
                "Central contour for '%s' not visible in grid.", label
            )

    ax.legend(fontsize=spec.typography.legend, framealpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$\tau_e$ ({_tau_e_unit})")
    ax.set_ylabel(rf"$\tau_R$ ({_tau_r_unit})")
    ax.set_title(
        f"Combined τ space  —  {ci_pct}% CI",
        fontsize=spec.typography.title,
    )

    ann = (
        f"$p_1(R_1)$ = {r1_fit_result['p1']:.3g}"
        f" ± {r1_fit_result['p1_err']:.2g}\n"
        f"$p_1$(width) = {width_fit_result['p1']:.3g}"
        f" ± {width_fit_result['p1_err']:.2g}\n"
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

    _tau_e_mid = np.sqrt(tau_e[0] * tau_e[-1])
    if _tau_e_mid < 1e-12:
        _tau_e_scale, _tau_e_unit = 1e15, "fs"
    elif _tau_e_mid < 1e-9:
        _tau_e_scale, _tau_e_unit = 1e12, "ps"
    else:
        _tau_e_scale, _tau_e_unit = 1e9, "ns"

    _tau_r_mid = np.sqrt(tau_R[0] * tau_R[-1])
    if _tau_r_mid < 1e-9:
        _tau_r_scale, _tau_r_unit = 1e12, "ps"
    elif _tau_r_mid < 1e-6:
        _tau_r_scale, _tau_r_unit = 1e9, "ns"
    else:
        _tau_r_scale, _tau_r_unit = 1e6, "µs"

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

    # Colour cycle across temperatures
    colors = cm.plasma(
        np.linspace(0.1, 0.9, len(records))
    )

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
        if p1_fit != 0:
            rel = z * abs(p1_err / p1_fit)
            log_lo = np.log10(max(1.0 - rel, 1e-6))
            log_hi = np.log10(1.0 + rel)
            if lo_val <= log_lo <= hi_val or lo_val <= log_hi <= hi_val:
                cf = ax.contourf(
                    TAU_E_2D, TAU_R_2D, lr,
                    levels=[log_lo, log_hi],
                    colors=[color],
                    alpha=0.20,
                )
                for coll in cf.collections:
                    coll.set_clip_on(True)

        # Central contour
        if lo_val <= 0.0 <= hi_val:
            cs = ax.contour(
                TAU_E_2D, TAU_R_2D, lr,
                levels=[0.0],
                colors=[color],
                linewidths=[1.6],
            )
            cs.collections[0].set_label(f"{T:.0f} K")
            for coll in cs.collections:
                coll.set_clip_on(True)
        else:
            logger.warning(
                "Central contour for T=%.1f K not visible in grid "
                "(p1_fit=%.4g, grid range [%.4g, %.4g]).",
                T, p1_fit,
                float(np.min(p1_grid)), float(np.max(p1_grid)),
            )

    obs_label = _OBS_LABELS.get(observable, observable)
    ax.legend(
        title="Temperature",
        fontsize=spec.typography.legend,
        framealpha=0.8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
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
