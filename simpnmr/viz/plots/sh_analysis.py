# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Spin-Hamiltonian solution-line plots for the axial VT-fit case.

For a purely axial system (rh = 0) the fitted ax_intercept constrains the
curve g_ax(g_iso) and the fitted ax_slope constrains D(g_iso), but neither
pins g_iso uniquely.  This figure overlays both curves and marks two
candidate g_iso estimates:

  1.  From the isotropic chiT intercept (fit extrapolated to 1/T → 0).
  2.  From the highest-temperature data point (most direct high-T estimate).
"""

import csv
import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from simpnmr.core.const.physics import C, GE, H, KB
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.uncertainty import format_compact_uncertainty

logger = logging.getLogger(__name__)


def plot_g_iso_solution_lines(
    ax_intercept: float,
    ax_slope: float,
    iso_intercept: float,
    chiT_red_iso: np.ndarray,
    inv_t: np.ndarray,
    spin: float,
    total_J: float | None = None,
    ax_intercept_err: float = 0.0,
    ax_slope_err: float = 0.0,
    iso_intercept_err: float = 0.0,
    chiT_err_iso: np.ndarray | None = None,
    zeta_eff: float | None = None,
    zeta_eff_tol: float = 0.15,
    evans_g_iso: float | None = None,
    evans_g_iso_err: float = 0.0,
    orca_point: tuple[float, float, float] | None = None,
    hfc_file: str = "",
    tip_correction: float | None = None,
    spec: PlotSpec | None = None,
    show: bool = True,
    save: bool = True,
    save_name: str = "g_iso_solution",
    window_title: str = "g-tensor solution lines",
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Plot g_ax and D solution curves as a function of assumed g_iso.

    For a fixed ax_intercept and ax_slope, the values of g_ax = g_∥ − g_⊥
    and D (ZFS, cm⁻¹) depend on what g_iso is assumed.  This plot shows
    those dependences as continuous curves, with vertical markers at two
    reference g_iso values so the user can read off the implied g_ax / D
    without committing to a single isotropic g.

    Args:
        ax_intercept: Fitted intercept of the axial chiT component
            (dimensionless, Curie-normalised).
        ax_slope: Fitted slope of the axial chiT component (K, from B/T term).
        iso_intercept: Fitted intercept of the isotropic chiT component.
        chiT_red_iso: Array of Curie-normalised isotropic chiT values at each
            temperature (actual data, not fit).
        inv_t: Inverse temperatures (K⁻¹) corresponding to ``chiT_red_iso``.
        spin: Spin quantum number S.
        total_J: Total angular momentum J (replaces S in f_S when provided).
        ax_intercept_err: 1-σ uncertainty on ``ax_intercept`` from the fit.
        ax_slope_err: 1-σ uncertainty on ``ax_slope`` from the fit.
        iso_intercept_err: 1-σ uncertainty on ``iso_intercept`` from the fit;
            propagated into the g_iso uncertainty for the intercept reference.
        chiT_err_iso: Per-temperature 1-σ uncertainties on ``chiT_red_iso``;
            used to estimate the g_iso error for the high-T reference point.
        zeta_eff: Known (or estimated) effective spin-orbit coupling constant
            in cm⁻¹.  When provided, a horizontal reference band is drawn on
            the ζ̄_eff panel at ``zeta_eff ± zeta_eff_tol`` (fractional).
        zeta_eff_tol: Fractional uncertainty on ``zeta_eff`` used for the
            shaded band (default 0.15, i.e. ±15 %).
        evans_g_iso: Isotropic g-value derived from an Evans-method NMR
            susceptibility measurement.  When provided, vertical read-off
            markers are drawn on all three panels so that g_ax, D and ζ̄_eff
            at the Evans g_iso can be read directly from the curves.
        evans_g_iso_err: 1-σ uncertainty on ``evans_g_iso`` (default 0).
        spec: Plot style specification.
        show: Display the figure interactively.
        save: Save the figure to ``save_name``.
        save_name: Output file path (extension appended by the export layer).
        window_title: Matplotlib window/figure title.

    Returns:
        ``(fig, (ax1, ax2, ax3))``.
    """
    inv_t = np.asarray(inv_t, dtype=float)
    chiT_red_iso = np.asarray(chiT_red_iso, dtype=float)

    # --- Reference g_iso estimates -------------------------------------------
    g_iso_intercept = iso_intercept / GE

    idx_tmax = int(np.argmin(inv_t))
    _chiT_hT = chiT_red_iso[idx_tmax]
    if tip_correction is not None and np.isfinite(float(tip_correction)):
        _chiT_hT = _chiT_hT - float(tip_correction) / inv_t[idx_tmax]
    g_iso_high_t = _chiT_hT / GE

    # --- g_iso scan axis -----------------------------------------------------
    g_center = 0.5 * (g_iso_intercept + g_iso_high_t)
    half_span = max(abs(g_iso_intercept - g_iso_high_t) * 2.5, 0.25)
    # Expand to cover NEVPT2/Evans g_iso so curves reach the full x-axis range
    if orca_point is not None:
        half_span = max(half_span, abs(orca_point[0] - g_center) * 1.15)
    if evans_g_iso is not None:
        half_span = max(half_span, abs(evans_g_iso - g_center) * 1.15)
    g_iso_arr = np.linspace(g_center - half_span, g_center + half_span, 400)

    # --- Axial solution: g_⊥ and g_∥ as functions of g_iso ------------------
    sqrt_arg = ax_intercept / 3.0 + g_iso_arr ** 2
    sqrt_term = np.sqrt(np.maximum(sqrt_arg, 0.0))
    g_perp = 2.0 * g_iso_arr - sqrt_term
    g_par = -g_iso_arr + 2.0 * sqrt_term
    g_ax_curve = g_par - g_perp

    valid = (g_perp > 0) & (g_par > 0) & np.isfinite(g_ax_curve)

    # --- D(g_iso) from ax_slope ----------------------------------------------
    J_eff = total_J if total_J is not None else spin
    f_S = (2.0 * J_eff - 1.0) * (2.0 * J_eff + 3.0)

    denom = f_S * (2.0 * g_par ** 2 + g_perp ** 2)
    D_J = np.where(np.abs(denom) > 1e-30,
                   -ax_slope * 30.0 * KB / denom, np.nan)
    D_cm = D_J / (H * C * 100.0)

    # --- Point evaluators at specific g_iso values ---------------------------
    def _point(g_iso_val: float) -> tuple[float, float]:
        s = float(np.sqrt(max(ax_intercept / 3.0 + g_iso_val ** 2, 0.0)))
        gp = 2.0 * g_iso_val - s
        gz = -g_iso_val + 2.0 * s
        if gp <= 0 or gz <= 0:
            return float("nan"), float("nan")
        d = f_S * (2.0 * gz ** 2 + gp ** 2)
        if abs(d) < 1e-30:
            return float(gz - gp), float("nan")
        return float(gz - gp), float(
            (-ax_slope * 30.0 * KB / d) / (H * C * 100.0)
        )

    # --- zeta_eff curve: -4S * D / g_ax -------------------------------------
    J_spin = total_J if total_J is not None else spin
    zeta_curve = np.where(
        valid & np.isfinite(D_cm) & (np.abs(g_ax_curve) > 1e-10),
        -4.0 * J_spin * D_cm / g_ax_curve,
        np.nan,
    )
    zeta_valid = np.isfinite(zeta_curve)

    def _zeta_point(g_iso_val: float) -> float:
        g_ax_val, D_val = _point(g_iso_val)
        if (
            not np.isfinite(g_ax_val)
            or not np.isfinite(D_val)
            or abs(g_ax_val) < 1e-10
        ):
            return float("nan")
        return -4.0 * J_spin * D_val / g_ax_val

    # --- Error propagation via central finite differences --------------------
    def _point_at(ai: float, asl: float, gi: float) -> tuple[float, float]:
        s = float(np.sqrt(max(ai / 3.0 + gi ** 2, 0.0)))
        gp = 2.0 * gi - s
        gz = -gi + 2.0 * s
        if gp <= 0 or gz <= 0:
            return float("nan"), float("nan")
        d = f_S * (2.0 * gz ** 2 + gp ** 2)
        if abs(d) < 1e-30:
            return float(gz - gp), float("nan")
        return float(gz - gp), float((-asl * 30.0 * KB / d) / (H * C * 100.0))

    def _propagate_errors(
        g_iso_val: float, g_iso_err: float
    ) -> tuple[float, float]:
        """Return (σ_g_ax, σ_D) from fit-parameter uncertainties."""
        h_ai = max(abs(ax_intercept) * 1e-5, 1e-10)
        h_asl = (
            max(abs(ax_slope) * 1e-5, 1e-10)
            if abs(ax_slope) > 1e-15 else 1e-10
        )
        h_gi = max(abs(g_iso_val) * 1e-5, 1e-8)

        ga_p, D_p = _point_at(ax_intercept + h_ai, ax_slope, g_iso_val)
        ga_m, D_m = _point_at(ax_intercept - h_ai, ax_slope, g_iso_val)
        dga_dai = (ga_p - ga_m) / (2.0 * h_ai)
        dD_dai = (D_p - D_m) / (2.0 * h_ai)

        _, D_sp = _point_at(ax_intercept, ax_slope + h_asl, g_iso_val)
        _, D_sm = _point_at(ax_intercept, ax_slope - h_asl, g_iso_val)
        dD_dasl = (D_sp - D_sm) / (2.0 * h_asl)

        ga_gp, D_gp = _point_at(ax_intercept, ax_slope, g_iso_val + h_gi)
        ga_gm, D_gm = _point_at(ax_intercept, ax_slope, g_iso_val - h_gi)
        dga_dgi = (ga_gp - ga_gm) / (2.0 * h_gi)
        dD_dgi = (D_gp - D_gm) / (2.0 * h_gi)

        def _sq(x: float) -> float:
            return x * x if np.isfinite(x) else 0.0

        sigma_ga = float(np.sqrt(
            _sq(dga_dai * ax_intercept_err) + _sq(dga_dgi * g_iso_err)
        ))
        sigma_D = float(np.sqrt(
            _sq(dD_dai * ax_intercept_err) +
            _sq(dD_dasl * ax_slope_err) +
            _sq(dD_dgi * g_iso_err)
        ))
        return sigma_ga, sigma_D

    def _zeta_at(ai: float, asl: float, gi: float) -> float:
        """ζ̄_eff for arbitrary (ax_intercept, ax_slope, g_iso)."""
        ga, D = _point_at(ai, asl, gi)
        if not np.isfinite(ga) or not np.isfinite(D) or abs(ga) < 1e-10:
            return float("nan")
        return -4.0 * J_spin * D / ga

    def _zeta_sigma(g_iso_val: float, g_iso_err: float) -> float:
        """σ(ζ) at a given g_iso from fit-parameter uncertainties."""
        h_ai = max(abs(ax_intercept) * 1e-5, 1e-10)
        h_asl = (
            max(abs(ax_slope) * 1e-5, 1e-10)
            if abs(ax_slope) > 1e-15 else 1e-10
        )
        h_gi = max(abs(g_iso_val) * 1e-5, 1e-8)

        dz_dai = (
            _zeta_at(ax_intercept + h_ai, ax_slope, g_iso_val)
            - _zeta_at(ax_intercept - h_ai, ax_slope, g_iso_val)
        ) / (2.0 * h_ai)
        dz_dasl = (
            _zeta_at(ax_intercept, ax_slope + h_asl, g_iso_val)
            - _zeta_at(ax_intercept, ax_slope - h_asl, g_iso_val)
        ) / (2.0 * h_asl)
        dz_dgi = (
            _zeta_at(ax_intercept, ax_slope, g_iso_val + h_gi)
            - _zeta_at(ax_intercept, ax_slope, g_iso_val - h_gi)
        ) / (2.0 * h_gi)

        def _sq(x: float) -> float:
            return x * x if np.isfinite(x) else 0.0

        return float(np.sqrt(
            _sq(dz_dai * ax_intercept_err) +
            _sq(dz_dasl * ax_slope_err) +
            _sq(dz_dgi * g_iso_err)
        ))

    def _cross_g_iso_sigma(g_cross: float) -> float:
        """Return σ(g_iso) at the ζ-crossing via σ_ζ / |dζ/dg_iso|."""
        h_g = max(abs(g_cross) * 1e-4, 1e-6)
        zp = _zeta_point(g_cross + h_g)
        zm = _zeta_point(g_cross - h_g)
        if not (np.isfinite(zp) and np.isfinite(zm)):
            return 0.0
        dz_dg = (zp - zm) / (2.0 * h_g)
        if abs(dz_dg) < 1e-10:
            return 0.0

        h_ai = max(abs(ax_intercept) * 1e-5, 1e-10)
        h_asl = (
            max(abs(ax_slope) * 1e-5, 1e-10)
            if abs(ax_slope) > 1e-15 else 1e-10
        )

        def _sq(x: float) -> float:
            return x * x if np.isfinite(x) else 0.0

        dz_dai = (
            _zeta_at(ax_intercept + h_ai, ax_slope, g_cross)
            - _zeta_at(ax_intercept - h_ai, ax_slope, g_cross)
        ) / (2.0 * h_ai)
        dz_dasl = (
            _zeta_at(ax_intercept, ax_slope + h_asl, g_cross)
            - _zeta_at(ax_intercept, ax_slope - h_asl, g_cross)
        ) / (2.0 * h_asl)
        sigma_zeta = float(np.sqrt(
            _sq(dz_dai * ax_intercept_err) +
            _sq(dz_dasl * ax_slope_err)
        ))
        return float(sigma_zeta / abs(dz_dg))

    # --- g_iso uncertainties at each reference point -------------------------
    g_iso_err_intercept = iso_intercept_err / GE
    g_iso_err_high_t = 0.0
    if chiT_err_iso is not None:
        _chiT_err = np.asarray(chiT_err_iso, dtype=float)
        g_iso_err_high_t = float(_chiT_err[idx_tmax]) / GE

    # --- Styling ---------------------------------------------------------
    if spec is not None:
        col_curve = spec.palette.primary
        col_intercept = spec.palette.primary
        col_high_t = spec.palette.highlight
        col_zeta = getattr(spec.palette, "secondary", "#5a9e6f")
        col_evans = getattr(spec.palette, "tertiary", "#e07b00")
        tick_fs = spec.typography.tick_label
        label_fs = spec.typography.axis_label
    else:
        col_curve = "#2a6496"
        col_intercept = "#2a6496"
        col_high_t = "#e05252"
        col_zeta = "#5a9e6f"
        col_evans = "#e07b00"
        tick_fs = label_fs = 8

    col_orca = col_zeta

    n_panels = 3
    width_scale = 1.05 * n_panels

    if spec is not None:
        from simpnmr.viz.layout.figure import get_figsize
        base = get_figsize(spec.profile, "standard")
        figsize = (base[0] * width_scale, base[1])
    else:
        figsize = (3.3 * n_panels, 2.2)

    axes_list: list[plt.Axes]
    fig, axes_list = plt.subplots(
        1, n_panels,
        figsize=figsize,
        num=window_title,
        layout="constrained",
    )
    ax1, ax2 = axes_list[0], axes_list[1]
    ax3: plt.Axes = axes_list[2]

    # ── Find ζ crossing on a temporary expanding grid ────────────────────
    crossings: list[float] = []
    if zeta_eff is not None:
        _srch_half = half_span
        _srch_arr = g_iso_arr.copy()
        _srch_curve = zeta_curve.copy()
        _srch_valid = zeta_valid.copy()
        for _expand in range(8):
            _diff = np.where(_srch_valid, _srch_curve - zeta_eff, np.nan)
            _has_cross = any(
                _diff[k] * _diff[k + 1] <= 0
                and np.isfinite(_diff[k]) and np.isfinite(_diff[k + 1])
                for k in range(len(_diff) - 1)
            )
            if _has_cross or _srch_half >= 2.0:
                break
            _srch_half = min(_srch_half * 1.6, 2.0)
            _srch_arr = np.linspace(
                g_center - _srch_half, g_center + _srch_half, 600
            )
            _sq3 = ax_intercept / 3.0 + _srch_arr ** 2
            _st3 = np.sqrt(np.maximum(_sq3, 0.0))
            _gp3 = 2.0 * _srch_arr - _st3
            _gz3 = -_srch_arr + 2.0 * _st3
            _ga3 = _gz3 - _gp3
            _dn3 = f_S * (2.0 * _gz3 ** 2 + _gp3 ** 2)
            _D3 = np.where(
                np.abs(_dn3) > 1e-30,
                -ax_slope * 30.0 * KB / _dn3 / (H * C * 100.0),
                np.nan,
            )
            _v3 = (_gp3 > 0) & (_gz3 > 0) & np.isfinite(_ga3)
            _srch_curve = np.where(
                _v3 & np.isfinite(_D3) & (np.abs(_ga3) > 1e-10),
                -4.0 * J_eff * _D3 / _ga3,
                np.nan,
            )
            _srch_valid = np.isfinite(_srch_curve)

        _zf = _srch_curve[_srch_valid]
        _gf = _srch_arr[_srch_valid]
        _df = _zf - zeta_eff
        for k in range(len(_df) - 1):
            if (
                _df[k] * _df[k + 1] <= 0
                and np.isfinite(_df[k]) and np.isfinite(_df[k + 1])
            ):
                frac = _df[k] / (_df[k] - _df[k + 1])
                crossings.append(float(_gf[k] + frac * (_gf[k + 1] - _gf[k])))

    g_iso_cross: float | None = crossings[0] if crossings else None

    # ── Build tight z3_arr for panel 3 ───────────────────────────────────────
    # Cover reference points + crossing + NEVPT2 point, then add a 10% margin.
    _z3_pts = [g_iso_intercept, g_iso_high_t]
    if g_iso_cross is not None:
        _z3_pts.append(g_iso_cross)
    if orca_point is not None:
        _z3_pts.append(orca_point[0])
    if evans_g_iso is not None:
        _z3_pts.append(evans_g_iso)
    _z3_lo = min(_z3_pts)
    _z3_hi = max(_z3_pts)
    _z3_margin = max((_z3_hi - _z3_lo) * 0.10, 0.05)
    z3_arr = np.linspace(_z3_lo - _z3_margin, _z3_hi + _z3_margin, 600)
    _sq3 = ax_intercept / 3.0 + z3_arr ** 2
    _st3 = np.sqrt(np.maximum(_sq3, 0.0))
    _gp3 = 2.0 * z3_arr - _st3
    _gz3 = -z3_arr + 2.0 * _st3
    _ga3 = _gz3 - _gp3
    _dn3 = f_S * (2.0 * _gz3 ** 2 + _gp3 ** 2)
    _D3_arr = np.where(
        np.abs(_dn3) > 1e-30,
        -ax_slope * 30.0 * KB / _dn3 / (H * C * 100.0),
        np.nan,
    )
    _v3 = (_gp3 > 0) & (_gz3 > 0) & np.isfinite(_ga3)
    z3_curve = np.where(
        _v3 & np.isfinite(_D3_arr) & (np.abs(_ga3) > 1e-10),
        -4.0 * J_eff * _D3_arr / _ga3,
        np.nan,
    )
    z3_valid = np.isfinite(z3_curve)

    # Shared x-axis limits: all three panels use the z3_arr range
    _x_lo = float(z3_arr[0])
    _x_hi = float(z3_arr[-1])

    # ── Helpers ──────────────────────────────────────────────────────────
    _ap = dict(lw=0.8, mutation_scale=5)   # shared arrow style kwargs

    def _L_arrows(
        ax: plt.Axes, g_ref: float, col: str, value: float,
    ) -> None:
        """L-shaped read-off arrows: x-axis → curve point → y-axis.

        Draws two thin arrows:
          1. Vertical: from the bottom of the visible plot up to the curve.
          2. Horizontal: from the curve point left to the y-axis spine.
        A small circle marks the intersection with the curve.
        """
        y_bot = ax.get_ylim()[0]
        x0, x1 = ax.get_xlim()
        x_stop = x0 + 0.02 * (x1 - x0)   # stop just before the y-axis spine
        ap = dict(arrowstyle="-|>", color=col, **_ap)
        # vertical: bottom → curve
        ax.annotate("", xy=(g_ref, value), xytext=(g_ref, y_bot),
                    arrowprops=ap, annotation_clip=False)
        # horizontal: curve → y-axis
        ax.annotate("", xy=(x_stop, value), xytext=(g_ref, value),
                    arrowprops=ap, annotation_clip=False)
        ax.plot(g_ref, value, marker="o", color=col, ms=4, zorder=6, lw=0)

    def _in_panel_legend(
        ax: plt.Axes,
        entries: list[tuple[str, str]],
        loc: str = "upper right",
        x_pad: float = 0.03,
        y_pad: float = 0.03,
    ) -> None:
        """Draw stacked coloured text lines inside the panel.

        Args:
            entries: List of (text, colour) pairs, drawn top-to-bottom.
            loc: Corner placement — "upper right", "upper left",
                 "lower right", "lower left".
            x_pad: Horizontal margin from the chosen edge (axes fraction).
            y_pad: Vertical margin from the chosen edge (axes fraction).
        """
        n = len(entries)
        fs = tick_fs * 0.82
        line_h = fs / 72.0 / figsize[1] * 1.5   # approx axes-fraction per line

        if "right" in loc:
            x, ha = 1.0 - x_pad, "right"
        else:
            x, ha = x_pad, "left"
        if "upper" in loc:
            y0 = 1.0 - y_pad
            dy = -line_h
        else:
            y0 = y_pad + (n - 1) * line_h
            dy = -line_h

        # White background patch behind the whole block
        _max_chars = max(len(text) for text, _ in entries)
        _panel_w_pt = figsize[0] * 72.0 / n_panels
        block_w = min(_max_chars * fs * 0.30 / _panel_w_pt + 0.04, 0.90)
        pad_y = line_h * 0.25
        blk_h = n * line_h + 2.0 * pad_y
        blk_y = (y0 - n * line_h) - pad_y
        blk_x = (x - block_w) if "right" in loc else x
        ax.add_patch(mpatches.FancyBboxPatch(
            (blk_x, blk_y), block_w, blk_h,
            boxstyle="square,pad=0.0",
            facecolor="white", edgecolor="none",
            alpha=0.75, transform=ax.transAxes, zorder=4,
        ))

        for i, (text, col) in enumerate(entries):
            ax.text(
                x, y0 + i * dy, text,
                transform=ax.transAxes,
                fontsize=fs,
                ha=ha, va="top",
                color=col,
                fontfamily="monospace",
                clip_on=False,
                zorder=5,
            )

    def _fmt(value: float, sigma: float | None) -> str:
        return format_compact_uncertainty(
            value,
            sigma if (sigma is not None and sigma > 0) else None,
        )

    def _fmt_zeta(value: float, sigma: float | None) -> str:
        """Integer cm⁻¹ compact notation for ζ values: 530(85)."""
        v = int(round(value))
        if sigma is None or sigma <= 0:
            return str(v)
        return f"{v}({int(round(sigma))})"

    # --- 1-σ bands: propagate fit-parameter errors on z3_arr -----------------
    _errs = [_propagate_errors(float(g), 0.0) for g in z3_arr]
    sigma_g_ax_arr = np.array([e[0] for e in _errs])
    sigma_D_arr = np.array([e[1] for e in _errs])
    sigma_zeta_arr = np.array(
        [_zeta_sigma(float(g), 0.0) for g in z3_arr]
    )

    # ── Panel 1: Δg vs g_iso ─────────────────────────────────────────────────
    ax1.fill_between(
        z3_arr[_v3],
        (_ga3 - sigma_g_ax_arr)[_v3],
        (_ga3 + sigma_g_ax_arr)[_v3],
        color=col_curve, alpha=0.20, lw=0, zorder=2,
    )
    ax1.plot(
        z3_arr[_v3], _ga3[_v3],
        color=col_curve, lw=1.4, zorder=3,
    )

    p1_entries: list[tuple[str, str]] = []
    for label, g_ref, col, gi_err in [
        ("iso",    g_iso_intercept, col_intercept, g_iso_err_intercept),
        ("high-T", g_iso_high_t,    col_high_t,    g_iso_err_high_t),
    ]:
        g_ax_ref, _ = _point(g_ref)
        if np.isfinite(g_ax_ref):
            s_ga, _ = _propagate_errors(g_ref, gi_err)
            _L_arrows(ax1, g_ref, col, g_ax_ref)
            p1_entries.append(
                (rf"$g_\mathrm{{ax}}$({label}) = {_fmt(g_ax_ref, s_ga)}", col)
            )

    if g_iso_cross is not None:
        g_ax_cross, _ = _point(g_iso_cross)
        if np.isfinite(g_ax_cross):
            s_ga_c, _ = _propagate_errors(g_iso_cross, 0.0)
            _L_arrows(ax1, g_iso_cross, col_orca, g_ax_cross)
            p1_entries.append(
                (
                    rf"$g_\mathrm{{ax}}$($\zeta$) = "
                    rf"{_fmt(g_ax_cross, s_ga_c)}",
                    col_orca,
                )
            )

    if orca_point is not None:
        g_iso_o, g_ax_o, _ = orca_point
        ax1.plot(g_iso_o, g_ax_o, marker="*", color=col_orca,
                 ms=8, zorder=6, lw=0)

    if evans_g_iso is not None:
        g_ax_ev, _ = _point(evans_g_iso)
        if np.isfinite(g_ax_ev):
            s_ga_ev, _ = _propagate_errors(evans_g_iso, evans_g_iso_err)
            _L_arrows(ax1, evans_g_iso, col_evans, g_ax_ev)
            p1_entries.append((
                rf"$g_\mathrm{{ax}}$(Evans) = {_fmt(g_ax_ev, s_ga_ev)}",
                col_evans,
            ))

    if orca_point is not None:
        p1_entries.append((r"$\bigstar$ NEVPT2", col_orca))

    if p1_entries:
        _in_panel_legend(ax1, p1_entries, loc="upper right")

    ax1.set_xlim(_x_lo, _x_hi)
    ax1.set_xlabel(r"$g_\mathrm{iso}$", fontsize=label_fs)
    ax1.set_ylabel(r"$g_\mathrm{ax}$", fontsize=label_fs)
    ax1.tick_params(labelsize=tick_fs)

    # ── Panel 2: D vs g_iso ──────────────────────────────────────────────────
    D_valid = _v3 & np.isfinite(_D3_arr)
    ax2.fill_between(
        z3_arr[D_valid],
        (_D3_arr - sigma_D_arr)[D_valid],
        (_D3_arr + sigma_D_arr)[D_valid],
        color=col_curve, alpha=0.20, lw=0, zorder=2,
    )
    ax2.plot(
        z3_arr[D_valid], _D3_arr[D_valid],
        color=col_curve, lw=1.4, zorder=3,
    )

    p2_entries: list[tuple[str, str]] = []
    for label, g_ref, col, gi_err in [
        ("iso",    g_iso_intercept, col_intercept, g_iso_err_intercept),
        ("high-T", g_iso_high_t,    col_high_t,    g_iso_err_high_t),
    ]:
        _, D_ref = _point(g_ref)
        if np.isfinite(D_ref):
            _, s_D = _propagate_errors(g_ref, gi_err)
            _L_arrows(ax2, g_ref, col, D_ref)
            p2_entries.append(
                (rf"$D$({label}) = {_fmt(D_ref, s_D)}", col)
            )

    if g_iso_cross is not None:
        _, D_cross = _point(g_iso_cross)
        if np.isfinite(D_cross):
            _, s_D_c = _propagate_errors(g_iso_cross, 0.0)
            _L_arrows(ax2, g_iso_cross, col_orca, D_cross)
            p2_entries.append(
                (rf"$D$($\zeta$) = {_fmt(D_cross, s_D_c)}", col_orca)
            )

    if orca_point is not None:
        g_iso_o, _, D_o = orca_point
        ax2.plot(g_iso_o, D_o, marker="*", color=col_orca,
                 ms=8, zorder=6, lw=0)

    if evans_g_iso is not None:
        _, D_ev = _point(evans_g_iso)
        if np.isfinite(D_ev):
            _, s_D_ev = _propagate_errors(evans_g_iso, evans_g_iso_err)
            _L_arrows(ax2, evans_g_iso, col_evans, D_ev)
            p2_entries.append((
                rf"$D$(Evans) = {_fmt(D_ev, s_D_ev)}",
                col_evans,
            ))

    if orca_point is not None:
        p2_entries.append((r"$\bigstar$ NEVPT2", col_orca))

    if p2_entries:
        _in_panel_legend(ax2, p2_entries, loc="lower right", y_pad=0.10)

    ax2.set_xlim(_x_lo, _x_hi)
    ax2.set_xlabel(r"$g_\mathrm{iso}$", fontsize=label_fs)
    ax2.set_ylabel(r"$D$ (cm$^{-1}$)", fontsize=label_fs)
    ax2.tick_params(labelsize=tick_fs)

    # ── Panel 3: ζ̄_eff vs g_iso ─────────────────────────────────────────
    # 1-σ band for ζ (sigma_zeta_arr is already on z3_arr)
    ax3.fill_between(
        z3_arr[z3_valid],
        (z3_curve - sigma_zeta_arr)[z3_valid],
        (z3_curve + sigma_zeta_arr)[z3_valid],
        color=col_curve, alpha=0.20, lw=0, zorder=2,
    )
    ax3.plot(
        z3_arr[z3_valid], z3_curve[z3_valid],
        color=col_curve, lw=1.4, zorder=3,
    )

    if zeta_eff is not None:
        y_half = max(abs(float(zeta_eff)) * 0.5, 200.0)
        ax3.set_ylim(float(zeta_eff) - y_half, float(zeta_eff) + y_half)
        y_lo, y_hi = ax3.get_ylim()
        zeta_lo = zeta_eff * (1.0 - zeta_eff_tol)
        zeta_hi = zeta_eff * (1.0 + zeta_eff_tol)
        ax3.axhspan(zeta_lo, zeta_hi, color=col_zeta, alpha=0.18, zorder=1)
        ax3.axhline(zeta_eff, color=col_zeta, lw=1.2, ls=":", zorder=4)
    else:
        _finite_z3 = z3_curve[z3_valid]
        if len(_finite_z3) > 0:
            _z3_span = float(_finite_z3.max() - _finite_z3.min())
            _z3_pad = max(_z3_span * 0.10, 50.0)
            y_lo = float(_finite_z3.min()) - _z3_pad
            y_hi = float(_finite_z3.max()) + _z3_pad
        else:
            y_lo, y_hi = -500.0, 500.0
        ax3.set_ylim(y_lo, y_hi)

    p3_bot_entries: list[tuple[str, str]] = []
    if zeta_eff is not None:
        p3_bot_entries.append((
            rf"$\bar{{\zeta}}_\mathrm{{eff}}$ = "
            rf"{zeta_eff:.0f} cm$^{{-1}}$",
            col_zeta,
        ))

    p3_entries: list[tuple[str, str]] = []

    # Black/red references: L-arrows + ζ value label near y-axis
    for label, g_ref, col, gi_err in [
        ("iso",    g_iso_intercept, col_intercept, g_iso_err_intercept),
        ("high-T", g_iso_high_t,    col_high_t,    g_iso_err_high_t),
    ]:
        z_ref = _zeta_point(g_ref)
        sz = _zeta_sigma(g_ref, gi_err) if np.isfinite(z_ref) else 0.0
        if np.isfinite(z_ref) and y_lo <= z_ref <= y_hi:
            _L_arrows(ax3, g_ref, col, z_ref)
        p3_entries.append(
            (rf"$g_\mathrm{{iso}}$({label}) = {_fmt(g_ref, gi_err)}", col)
        )
        if np.isfinite(z_ref):
            p3_bot_entries.append(
                (
                    rf"$\bar{{\zeta}}$({label}) = "
                    rf"{_fmt_zeta(z_ref, sz)}",
                    col,
                )
            )

    # ζ crossing: L-arrows from y-axis → diamond → x-axis (green only)
    for g_cross in crossings:
        zeta_cross = _zeta_point(g_cross)
        if not np.isfinite(zeta_cross):
            continue
        x0_3, x1_3 = ax3.get_xlim()
        x_start3 = x0_3 + 0.02 * (x1_3 - x0_3)
        ap3 = dict(arrowstyle="-|>", color=col_orca, **_ap)
        ax3.annotate("", xy=(g_cross, zeta_cross),
                     xytext=(x_start3, zeta_cross),
                     arrowprops=ap3, annotation_clip=False)
        ax3.annotate("", xy=(g_cross, y_lo),
                     xytext=(g_cross, zeta_cross),
                     arrowprops=ap3, annotation_clip=False)
        ax3.plot(g_cross, zeta_cross, marker="D",
                 color=col_orca, ms=5, zorder=6, lw=0)
        sigma_g_cross = _cross_g_iso_sigma(g_cross)
        p3_entries.append((
            rf"$g_\mathrm{{iso}}$($\zeta$) = "
            rf"{_fmt(g_cross, sigma_g_cross)}",
            col_orca,
        ))

    if evans_g_iso is not None:
        z_ev = _zeta_point(evans_g_iso)
        sz_ev = (
            _zeta_sigma(evans_g_iso, evans_g_iso_err)
            if np.isfinite(z_ev) else 0.0
        )
        if np.isfinite(z_ev) and y_lo <= z_ev <= y_hi:
            _L_arrows(ax3, evans_g_iso, col_evans, z_ev)
        p3_entries.append((
            rf"$g_\mathrm{{iso}}$(Evans) = "
            rf"{_fmt(evans_g_iso, evans_g_iso_err)}",
            col_evans,
        ))
        if np.isfinite(z_ev):
            p3_bot_entries.append((
                rf"$\bar{{\zeta}}$(Evans) = {_fmt_zeta(z_ev, sz_ev)}",
                col_evans,
            ))

    if p3_entries:
        _in_panel_legend(ax3, p3_entries, loc="upper right")
    if p3_bot_entries:
        _in_panel_legend(ax3, p3_bot_entries, loc="lower left", y_pad=0.10)
    ax3.set_xlim(_x_lo, _x_hi)
    ax3.set_xlabel(r"$g_\mathrm{iso}$", fontsize=label_fs)
    ax3.set_ylabel(
        r"$\bar{\zeta}_\mathrm{eff} = -4S \cdot D\,/\,"
        r"g_\mathrm{ax}$ (cm$^{-1}$)",
        fontsize=label_fs,
    )
    ax3.tick_params(labelsize=tick_fs)

    # --- CSV export of all SH parameters with confidence intervals -----------
    if save:
        _rows: list[dict] = []
        for _label, _g_ref, _gi_err in [
            ("iso",    g_iso_intercept, g_iso_err_intercept),
            ("high_T", g_iso_high_t,    g_iso_err_high_t),
        ]:
            _ga, _D = _point(_g_ref)
            _s_ga, _s_D = _propagate_errors(_g_ref, _gi_err)
            _z = _zeta_point(_g_ref)
            _sz = _zeta_sigma(_g_ref, _gi_err) if np.isfinite(_z) else 0.0
            _rows.append({
                "point":       _label,
                "g_iso":       _g_ref,
                "g_iso_err":   _gi_err,
                "g_ax":        _ga,
                "g_ax_err":    _s_ga,
                "D_cm1":       _D,
                "D_cm1_err":   _s_D,
                "zeta_cm1":    _z,
                "zeta_cm1_err": _sz,
            })
        if g_iso_cross is not None:
            _ga_c, _D_c = _point(g_iso_cross)
            _s_ga_c, _s_D_c = _propagate_errors(g_iso_cross, 0.0)
            _s_gi_c = _cross_g_iso_sigma(g_iso_cross)
            _z_c = _zeta_point(g_iso_cross)
            _rows.append({
                "point":       "zeta_cross",
                "g_iso":       g_iso_cross,
                "g_iso_err":   _s_gi_c,
                "g_ax":        _ga_c,
                "g_ax_err":    _s_ga_c,
                "D_cm1":       _D_c,
                "D_cm1_err":   _s_D_c,
                "zeta_cm1":    _z_c,
                "zeta_cm1_err": 0.0,
            })
        if orca_point is not None:
            _gi_o, _ga_o, _D_o = orca_point
            _z_o = (
                -4.0 * J_spin * _D_o / _ga_o
                if abs(_ga_o) > 1e-10 else float("nan")
            )
            _rows.append({
                "point":       "nevpt2",
                "g_iso":       _gi_o,
                "g_iso_err":   float("nan"),
                "g_ax":        _ga_o,
                "g_ax_err":    float("nan"),
                "D_cm1":       _D_o,
                "D_cm1_err":   float("nan"),
                "zeta_cm1":    _z_o,
                "zeta_cm1_err": float("nan"),
            })
        if evans_g_iso is not None:
            _ga_ev, _D_ev = _point(evans_g_iso)
            _s_ga_ev, _s_D_ev = _propagate_errors(
                evans_g_iso, evans_g_iso_err
            )
            _z_ev2 = _zeta_point(evans_g_iso)
            _sz_ev2 = (
                _zeta_sigma(evans_g_iso, evans_g_iso_err)
                if np.isfinite(_z_ev2) else 0.0
            )
            _rows.append({
                "point":       "evans",
                "g_iso":       evans_g_iso,
                "g_iso_err":   evans_g_iso_err,
                "g_ax":        _ga_ev,
                "g_ax_err":    _s_ga_ev,
                "D_cm1":       _D_ev,
                "D_cm1_err":   _s_D_ev,
                "zeta_cm1":    _z_ev2,
                "zeta_cm1_err": _sz_ev2,
            })

        _csv_path = Path(save_name).with_suffix(".csv")
        _fields = [
            "point",
            "g_iso", "g_iso_err",
            "g_ax", "g_ax_err",
            "D_cm1", "D_cm1_err",
            "zeta_cm1", "zeta_cm1_err",
        ]
        with open(_csv_path, "w", newline="") as _fh:
            _fh.write(f"# hfc_file: {hfc_file}\n")
            _fh.write(
                f"# tip_correction: "
                f"{tip_correction if tip_correction is not None else 'N/A'}\n"
            )
            _writer = csv.DictWriter(_fh, fieldnames=_fields)
            _writer.writeheader()
            _writer.writerows(_rows)
        logger.info("SH parameters saved to %s", _csv_path)

    render_figure(fig, save=save, show=show, save_name=save_name)
    if save:
        logger.info("g_iso solution-line plot saved to %s", save_name)

    return fig, (ax1, ax2, ax3)
