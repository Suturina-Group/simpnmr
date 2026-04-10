# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot magnetic susceptibility tensor components.

Provides plotting utilities for chi_iso, chi_ax, and chi_rh trends versus
inverse temperature, with optional precomputed fit curves and uncertainty bands.
"""

import logging

import matplotlib.ticker as ticker
import numpy as np

from simpnmr.viz.layout.canvas import create_canvas, create_header_plot_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.table import render_compact_table
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.uncertainty import format_compact_uncertainty

logger = logging.getLogger(__name__)


def plot_isoaxrh(
    vals: dict,
    errs: dict,
    params: dict,
    inv_t: np.ndarray,
    spec: PlotSpec,
    show: bool = True,
    save: bool = True,
    save_name: str = "susceptibility_components",
    verbose: bool = True,
) -> None:
    """Plots iso/ax/rh susceptibility components vs x-values.

    Notes:
        - This function is intentionally "dumb": it only visualizes arrays that are
          passed in and does not evaluate fit models.
        - `temperatures` are treated as x-values. In the current pipeline they are
          expected to be inverse temperatures (1/T) prepared upstream.
        - `params` are required, and the function will plot fit curves/bands only if
          the precomputed arrays are present under:
              params[component]["fit_y"]
              params[component]["fit_y_low"]
              params[component]["fit_y_high"]
    """

    # Early guard clause for empty vals
    if not vals:
        raise ValueError("plot_isoaxrh: no components provided in `vals`")

    glyphs = spec.glyphs

    _chiT_label_map = {
        "iso": r"\mathrm{iso}",
        "ax": r"\mathrm{ax}",
        "rh": r"\mathrm{rh}",
    }

    for component in vals.keys():
        p = params[component]
        palette = spec.palette

        fig, header_ax, ax = create_header_plot_canvas(
            spec.profile,
            variant="vertical",
            layout="constrained",
            header_ratio=0.90,
            plot_ratio=3.10,
            hspace=0.02,
        )
        fig.patch.set_facecolor("white")
        header_ax.set_facecolor("white")
        ax.set_facecolor("white")

        inv_t_plot = inv_t * 1.0e3

        # Experimental values with error bars
        ax.errorbar(
            inv_t_plot,
            vals[component],
            yerr=errs[component],
            lw=0,
            elinewidth=glyphs.elinewidth,
            color=palette.primary,
            capsize=glyphs.capsize,
            marker=glyphs.marker,
            markeredgecolor=glyphs.mec,
            ms=glyphs.ms,
            label="Exp.",
        )

        # Plot experimental values without TIP contribution
        tip = p.get("tip", 0.0)
        if np.isfinite(tip) and abs(float(tip)) > 0.0:
            ax.errorbar(
                inv_t_plot,
                vals[component] - (tip / inv_t),
                lw=0,
                elinewidth=glyphs.elinewidth,
                color=palette.highlight,
                alpha=glyphs.series_alpha_muted,
                capsize=glyphs.capsize,
                marker=glyphs.marker,
                markeredgecolor=glyphs.mec,
                ms=glyphs.ms,
                label="Exp. w/o TIP",
            )

        # Precomputed fit curve + precomputed uncertainty band
        fit_y = p.get("fit_y")
        fit_y_low = p.get("fit_y_low")
        fit_y_high = p.get("fit_y_high")
        if fit_y is not None:
            ax.plot(
                inv_t_plot,
                fit_y,
                linestyle="-",
                linewidth=glyphs.fit_lw,
                color=palette.primary,
                label="Fit.",
            )

        if fit_y_low is not None and fit_y_high is not None:
            ax.fill_between(
                inv_t_plot,
                fit_y_low,
                fit_y_high,
                color=spec.palette.primary,
                alpha=glyphs.band_alpha,
                linewidth=glyphs.band_lw,
            )

        # Compact summary strip: flatten all display items into the current
        # two-column table contract.
        _adj_r2 = p.get("adj_r2")
        if _adj_r2 is None or np.isnan(_adj_r2):
            _adj_r2_txt = "N/A"
        else:
            _adj_r2_txt = f"{_adj_r2:.3f}"

        fit_quality_items = [rf"adj. $R^{{2}}$: {_adj_r2_txt}"]

        model_items: list[str] = []

        intercept = p.get("intercept")
        intercept_err = p.get("intercept_err")
        if intercept is not None and intercept_err is not None and intercept_err > 0:
            model_items.append(
                f"Intercept: {format_compact_uncertainty(intercept, intercept_err)}"
            )
        elif intercept is not None:
            model_items.append(f"Intercept: {intercept:.1f}")

        slope = p.get("slope")
        slope_err = p.get("slope_err")
        if slope is not None and slope_err is not None and slope_err > 0:
            model_items.append(f"Slope: {format_compact_uncertainty(slope, slope_err)}")
        elif slope is not None:
            model_items.append(f"Slope: {slope:.1f}")

        tip_val = p.get("tip")
        if tip_val is not None and tip_val != 0.0:
            exp = int(np.floor(np.log10(abs(tip_val))))
            mant = tip_val / 10**exp
            model_items.append(rf"TIP: ${mant:.1f} \times 10^{{{exp}}}$")

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if np.isfinite(y_range) and y_range > 0:
            y_pad_frac = 0.10
            pad = y_pad_frac * y_range
            ax.set_ylim(y_min - pad, y_max + pad)

        # Axis labels/styling
        ax.set_xlabel(r"$1/T\;10^{3}$ (K$^{-1}$)")
        chi_sub = _chiT_label_map.get(component, component)
        ax.set_ylabel(rf"$\Delta\chi T^{{\mathrm{{red}}}}_{{{chi_sub}}}$")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Subtle grid for readability (major + minor)
        ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.25)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.2, alpha=0.15)

        # Secondary top axis for T(K): uses axis transform only
        def _inv_to_t(inv_plot: float | np.ndarray) -> float | np.ndarray:
            inv_arr = np.asarray(inv_plot, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = 1.0e3 / inv_arr
            return out

        def _t_to_inv(t: float | np.ndarray) -> float | np.ndarray:
            t_arr = np.asarray(t, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = 1.0e3 / t_arr
            return out

        top_ax = ax.secondary_xaxis("top", functions=(_inv_to_t, _t_to_inv))
        top_ax.set_xlabel(r"$T$ (K)")
        top_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Typography (centralised): apply after axes + secondary axes exist.
        spec.skin_axes(ax)
        spec.skin_axes(top_ax)

        render_compact_table(
            header_ax,
            blocks=[
                ("Fit", fit_quality_items),
                ("Model", model_items),
            ],
            spec=spec,
        )
        ax.legend(loc="best", ncol=1)

        comp_save_name = f"{save_name}_{component}"

        render_figure(
            fig,
            save=save,
            show=show,
            save_name=comp_save_name,
        )

        if save and verbose:
            logger.info(
                "Temperature dependence plot saved to %s",
                f"{comp_save_name}.pdf",
            )


def plot_exp_vs_ab_initio(
    params: dict,
    g_sq: dict[str, float],
    inv_t: np.ndarray,
    ab_series: dict[str, np.ndarray],
    analytic_chi_vt: dict[str, np.ndarray],
    spec: PlotSpec,
    show: bool = True,
    save: bool = True,
    save_name: str = "exp_vs_ab_initio_susc",
    verbose: bool = True,
) -> None:
    """Plots fitted chiT model vs ab initio chiT components (iso/ax/rh).

    Args:
        params: Fitted chiT model parameters keyed by component (iso/ax/rh). Each
            entry must include a precomputed array under key "fit_y" evaluated on
            `inv_t`.
        inv_t: Inverse-temperature grid for the fitted model.
        ab_series: Ab initio chiT series on the ab initio grid, with keys: "inv_t",
            "iso", "ax", "rh". All values are arrays on the ab initio grid.
        analytic_chi_vt: Analytic VT 2nd order susceptibility,
        as arrays on the ab initio grid for each component.
    """
    _chiT_label_map = {
        "iso": r"\mathrm{iso}",
        "ax": r"\mathrm{ax}",
        "rh": r"\mathrm{rh}",
    }

    comp_to_gsq_key = {
        "iso": "g_sq_iso",
        "ax": "g_sq_ax",
        "rh": "g_sq_rh",
    }

    for component in params.keys():
        inv_t_plot = inv_t * 1.0e3

        fig, ax = create_canvas(
            spec.profile,
            variant="standard",
            layout="constrained",
        )

        p_exp = params[component]
        fit_y = p_exp.get("fit_y")
        if fit_y is None:
            raise ValueError(
                f"plot_exp_vs_ab_initio: missing params[{component!r}]['fit_y']"
            )
        y_fit = np.asarray(fit_y, dtype=float)

        # Fitted data using SimpNMR
        ax.plot(
            inv_t_plot,
            y_fit,
            color=spec.palette.primary,
            linestyle="-",
            linewidth=spec.glyphs.aux_lw,
            marker="s",
            markersize=spec.glyphs.ms,
            label="pNMR",
        )

        inv_t_ab = np.asarray(ab_series["inv_t"], dtype=float)
        vt_arr = np.asarray(analytic_chi_vt[component], dtype=float)

        if vt_arr.shape != inv_t_ab.shape:
            raise ValueError(
                "plot_exp_vs_ab_initio: analytic_chi_vt[%r] shape %r does not match "
                "ab_series['inv_t'] shape %r"
                % (component, vt_arr.shape, inv_t_ab.shape)
            )

        # Convert analytic chi(T) to chiT on the ab initio temperature grid.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ab = 1.0 / inv_t_ab
        x_vt = inv_t_ab * 1.0e3
        y_vt = vt_arr * t_ab

        m_vt = np.isfinite(y_vt)
        if np.any(m_vt):
            ax.plot(
                x_vt[m_vt],
                y_vt[m_vt],
                color=spec.palette.reference,
                linestyle="-",
                linewidth=spec.glyphs.aux_lw,
                label="VT 2nd order",
            )

        # g^2 series (scalar Y value on the pNMR x-grid)
        gsq_key = comp_to_gsq_key.get(component)
        if gsq_key is not None and gsq_key in g_sq:
            gsq_val = float(g_sq[gsq_key])
            y_gsq = np.full_like(inv_t_plot, gsq_val, dtype=float)
            ax.plot(
                inv_t_plot,
                y_gsq,
                linestyle="-",
                linewidth=spec.glyphs.aux_lw,
                color=spec.palette.secondary,
                label=rf"$g^{{2}}_{{{_chiT_label_map.get(component, component)}}}$",
            )

        # Ab initio data (matched to the experimental grid upstream)
        x_ab = np.asarray(ab_series["inv_t"], dtype=float) * 1.0e3
        y_ab = np.asarray(ab_series[component], dtype=float)
        m_ab = np.isfinite(y_ab)
        if np.any(m_ab):
            ax.plot(
                x_ab[m_ab],
                y_ab[m_ab],
                linestyle="-",
                linewidth=spec.glyphs.aux_lw,
                color=spec.palette.auxiliary,
                label="Ab initio",
            )

        # Axis labels/styling
        ax.set_xlabel(r"$1/T\;10^{3}$ (K$^{-1}$)")
        chi_sub = _chiT_label_map.get(component, component)
        ax.set_ylabel(rf"$\Delta\chi T^{{\mathrm{{red}}}}_{{{chi_sub}}}$")

        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Subtle grid for readability (major + minor)
        ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.25)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.2, alpha=0.15)

        # Add asymmetric padding on y-axis
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if np.isfinite(y_range) and y_range > 0:
            lower_pad = 0.10 * y_range
            upper_pad = 0.40 * y_range
            ax.set_ylim(y_min - lower_pad, y_max + upper_pad)

        # Secondary top axis: T(K)
        def _inv_to_t(inv_plot: float | np.ndarray) -> float | np.ndarray:
            inv_arr = np.asarray(inv_plot, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0e3 / inv_arr

        def _t_to_inv(t: float | np.ndarray) -> float | np.ndarray:
            t_arr = np.asarray(t, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0e3 / t_arr

        top_ax = ax.secondary_xaxis("top", functions=(_inv_to_t, _t_to_inv))
        top_ax.set_xlabel(r"$T$ (K)")
        top_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Typography (centralised): apply after axes + secondary axes exist.
        spec.skin_axes(ax)
        spec.skin_axes(top_ax)

        ax.legend(loc="upper left", ncol=2)

        comp_save_name = f"{save_name}_{component}"

        render_figure(
            fig,
            save=save,
            show=show,
            save_name=comp_save_name,
        )

        if save and verbose:
            logger.info(
                "Experiment vs Ab Initio comparison plot saved to %s",
                f"{comp_save_name}.pdf",
            )
