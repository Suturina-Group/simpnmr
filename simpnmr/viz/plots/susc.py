# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot magnetic susceptibility tensor components.

Provides plotting utilities for chi_iso, chi_ax, and chi_rho trends versus
inverse temperature, with optional precomputed fit curves and uncertainty bands.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter

from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.glyphs import get_glyphs
from simpnmr.viz.style.theme import DEFAULT_SIZE
from simpnmr.viz.style.typography import apply_typography

logger = logging.getLogger(__name__)
glyphs = get_glyphs(DEFAULT_SIZE)


def plot_isoaxrho(
    vals: dict,
    errs: dict,
    params: dict,
    inv_t: np.ndarray,
    show: bool = True,
    save: bool = True,
    save_name: str = "susceptibility_components",
    verbose: bool = True,
) -> None:
    """Plots iso/ax/rho susceptibility components vs x-values.

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
        raise ValueError("plot_isoaxrho: no components provided in `vals`")

    _chiT_label_map = {
        "iso": r"\mathrm{iso}",
        "ax": r"\mathrm{ax}",
        "rho": r"\mathrm{rh}",
    }

    for component in vals.keys():
        p = params[component]

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))

        # Experimental values with error bars (markers only)
        ax.errorbar(
            inv_t,
            vals[component],
            yerr=errs[component],
            lw=0,
            elinewidth=glyphs.elinewidth,
            color="black",
            capsize=glyphs.capsize,
            marker=glyphs.marker,
            markeredgecolor=glyphs.mec,
            ms=glyphs.ms,
            label="Exp.",
        )

        # Optional: plot experimental values without TIP contribution
        tip = p.get("tip", 0.0)
        if np.isfinite(tip) and abs(float(tip)) > 0.0:
            ax.errorbar(
                inv_t,
                vals[component] - (tip / inv_t),
                lw=0,
                elinewidth=glyphs.elinewidth,
                color="#bdbdbd",
                alpha=glyphs.series_alpha_muted,
                capsize=glyphs.capsize,
                marker=glyphs.marker,
                markeredgecolor=glyphs.mec,
                ms=glyphs.ms,
                label="Exp. w/o TIP",
            )

        # Optional: precomputed fit curve + precomputed uncertainty band
        caption_lines = []
        fit_y = p.get("fit_y")
        fit_y_low = p.get("fit_y_low")
        fit_y_high = p.get("fit_y_high")
        if fit_y is not None:
            ax.plot(
                inv_t,
                fit_y,
                linestyle="-",
                linewidth=glyphs.fit_lw,
                color="black",
                label="Slope/Intercept Fit",
            )

        if fit_y_low is not None and fit_y_high is not None:
            ax.fill_between(
                inv_t,
                fit_y_low,
                fit_y_high,
                alpha=glyphs.band_alpha,
                linewidth=glyphs.band_lw,
            )

        # Caption panel: only display values already present in params
        _adj_r2 = p.get("adj_r2")
        if _adj_r2 is None or np.isnan(_adj_r2):
            _adj_r2_txt = "N/A"
        else:
            _adj_r2_txt = f"{_adj_r2:.3f}"

        caption_lines = [rf"$R^{{2}}_{{\mathrm{{adj}}}} = {_adj_r2_txt}$"]

        intercept = p.get("intercept")
        intercept_err = p.get("intercept_err")
        if intercept is not None and intercept_err is not None:
            caption_lines.append(
                rf"$\mathrm{{Intercept}} = {intercept:.1f} \pm {intercept_err:.1f}$"
            )
        elif intercept is not None:
            caption_lines.append(rf"$\mathrm{{Intercept}} = {intercept:.1f}$")

        slope = p.get("slope")
        slope_err = p.get("slope_err")
        if slope is not None and slope_err is not None:
            caption_lines.append(
                rf"$\mathrm{{Slope}} = {slope:.1f} \pm {slope_err:.1f}$"
            )
        elif slope is not None:
            caption_lines.append(rf"$\mathrm{{Slope}} = {slope:.1f}$")

        tip_txt = p.get("tip")
        if tip_txt is not None:
            caption_lines.append(rf"$\mathrm{{TIP}} = {float(tip_txt):.1e}$")

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if np.isfinite(y_range) and y_range > 0:
            y_pad_frac = 0.20
            pad = y_pad_frac * y_range
            ax.set_ylim(y_min - pad, y_max + pad)

        # Axis labels/styling
        ax.set_xlabel(r"$1/T$ (K$^{-1})$")
        chi_sub = _chiT_label_map.get(component, component)
        ax.set_ylabel(rf"$\chi T^{{\mathrm{{red}}}}_{{{chi_sub}}}$")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0e}"))

        # Add 10% padding on x-axis (inverse temperature)
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        if np.isfinite(x_range) and x_range > 0:
            pad = 0.10 * x_range
            ax.set_xlim(x_min - pad, x_max + pad)

        # Secondary top axis for T(K): uses axis transform only
        def _inv_to_t(inv: float | np.ndarray) -> float | np.ndarray:
            inv_arr = np.asarray(inv, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = 1.0 / inv_arr
            return out

        def _t_to_inv(t: float | np.ndarray) -> float | np.ndarray:
            t_arr = np.asarray(t, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = 1.0 / t_arr
            return out

        top_ax = ax.secondary_xaxis("top", functions=(_inv_to_t, _t_to_inv))
        top_ax.set_xlabel(r"$T$ (K)")
        top_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Typography (centralised): apply after axes + secondary axes exist.
        scale = apply_typography(ax, size=DEFAULT_SIZE)
        apply_typography(top_ax, size=DEFAULT_SIZE)

        # Move the caption annotation inside the main axis
        if caption_lines:
            ax.annotate(
                " ".join(str(s) for s in caption_lines if s),
                xy=(0.98, 0.03),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=scale.annotation,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="black",
                    lw=1.0,
                ),
            )

        # Legend styling (white background + black border)
        leg = ax.legend(
            loc="upper left",
            ncol=3,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            fontsize=scale.legend,
            columnspacing=1.2,
            handletextpad=0.6,
            borderpad=0.6,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.0)

        fig.tight_layout()

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
    inv_t: np.ndarray,
    ab_series: dict,
    show: bool = True,
    save: bool = True,
    save_name: str = "exp_vs_ab_initio_susc",
    verbose: bool = True,
) -> None:
    """Plots fitted chiT model vs ab initio chiT components (iso/ax/rho).

    Args:
        params: Fitted chiT model parameters keyed by component (iso/ax/rho). Each
            entry must include a precomputed array under key "fit_y" evaluated on
            `inv_t`.
        inv_t: Inverse-temperature grid for the fitted model.
        ab_series: Ab initio chiT series matched to `inv_t`, with keys: "inv_t",
            "iso", "ax", "rho".
    """
    _chiT_label_map = {
        "iso": r"\mathrm{iso}",
        "ax": r"\mathrm{ax}",
        "rho": r"\mathrm{rh}",
    }

    for component in params.keys():
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))

        p_exp = params[component]
        fit_y = p_exp.get("fit_y")
        if fit_y is None:
            raise ValueError(
                f"plot_exp_vs_ab_initio: missing params[{component!r}]['fit_y']"
            )
        y_fit = np.asarray(fit_y, dtype=float)

        ax.plot(
            inv_t,
            y_fit,
            lw=0,
            color="#E69F00",  # orange
            marker=glyphs.marker,
            markeredgecolor=glyphs.mec,
            ms=glyphs.ms,
            label="pNMR",
        )

        # Ab initio data (matched to the experimental grid upstream; may contain NaNs)
        y_ab = np.asarray(ab_series[component], dtype=float)
        m_ab = np.isfinite(y_ab)
        if np.any(m_ab):
            ax.errorbar(
                inv_t[m_ab],
                y_ab[m_ab],
                yerr=None,
                lw=0,
                elinewidth=glyphs.elinewidth,
                color="#1f77b4",  # muted blue
                capsize=glyphs.capsize,
                marker=glyphs.marker,
                markeredgecolor=glyphs.mec,
                ms=glyphs.ms,
                label="Ab initio",
            )

        # Axis labels/styling
        ax.set_xlabel(r"$1/T$ (K$^{-1})$")
        chi_sub = _chiT_label_map.get(component, component)
        ax.set_ylabel(rf"$\chi T^{{\mathrm{{red}}}}_{{{chi_sub}}}$")

        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0e}"))

        # Add 10% padding on y-axis
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if np.isfinite(y_range) and y_range > 0:
            pad = 0.20 * y_range
            ax.set_ylim(y_min - pad, y_max + pad)

        # Secondary top axis: T(K)
        def _inv_to_t(inv: float | np.ndarray) -> float | np.ndarray:
            inv_arr = np.asarray(inv, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / inv_arr

        def _t_to_inv(t: float | np.ndarray) -> float | np.ndarray:
            t_arr = np.asarray(t, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / t_arr

        top_ax = ax.secondary_xaxis("top", functions=(_inv_to_t, _t_to_inv))
        top_ax.set_xlabel(r"$T$ (K)")
        top_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Typography (centralised): apply after axes + secondary axes exist.
        scale = apply_typography(ax, size="standard")
        apply_typography(top_ax, size="standard")

        # Legend
        leg = ax.legend(
            loc="upper left",
            ncol=2,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            fontsize=scale.legend,
            columnspacing=1.2,
            handletextpad=0.6,
            borderpad=0.6,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.0)

        fig.tight_layout()

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
