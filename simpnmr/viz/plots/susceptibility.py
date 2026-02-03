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

logger = logging.getLogger(__name__)


def plot_isoaxrho(
    vals: dict,
    errs: dict,
    params: dict,
    inv_t: np.ndarray,
    show: bool = True,
    save: bool = True,
    save_name: str = "susceptibility_components",
    window_title: str = "Isotropic, Axial, and Rhombic susceptibilities",
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

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(7.0, 5.0),
            num=f"{window_title} — {component}",
        )

        # Experimental values with error bars (markers only)
        ax.errorbar(
            inv_t,
            vals[component],
            yerr=errs[component],
            lw=0,
            elinewidth=1.5,
            color="black",
            capsize=1.5,
            marker="s",
            ms=5,
            label="Exp.",
        )

        # Optional: plot experimental values without TIP contribution
        tip = p.get("tip", 0.0)
        if np.isfinite(tip) and abs(float(tip)) > 0.0:
            ax.errorbar(
                inv_t,
                vals[component] - (tip / inv_t),
                lw=0,
                elinewidth=1.5,
                color="#bdbdbd",
                alpha=0.65,
                capsize=1.5,
                marker="s",
                markeredgecolor="none",
                ms=5,
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
                linewidth=1.5,
                color="black",
                label="Slope/Intercept Fit",
            )

        if fit_y_low is not None and fit_y_high is not None:
            ax.fill_between(
                inv_t,
                fit_y_low,
                fit_y_high,
                alpha=0.15,
                linewidth=0,
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

        # Move the caption annotation inside the main axis
        if caption_lines:
            ax.annotate(
                " ".join(str(s) for s in caption_lines if s),
                xy=(0.98, 0.03),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="black",
                    lw=1.0,
                ),
            )

        # Axis labels/styling
        ax.set_xlabel(r"$1/T$ (K$^{-1})$", fontsize=14)
        chi_sub = _chiT_label_map.get(component, component)
        ax.set_ylabel(
            rf"$\chi T^{{\mathrm{{red}}}}_{{{chi_sub}}}$",
            fontsize=14,
        )
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0e}"))

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
        top_ax.set_xlabel(r"$T$ (K)", fontsize=14)
        top_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Legend styling (white background + black border)
        leg = ax.legend(
            loc="upper left",
            ncol=3,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            fontsize="10",
            columnspacing=1.2,
            handletextpad=0.6,
            borderpad=0.6,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.0)

        ax.tick_params(axis="both", labelsize=12)
        top_ax.tick_params(axis="x", labelsize=12)

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


# def plot_exp_ab_initio_comparison(
#     ab_initio_params: dict,
#     params: dict,
#     inv_t: np.ndarray,
#     show: bool = True,
#     save: bool = True,
#     y_label: str = "ChiT",
#     save_name: str = "susceptibility_components",
#     window_title: str = "Isotropic, Axial, and Rhombic susceptibilities",
#     verbose: bool = True,
# ) -> None:
#     """Plots
#     """
