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

from simpnmr.viz.layout.export import render_figure

logger = logging.getLogger(__name__)


def plot_isoaxrho(
    vals: dict,
    errs: dict,
    params: dict | None,
    inv_t: np.ndarray,
    show: bool = True,
    save: bool = True,
    y_label: str = "ChiT",
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
        - If `params` are provided, the function will only plot fit curves/bands if
          precomputed arrays are present under:
              params[component]["fit_y"]
              params[component]["fit_y_low"]
              params[component]["fit_y_high"]
    """
    # Early guard clause for empty vals
    if not vals:
        raise ValueError("plot_isoaxrho: no components provided in `vals`")

    for component in vals.keys():
        p = None if params is None else params.get(component)

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
            marker="o",
            ms=5,
            label="Exp",
        )

        # Optional: precomputed fit curve + precomputed uncertainty band
        caption_lines = []
        if p is not None:
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

            caption_lines = [rf"$R^2_\mathregular{{adj.}} = {_adj_r2_txt}$"]

            if "intercept" in p and "intercept_err" in p:
                caption_lines.append(
                    rf"$Intercept = {p['intercept']:.1f} \pm {p['intercept_err']:.1f}$"
                )
            elif "intercept" in p:
                caption_lines.append(rf"$Intercept = {p['intercept']:.1f}$")  # integer

            if "slope" in p and "slope_err" in p:
                caption_lines.append(
                    rf"$Slope = {p['slope']:.1f} \pm {p['slope_err']:.1f}$"  # integer
                )
            elif "slope" in p:
                caption_lines.append(rf"$Slope = {p['slope']:.1f}$")  # integer

            if "tip" in p:
                caption_lines.append(rf"$TIP = {p['tip']:.3g}$")  # move to e

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if np.isfinite(y_range) and y_range > 0:
            y_pad_frac = 0.20
            pad = y_pad_frac * y_range
            ax.set_ylim(y_min - pad, y_max + pad)

        # Move the caption annotation inside the main axis
        if caption_lines:
            y_box = 0.03
            va = "bottom"

            ax.annotate(
                " ".join(str(s) for s in caption_lines if s),
                xy=(0.97, y_box),
                xycoords="axes fraction",
                ha="right",
                va=va,
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
        ax.set_ylabel(f"{y_label} {component}", fontsize=14)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

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
            ncol=2,
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
