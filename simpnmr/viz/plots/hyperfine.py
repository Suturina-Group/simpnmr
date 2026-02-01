# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot hyperfine coupling data.

Provides bar and violin plot utilities for selected hyperfine tensor components.
"""

import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from simpnmr.core.domain.molecule import Nucleus
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.utils.tensor_components import comp2ind

logger = logging.getLogger(__name__)


def plot_hyperfine(
    nuclei: list[Nucleus],
    components: list[str],
    save: bool = False,
    show: bool = True,
    save_name: str = "hyperfines.dat",
    verbose: bool = False,
    window_title: str = "Hyperfine data",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots selected hyperfine-tensor components for a list of nuclei.

    Args:
        nuclei: Nuclei to plot.
        components: Names of hyperfine components to plot (e.g. ``"xx"``, ``"yy"``,
            ``"iso"``, ``"ax"``, ``"rho"``, ``"dxy"``).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        verbose: If ``True``, prints the output file name when saving.
        window_title: Figure window title.

    Returns:
        A tuple ``(fig, ax)``.
    """

    complabels = {component: "" for component in components}

    hf_components = {
        component: {nuc.label: [] for nuc in nuclei} for component in components
    }

    for component in components:
        if component == "iso":
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.iso
                complabels[component] = r"$A_\mathregular{iso}$"
        elif component == "ax":
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[0, 0] - nuc.A.dip[1, 1]
                complabels[component] = r"$A_\mathregular{dip, ax}$"
        elif component == "rho":
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[0, 0] + nuc.A.dip[1, 1]
                complabels[component] = r"$A_\mathregular{dip, rho}$"
        elif "d" in component:
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[comp2ind(component[1:])]
                complabels[component] = (
                    rf"$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$"
                )
        elif component in ["x", "y", "z"]:  # eigenvalues
            _to_ind = {"x": 0, "y": 1, "z": 2}
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.eigvals[_to_ind[component]]
                complabels[component] = rf"$A_{{\mathregular{{{component}}}}}$"
        else:
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.tensor[comp2ind(component)]
                complabels[component] = rf"$A_\mathregular{{{component}}}$"

    fig, ax = plt.subplots(1, 1, num=window_title)

    n_nuclei = len(nuclei)

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(components) + 1)
    shifts = [width + width * it for it in range(n_nuclei)]

    # Tick positions
    xvals = np.arange(1, n_nuclei + 1)

    unique_nuclabels = np.unique([nuc.label for nuc in nuclei])

    xvals = np.arange(1, len(unique_nuclabels) + 1)

    for (comp_name, comp_values), shift in zip(hf_components.items(), shifts):
        ax.bar(
            xvals + shift,
            list(comp_values.values()),
            width=width,
            label=complabels[comp_name],
        )

    if len(hf_components) < 11:
        step = 1
    else:
        step = 2

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::step] + 0.5)
    labels = list(unique_nuclabels)
    ax.set_xticklabels(labels[::step])
    ax.grid(axis="x", ls="--", which="minor")
    ax.set_xlim(0.5, len(labels) + 1.5)
    ax.xaxis.set_tick_params("major", length=0)

    ax.hlines(0, 0.5, len(labels) + 1.5, lw=0.5, color="k")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend()

    ax.set_ylabel(r"Hyperfine Coupling (ppm Å$^\mathregular{-3}$)")
    fig.tight_layout()

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Hyperfine plot saved to %s", f"{save_name}.pdf")

    return fig, ax


def plot_hyperfine_iso_vs_ax(
    value_dict: dict[str, float],
    order: list[int],
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    symbol="x",
    save: bool = False,
    show: bool = True,
    save_name: str = "hyperfines.dat",
    verbose: bool = False,
    window_title: str = "hyperfine data",
):
    if all([fig is None, ax is None]):
        fig, ax = plt.subplots(num=window_title)

    vals = list(value_dict.values())

    ax.plot([vals[o] for o in order], lw=0, marker=symbol, fillstyle="none", color="C1")

    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(value_dict))))
    labels = [label for label in value_dict.keys()]
    ax.set_xticklabels([labels[o] for o in order])

    ax.set_ylabel(
        r"$A_\mathregular{iso} / (A_\mathregular{dip_{xx}} + A_\mathregular{dip_{yy}})$"
    )

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Hyperfine plot saved to %s", f"{save_name}.pdf")

    return


def plot_hyperfine_spread(
    nuclei: list[Nucleus],
    components: list[str] | None = None,
    save: bool = False,
    show: bool = True,
    save_name: str = "hyperfines.png",
    window_title: str = "Hyperfine Components",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots the spread of hyperfine components for each chemical label.

    Args:
        nuclei: Nuclei to plot.
        components: Names of hyperfine components to plot (e.g. ``"xx"``, ``"yy"``,
            ``"iso"``, ``"ax"``, ``"rho"``, ``"dxy"``).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    if components is None:
        components = []

    legend_labels = {component: "" for component in components}

    a_comps = {
        component: {nuc.chem_math_label: [] for nuc in nuclei}
        for component in components
    }

    for component in components:
        if component == "iso":
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(nuc.A.iso)
                legend_labels[component] = r"$A_\mathregular{iso}$"
        elif component == "ax":
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.dip[0, 0] - nuc.A.dip[1, 1]
                )
                legend_labels[component] = r"$A_\mathregular{dip, ax}$"
        elif component == "rho":
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.dip[0, 0] + nuc.A.dip[1, 1]
                )
                legend_labels[component] = r"$A_\mathregular{dip, rho}$"
        elif "d" in component:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.dip[comp2ind(component[1:])]
                )
                legend_labels[component] = (
                    rf"$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$"
                )
        else:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.tensor[comp2ind(component)]
                )
                legend_labels[component] = rf"$A_\mathregular{{{component}}}$"

    unique_chemlabels = []
    for nuc in nuclei:
        if nuc.chem_math_label not in unique_chemlabels:
            unique_chemlabels.append(nuc.chem_math_label)

    fig, ax = plt.subplots(1, 1, num=window_title)

    xvals = np.arange(1, len(unique_chemlabels) + 1)

    legend_markers = []
    for comp_values in a_comps.values():
        _violin = ax.violinplot(
            dataset=list(comp_values.values()),
            positions=xvals + 0.5,
            vert=True,
            showmeans=True,
        )
        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )

    if len(a_comps) < 11:
        step = 1
    else:
        step = 2

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::step] + 0.5)
    labels = list(unique_chemlabels)
    ax.set_xticklabels(labels[::step])
    ax.grid(axis="x", ls="--", which="minor")
    ax.set_xlim(0.5, len(labels) + 1.5)
    ax.xaxis.set_tick_params("major", length=0)

    ax.hlines(0, 0.5, len(labels) + 1.5, lw=0.5, color="k")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend(legend_markers, [legend_labels[comp] for comp in a_comps.keys()])

    ax.set_ylabel(r"Hyperfine Coupling (ppm Å$^\mathregular{-3}$)")
    fig.tight_layout()

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Hyperfine spread plot saved to %s", f"{save_name}.pdf")

    return fig, ax
