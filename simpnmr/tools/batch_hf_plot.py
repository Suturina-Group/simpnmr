# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot hyperfine coupling data from multiple QC sources.

Loads hyperfine data from multiple calculation outputs and generates comparison
plots across sources.
"""

import argparse
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import yaml
import yaml_include
from matplotlib.lines import Line2D

from simpnmr.application.loaders.chem_labels import load_chem_labels_from_csv
from simpnmr.application.loaders.molecule import load_molecule_from_qca
from simpnmr.core.domain.molecule import Molecule
from simpnmr.viz.style.palette import SAFE_COLOURS

mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

plt.rcParams["font.family"] = "Arial"

yaml.add_constructor("!inc", yaml_include.Constructor(base_dir="."))


def load_hyperfine_data(
    sources: dict[str, str], chem_labels: str, elements="H"
) -> dict[str, Molecule]:
    """
    Load hyperfine coupling data from multiple sources and build Molecule objects.

    For each entry in `sources`, the function reads a quantum-chemistry output file,
    constructs a `Molecule`, applies unit conversion (via the Molecule factory),
    and attaches chemical labels from `chem_labels`.

    Args:
        sources (dict[str, str]): Mapping from a source name (e.g. a functional) to
            the corresponding input file path.
        chem_labels (str): Path to a CSV file containing chemical labels (and
            optionally math labels) for atoms.
        elements (str | list[str]): Elements to include. Use "all" to include all
            elements.

    Returns:
        dict[str, Molecule]: Mapping from source name to a populated Molecule
        instance for that source.
    """

    all_molecules = dict.fromkeys(sources, None)

    for source_name, source_file in sources.items():
        molecule = load_molecule_from_qca(
            source_file,
            elements=elements,
            converter=None,
        )

        al_to_cl, al_to_cml = load_chem_labels_from_csv(chem_labels)
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

        all_molecules[source_name] = molecule

    return all_molecules


def plot_component(
    func_comps: dict[str, dict[str, float]],
    ylabel: str,
    show: bool = True,
    save: bool = True,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    savename: str = "hyperfines.png",
    figure_title: str = "Hyperfine coupling constants",
):
    """
    Plot a bar chart comparing a single hyperfine component across sources.

    Args:
        func_comps (dict[str, dict[str, float]]): Mapping from source name to a
            mapping of atom label -> value to plot.
        ylabel (str): Y-axis label.
        show (bool): If True, display the plot window.
        save (bool): If True, save the figure to `savename`.
        fig (plt.Figure | None): Existing figure to plot into. If None, a new
            figure is created.
        ax (plt.Axes | None): Existing axes to plot into. If None, new axes are
            created.
        savename (str): Output filename for the saved figure.
        figure_title (str): Figure title used for the matplotlib window.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes containing the plot.
    """

    if None in [fig, ax]:
        fig, ax = plt.subplots(1, 1, num=figure_title, figsize=[5.5, 3.5])

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(func_comps) + 1)
    shifts = [width + width * it for it in range(len(func_comps))]

    _frst = list(func_comps.keys())[0]
    xvals = np.arange(1, len(func_comps[_frst]) + 1)

    for (functional, a_vals), shift in zip(func_comps.items(), shifts):
        if functional == "pdip":
            ax.bar(
                xvals + shift,
                a_vals.values(),
                width=width,
                label="Point Dipole",
                color="k",
            )
        else:
            ax.bar(xvals + shift, a_vals.values(), width=width, label=functional)

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xticks(xvals + 0.5)
    ax.set_xticklabels(func_comps[_frst].keys(), rotation=45)
    ax.grid(axis="x", ls="--", which="minor")
    ax.set_xlim(0.5, len(func_comps[_frst]) + 1.5)
    ax.xaxis.set_tick_params("major", length=0)

    ax.hlines(0, 0.5, len(func_comps[_frst]) + 1.5, lw=0.5, color="k")

    fig.legend(loc=7, frameon=False)
    ax.set_ylabel(ylabel, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.73)

    if save:
        plt.savefig(f"{savename}", dpi=500, transparent=True)
    if show:
        plt.show()

    return fig, ax


def plot_normalisation(
    norms: dict[str, float],
    chemlabels: dict[str, float],
    save=True,
    show=True,
    savename="normalisation.png",
    figure_title="Normalisation",
):
    """
    Plot normalisation (max |A_iso|) values across sources and color by label.

    Args:
        norms (dict[str, float]): Mapping from source name to the max absolute
            isotropic value used for normalisation.
        chemlabels (dict[str, str]): Mapping from source name to the label
            corresponding to the max absolute isotropic value.
        save (bool): If True, save the figure to `savename`.
        show (bool): If True, display the plot window.
        savename (str): Output filename for the saved figure.
        figure_title (str): Figure title used for the matplotlib window.

    Returns:
        None
    """

    unilabs = set(chemlabels.values())

    colours = {lab: col for col, lab in zip(SAFE_COLOURS, unilabs)}

    fig, ax = plt.subplots(num=figure_title)

    for name, value in norms.items():
        print(name, value)

    for it, (key, val) in enumerate(norms.items()):
        ax.plot(it, val, lw=0, marker="x", mew=2.0, color=colours[chemlabels[key]])

    legend_elements = [
        Line2D([0], [0], marker="x", color=colour, label=label, mew=2, lw=0)
        for label, colour in colours.items()
    ]

    ax.legend(handles=legend_elements)

    ax.set_xticks(np.arange(len(norms)))
    ax.set_xticklabels(norms.keys(), rotation=45)

    ax.set_ylabel(r"$A_\mathregular{iso, max} (\mathregular{MHz})$", fontsize=12)

    fig.tight_layout()

    if save:
        plt.savefig(f"{savename}", dpi=500)
    if show:
        plt.show()

    return


def main():
    parser = argparse.ArgumentParser(
        description=("This script allows you to plot multiple sets of Hyperfine data"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_file", type=str, help=".csv file containing source file information"
    )

    parser.add_argument(
        "chem_labels",
        type=str,
        help=".csv file containing chemical labels of each atom",
    )

    parser.add_argument(
        "-w", "--window_append", type=str, help="Appends to window titles"
    )

    parser.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default="all",
        help=("Elements to include in plot"),
    )

    uargs = parser.parse_args()

    # Load input file
    config = pd.read_csv(
        uargs.input_file, skip_blank_lines=True, skipinitialspace=True, comment="#"
    )

    sources = {name: file for name, file in zip(config["name"], config["input_file"])}

    molecules = load_hyperfine_data(
        sources, chem_labels=uargs.chem_labels, elements=uargs.elements
    )

    all_isos = {
        name: {nuc.chem_math_label: nuc.A.iso for nuc in molecule.nuclei}
        for name, molecule in molecules.items()
    }

    # for name, valdict in all_isos.items():
    #     all_isos[name] = dict(
    #         sorted(valdict.items(), key=lambda item: item[1], reverse=True)
    #     )

    # Isotropic parts
    plot_component(
        all_isos,
        r"$A_\mathregular{iso} \mathregular{(MHz)}$",
        figure_title=uargs.window_append,
        savename="isotropic.png",
    )

    # Isotropic parts relative to largest value for that functional

    all_relative_isos = copy.deepcopy(all_isos)
    norm_vals = dict.fromkeys(all_isos, 0.0)
    norm_clabs = dict.fromkeys(all_isos, "")

    for name, relative_isos in all_isos.items():
        for lab in relative_isos.keys():
            all_relative_isos[name][lab] /= np.max(np.abs(list(relative_isos.values())))
            norm_vals[name] = np.max(np.abs(list(relative_isos.values())))
            norm_clabs[name] = list(relative_isos.keys())[
                np.argmax(np.abs(list(relative_isos.values())))
            ]

    for name, valdict in all_relative_isos.items():
        all_relative_isos[name] = dict(
            sorted(valdict.items(), key=lambda item: item[1])
        )

    plot_normalisation(norm_vals, norm_clabs, figure_title=uargs.window_append)

    plot_component(
        all_relative_isos,
        r"$A_\mathregular{iso}$ / $A_\mathregular{iso, max}$",
        figure_title=uargs.window_append,
    )

    plot_component(
        all_relative_isos,
        r"$A_\mathregular{iso}$ / $A_\mathregular{iso, max}$",
        figure_title=uargs.window_append,
    )

    all_ax = {
        name: {
            nuc.chem_math_label: nuc.A.dip[0, 0] - nuc.A.dip[1, 1]
            for nuc in molecule.nuclei
        }
        for name, molecule in molecules.items()
    }

    all_rho = {
        name: {
            nuc.chem_math_label: -nuc.A.dip[0, 0] - nuc.A.dip[1, 1]
            for nuc in molecule.nuclei
        }
        for name, molecule in molecules.items()
    }

    plot_component(
        all_ax,
        r"$A_\mathregular{ax} \mathregular{(MHz)}$",
        figure_title=uargs.window_append,
        savename="axial.png",
    )

    plot_component(
        all_rho,
        r"$A_\mathregular{rho} \mathregular{(MHz)}$",
        figure_title=uargs.window_append,
        savename="rhombic.png",
    )
