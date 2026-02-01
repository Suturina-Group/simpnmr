# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot fitted susceptibility metrics across multiple sources.

Reads per-source susceptibility results and generates comparison plots for
chi_iso, chi_ax, chi_rho, and selected fit quality metrics.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_susceptibility_data(
    sources: dict[str, str], index: int
) -> dict[str, np.ndarray]:
    """
    Load susceptibility data for a given component index across multiple sources.

    Each source file is expected to be a whitespace-delimited table readable by
    `numpy.loadtxt`. The function loads each file (skipping the first row) and
    returns the selected row at `index`.

    Args:
        sources (dict[str, str]): Mapping from source name to the corresponding input
            file path.
        index (int): Row index to extract from each loaded table.

    Returns:
        dict[str, np.ndarray]: Mapping from source name to the extracted row values.
    """

    all_molecules = dict.fromkeys(sources, None)

    for source_name, source_file in sources.items():
        # Load quantum chemical hyperfine data
        _data = np.loadtxt(source_file, skiprows=1)

        all_molecules[source_name] = _data[index]

    return all_molecules


def plot_component(
    func_comps: dict[str, float],
    ylabel: str,
    show: bool = True,
    save: bool = True,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    savename: str = "hyperfines.png",
    figure_title: str = "Hyperfine coupling constants",
):
    """
    Plot a simple comparison chart for a scalar susceptibility metric across sources.

    Args:
        func_comps (dict[str, float]): Mapping from source name to the scalar value
            to plot.
        ylabel (str): Y-axis label.
        show (bool): If True, display the plot window.
        save (bool): If True, save the figure to `savename`.
        fig (plt.Figure | None): Existing figure to plot into. If None, a new figure
            is created.
        ax (plt.Axes | None): Existing axes to plot into. If None, new axes are
            created.
        savename (str): Output filename for the saved figure.
        figure_title (str): Figure title used for the matplotlib window.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes containing the plot.
    """

    if None in [fig, ax]:
        fig, ax = plt.subplots(1, 1, num=figure_title)

    ax.plot(func_comps.values(), lw=0, marker="x", color="k")

    ax.set_xticks(np.arange(len(func_comps)))
    ax.set_xticklabels(func_comps.keys(), rotation=45)

    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if save:
        plt.savefig(f"{savename}", dpi=500)
    if show:
        plt.show()

    return fig, ax


def main() -> None:
    """
    Parse CLI arguments and plot susceptibility metrics across sources.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description=("This script allows you to plot multiple sets of Hyperfine data"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_file", type=str, help=".csv file containing source file information"
    )

    parser.add_argument(
        "-w", "--window_append", type=str, help="Appends to window titles", default=""
    )

    uargs = parser.parse_args()

    # Load input file
    config = pd.read_csv(
        uargs.input_file, skip_blank_lines=True, skipinitialspace=True, comment="#"
    )

    # Make table each functional name and chi, r2a, and resid
    table = {}
    for name, folder in zip(config["name"], config["folder"]):
        # Load susceptibility data
        _susc = pd.read_csv(
            os.path.join(folder, "susceptibility_tensor.csv"),
            skip_blank_lines=True,
            skipinitialspace=True,
            comment="#",
        )

        table[name] = {
            "chi_iso": _susc["chi_iso (Å^3)"][0],
            "chi_ax": _susc["chi_ax (Å^3)"][0],
            "chi_rho": _susc["chi_rho (Å^3)"][0],
            "r2_adjusted": _susc["r2_adjusted ()"][0],
            "MAE": _susc["MAE (ppm)"][0],
        }

    # Isotropic parts
    plot_component(
        {name: val["chi_iso"] for name, val in table.items()},
        r"$\chi_\mathregular{iso} \mathregular{(\AA^{3})}$",
        show=False,
        figure_title="isotropic susceptibility " + uargs.window_append,
    )

    # Ax parts
    plot_component(
        {name: val["chi_ax"] for name, val in table.items()},
        r"$\Delta\chi_\mathregular{ax} \mathregular{(\AA^{3})}$",
        show=False,
        figure_title="axial susceptibility " + uargs.window_append,
    )

    # rhombic parts
    plot_component(
        {name: val["chi_rho"] for name, val in table.items()},
        r"$\Delta\chi_\mathregular{rho} \mathregular{(\AA^{3})}$",
        show=False,
        figure_title="rhombic susceptibility " + uargs.window_append,
    )

    plt.show()
