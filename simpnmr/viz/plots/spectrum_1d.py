# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot 1D NMR spectra from predicted and experimental data.

Provides utilities to build 1D spectra from per-nucleus shifts and to compare
predicted spectra with deconvoluted and raw experimental spectra.
"""

import copy
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from simpnmr.core.constants.gammas import NUCLEAR_GAMMAS
from simpnmr.core.domain.experiment import Experiment
from simpnmr.core.domain.molecule import Molecule
from simpnmr.core.spectrum.kernels import gaussian, lorentzian
from simpnmr.core.utils.arrays import find_index_of_nearest
from simpnmr.core.utils.strings import remove_numbers
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.utils.format import isotope_format

logger = logging.getLogger(__name__)


def plot_pred_spectrum(
    molecule: Molecule,
    isotope: str,
    shift_range: ArrayLike,
    save: bool = True,
    show: bool = True,
    save_name: str = "predicted_spectrum.png",
    window_title: str = "Predicted Spectrum",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots a predicted 1D spectrum from simulated shifts.

    Args:
        molecule: Molecule containing shift data.
        isotope: Isotope to plot (e.g. ``"1H"``).
        shift_range: Two-element sequence specifying min/max ppm.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Construct common ppm axis for the spectrum (x-axis)
    x_grid = np.linspace(np.min(shift_range), np.max(shift_range), 100000)

    # Construct spectrum intensities (y-axis)
    y_intensity = np.zeros(np.shape(x_grid))

    for nuc in molecule.nuclei:
        if nuc.isotope == isotope:
            y_intensity += lorentzian(x_grid, nuc.shift.lw, nuc.shift.avg, 1)

    # Normalise spectrum
    y_intensity /= np.max(y_intensity)

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    # Spectrum trace
    ax.plot(x_grid, y_intensity, color="k")

    # Labels
    avg_shifts = {
        nucleus.chem_math_label: nucleus.shift.avg
        for nucleus in molecule.nuclei
        if nucleus.isotope == isotope
    }

    # Ensure labels match shifts in sorted order
    sorted_shifts_labels = sorted(avg_shifts.items(), key=lambda x: x[1])
    sorted_labels = [label for label, _ in sorted_shifts_labels]
    sorted_shifts = [shift for _, shift in sorted_shifts_labels]

    # Grid y value closest to peak position
    closest_y = [y_intensity[find_index_of_nearest(x_grid, sh)] for sh in sorted_shifts]

    # Marker at shift peak position
    ax.plot(sorted_shifts, closest_y, lw=0, marker="x", color="k", markersize=7)

    # Draw text-label barrier 10% above the highest peak
    label_barrier = 1.1 * np.max(y_intensity)
    ax.hlines(
        label_barrier,
        np.min(shift_range),
        np.max(shift_range),
        linestyle="-",
        color="black",
        linewidth=0.8,
        alpha=0.7,
    )

    # Calculate initial distance matrix
    adj_label_xvals = copy.copy(sorted_shifts)
    distance = np.subtract.outer(adj_label_xvals, adj_label_xvals)
    np.fill_diagonal(distance, np.inf)

    # Define minimum acceptable distance between text-labels
    label_mindist = 0.03 * (np.max(shift_range) - np.min(shift_range))

    # Shift points until distance matrix has no values less than minimum dist
    while len(np.where(abs(distance) < label_mindist)[0]):
        [xlocs, ylocs] = np.where(abs(distance) < label_mindist)
        for x, y in zip(xlocs, ylocs):
            if y > x:
                adj_label_xvals[x] -= label_mindist / 2
                adj_label_xvals[y] += label_mindist / 2

        distance = np.subtract.outer(adj_label_xvals, adj_label_xvals)
        np.fill_diagonal(distance, np.inf)

    # Peak label y position (20% above max peak)
    label_y = 1.15 * np.max(y_intensity)

    # Add label and dashed lines
    for shift, label, label_x in zip(sorted_shifts, sorted_labels, adj_label_xvals):
        # Add label to plot
        ax.text(
            label_x,
            label_y,
            label,
            rotation="vertical",
            ha="center",
            va="bottom",
            fontsize="18",
        )

        # Draw segmented line from peak to label via horizontal line
        peak_index = find_index_of_nearest(x_grid, shift)
        ax.plot(
            [x_grid[peak_index], x_grid[peak_index], label_x],
            [y_intensity[peak_index], label_barrier, label_y],
            linestyle="--",
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

    ax.set_xlabel(r"{} $\delta$ (ppm)".format(isotope_format(isotope)), fontsize="18")

    # Deactivate borders, y axis and y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines[["right", "top", "left"]].set_visible(False)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlim([np.max(shift_range), np.min(shift_range)])

    fig.tight_layout()

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Predicted spectrum saved to %s", f"{save_name}.pdf")

    # Write spectrum (ppm and normalized intensity) to CSV for external visualization
    df = pd.DataFrame({"shift (ppm)": x_grid, "intensity (a.u.)": y_intensity})
    csv_path = os.path.join(
        os.path.dirname(save_name),
        f"shift_vs_intensity_{molecule.susc.temperature:.2f}_K.csv",
    )
    df.to_csv(csv_path, index=False)

    return fig, ax


def plot_raw_deconv_pred(
    molecule: Molecule,
    isotope: str,
    shift_range: ArrayLike,
    experiment: Experiment,
    save: bool = True,
    show: bool = True,
    save_name: str = "pred_and_exp_spectrum.png",
    window_title: str = "Raw, Deconvoluted, and Predicted Spectra",
    verbose: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes]]:
    """Plots raw, deconvoluted, and predicted spectra.

    Args:
        molecule: Molecule containing theoretical shift data.
        isotope: TODO
        shift_range: TODO
        experiment: Experiment containing the raw spectrum and deconvolution results.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Determine the number of subplots (include raw spectrum if available)
    n_subplots = 3 if experiment.spectrum is not None else 2

    # Construct common ppm axis for all spectra (x-axis)
    x_grid = np.linspace(np.min(shift_range), np.max(shift_range), 100000)

    # Construct simulated (predicted) spectrum intensities (y-axis)
    y_sim_intensity = np.zeros_like(x_grid)
    for nucleus in molecule.nuclei:
        if nucleus.isotope == isotope:
            y_sim_intensity += lorentzian(
                x_grid, nucleus.shift.lw, nucleus.shift.avg, 1
            )

    # Map each nucleus text-label to its simulated (predicted) peak position
    avg_shifts = {
        nucleus.chem_math_label: nucleus.shift.avg
        for nucleus in molecule.nuclei
        if nucleus.isotope == isotope
    }

    # Ensure nucleus text-label match simulated (predicted) shifts in sorted order
    sorted_shifts_labels = sorted(avg_shifts.items(), key=lambda x: x[1])
    labels = [label for label, _ in sorted_shifts_labels]
    shifts = [shift for _, shift in sorted_shifts_labels]

    # Extract simulated peak heights at the nearest grid points to each shift
    sim_peak_heights = [
        y_sim_intensity[find_index_of_nearest(x_grid, sh)] for sh in shifts
    ]

    # Construct deconvoluted (processed experimental) spectrum intensities (y-axis)
    y_deconv_intensity = np.zeros_like(x_grid)

    # Accumulate deconvoluted spectrum intensities
    for signal in experiment.signals:
        # Convert experimental linewidth from Hz to ppm
        exp_width_ppm = signal.width / (
            NUCLEAR_GAMMAS[remove_numbers(isotope)] * experiment.magnetic_field
        )
        # Add Lorentzian contribution
        y_deconv_intensity += signal.l_to_g * lorentzian(
            x_grid, exp_width_ppm, signal.shift, signal.area
        )
        # Add Gaussian contribution
        y_deconv_intensity += (1 - signal.l_to_g) * gaussian(
            x_grid, exp_width_ppm, signal.shift, signal.area
        )

    # Define plot space
    fig, ax = plt.subplots(
        n_subplots, 1, figsize=(8, 5.5), num=window_title, sharex=True
    )

    # SUBPLOT NUMBER 1 - Simulated spectrum with peak markers and nucleus text-labels

    ax[0].set_xlim(np.max(shift_range), np.min(shift_range))
    ax[0].plot(x_grid, y_sim_intensity, lw=1, color="k")
    ax[0].plot(shifts, sim_peak_heights, lw=0, marker="x", color="k")

    # Draw text-label barrier 10% above the highest simulated (predicted) peak
    label_barrier = 1.1 * np.max(y_sim_intensity)

    ax[0].hlines(
        label_barrier,
        np.min(shift_range),
        np.max(shift_range),
        linestyle="-",
        color="black",
        linewidth=0.5,
        alpha=0.7,
    )

    # Vertical position for peak text-labels (10% above the label barrier)
    labels_position_y = 1.05 * label_barrier

    # Define minimum acceptable distance between text-labels
    label_mindist = 0.03 * (np.max(shift_range) - np.min(shift_range))

    # Calculate initial distance matrix
    adj_label_xvals = copy.copy(shifts)
    distance = np.subtract.outer(adj_label_xvals, adj_label_xvals)
    np.fill_diagonal(distance, np.inf)

    # Shift points until distance matrix has no values less than minimum dist
    while len(np.where(abs(distance) < label_mindist)[0]):
        [xlocs, ylocs] = np.where(abs(distance) < label_mindist)
        for x, y in zip(xlocs, ylocs):
            if y > x:
                adj_label_xvals[x] -= label_mindist / 2
                adj_label_xvals[y] += label_mindist / 2

        distance = np.subtract.outer(adj_label_xvals, adj_label_xvals)
        np.fill_diagonal(distance, np.inf)

    for peak_x, peak_y, label_x, label in zip(
        shifts, sim_peak_heights, adj_label_xvals, labels
    ):
        # Add label
        ax[0].text(
            label_x,
            labels_position_y,
            label,
            fontsize="9",
            rotation="vertical",
            va="bottom",
            ha="center",
        )

        # Draw segmented line from peak to label via horizontal barrier
        ax[0].plot(
            [peak_x, peak_x, label_x],
            [peak_y, label_barrier, labels_position_y],
            linestyle="--",
            color="black",
            linewidth=0.7,
            alpha=0.4,
        )

    ax[0].set_title(
        "Simulation",
        loc="left",
        fontdict={"size": "smaller"},
        pad=-6,
    )

    # SUBPLOT NUMBER 2 - Deconvoluted (processed experimental) spectrum
    ax[1].plot(x_grid, y_deconv_intensity, lw=1, color="k")
    ax[1].set_title(
        "Paramagnetic Signals",
        loc="left",
        fontdict={"size": "smaller"},
        pad=-6,
    )

    # SUBPLOT NUMBER 3 - Raw experimental spectrum if available
    if n_subplots == 3:
        ax[2].plot(
            experiment.spectrum[:, 0],
            experiment.spectrum[:, 1],
            lw=1,
            color="k",
        )
        ax[2].set_title(
            "Full Spectrum",
            loc="left",
            fontdict={"size": "smaller"},
            pad=-6,
        )

    # Set x-axis at the bottom of the plot
    ax[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[-1].set_xlabel(r"{} $\delta$ (ppm)".format(isotope_format(isotope)))

    # Remove y-axis ticks, labels, and spines for a cleaner stacked-spectra layout
    for axis in ax:
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.spines[["right", "top", "left"]].set_visible(False)

    fig.tight_layout()

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Spectra saved to %s", f"{save_name}.pdf")

    return fig, ax
