# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot 1D NMR spectra from predicted and experimental data.

Provides utilities to build 1D spectra from per-nucleus shifts and to compare
predicted spectra with deconvoluted and raw experimental spectra.
"""

import copy
import logging
import os
from collections.abc import Mapping

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy.typing import ArrayLike

from simpnmr.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.spectrum.kernels import gaussian, lorentzian
from simpnmr.core.util.arrays import find_index_of_nearest
from simpnmr.core.util.strings import remove_numbers
from simpnmr.io.csv.spec import write_spectrum
from simpnmr.viz.layout.canvas import create_stacked_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.fmt import isotope_format

logger = logging.getLogger(__name__)


def _get_plot_linewidth(nucleus, linewidths_by_label):
    if linewidths_by_label is not None and nucleus.label in linewidths_by_label:
        return linewidths_by_label[nucleus.label]
    if nucleus.shift.lw is None:
        raise ValueError("Spectrum plotting requires linewidth values")
    return nucleus.shift.lw


def plot_pred_spectrum(
    molecule: Molecule,
    isotope: str,
    shift_range: ArrayLike,
    spec: PlotSpec,
    effective_linewidths_by_label: Mapping[str, float] | None = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "predicted_spectrum.pdf",
    window_title: str = "Predicted Spectrum",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a predicted 1D spectrum from simulated shifts.

    Args:
        molecule: Molecule containing shift data.
        isotope: Isotope to plot (e.g. ``"1H"``).
        shift_range: Two-element sequence specifying min/max ppm.
        spec: Plot styling contract.
        effective_linewidths_by_label: Optional per-nucleus linewidths in ppm
            resolved by the application pipeline.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Add extra 10% padding for better visibility
    extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]

    shift_range = [
        shift_range[0] + np.negative(np.max(extras)),
        shift_range[1] + np.positive(np.max(extras)),
    ]

    # Construct common ppm axis for the spectrum (x-axis)
    x_grid = np.linspace(np.min(shift_range), np.max(shift_range), 100000)

    # Construct spectrum intensities (y-axis)
    y_intensity = np.zeros(np.shape(x_grid))

    for nuc in molecule.nuclei:
        if nuc.isotope == isotope:
            y_intensity += lorentzian(
                x_grid,
                _get_plot_linewidth(nuc, effective_linewidths_by_label),
                nuc.shift.avg,
                1,
            )

    # Normalise spectrum
    y_intensity /= np.max(y_intensity)

    glyphs = spec.glyphs
    palette = spec.palette

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

    # Scale figure width and label density with number of peaks
    n_peaks = len(sorted_labels)
    base_width = 3.54  # inches (standard)
    fig_width = max(base_width, base_width * n_peaks / 8)
    base_height = 2.40
    label_mindist = max(0.005, 0.03 / max(1, n_peaks / 8))
    label_fontsize = max(4, round(spec.typography.label * min(1.0, 8 / max(1, n_peaks))))

    # Make plot
    fig, ax = plt.subplots(
        1, 1,
        figsize=(fig_width, base_height),
        num=window_title,
        layout="constrained",
    )
    spec.skin_axes(ax)

    # Spectrum trace
    ax.plot(x_grid, y_intensity, color=palette.primary, lw=glyphs.line_lw * 0.75)

    _annotate_peaks_with_barrier(
        ax,
        x_grid=x_grid,
        y_intensity=y_intensity,
        peak_x=sorted_shifts,
        labels=sorted_labels,
        shift_range=shift_range,
        spec=spec,
        palette=palette,
        glyphs=glyphs,
        reverse_axis=True,
        connector_alpha=0.6,
        label_fontsize=str(label_fontsize),
        label_mindist_scale=label_mindist,
        line_scale=0.4,
    )

    ax.set_xlabel(r"{} $\delta$ (ppm)".format(isotope_format(isotope)))

    # Deactivate borders, y axis and y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines[["right", "top", "left"]].set_visible(False)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlim([np.max(shift_range), np.min(shift_range)])

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Predicted spectrum saved to %s", f"{save_name}.pdf")

    # Write spectrum (ppm and normalized intensity) to CSV for external visualization
    csv_path = os.path.join(
        os.path.dirname(save_name),
        f"shift_vs_intensity_{molecule.susc.temperature:.2f}_K.csv",
    )

    write_spectrum(csv_path, x_grid, y_intensity)

    return fig, ax


def plot_raw_deconv_pred(
    molecule: Molecule,
    isotope: str,
    shift_range: ArrayLike,
    experiment: Experiment,
    spec: PlotSpec,
    effective_linewidths_by_label: Mapping[str, float] | None = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "pred_and_exp_spectrum.pdf",
    window_title: str = "Raw, Deconvoluted, and Predicted Spectra",
    verbose: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plots raw, deconvoluted, and predicted spectra.

    Args:
        molecule: Molecule containing theoretical shift data.
        isotope: Isotope to plot (e.g. ``"1H"``).
        shift_range: Two-element sequence specifying the initial min/max ppm.
            The final plotting window is expanded to include the experimental
            peak range with additional padding.
        experiment: Experiment containing the raw spectrum and deconvolution results.
        spec: Plot styling contract.
        effective_linewidths_by_label: Optional per-nucleus linewidths in ppm
            resolved by the application pipeline.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Always use two subplots: Simulation (top) and Experiment (bottom)
    n_subplots = 2

    # Use the union of simulation and experimental peak ranges to avoid clipping.
    exp_min = min(s.shift for s in experiment.signals)
    exp_max = max(s.shift for s in experiment.signals)
    range_min = min(shift_range[0], exp_min)
    range_max = max(shift_range[1], exp_max)

    # Add extra 10% padding for better visibility
    pad = 0.1 * max(abs(exp_min), abs(exp_max))
    shift_range = [range_min - pad, range_max + pad]

    # Construct common ppm axis for all spectra (x-axis)
    x_grid = np.linspace(np.min(shift_range), np.max(shift_range), 100000)

    # Construct simulated (predicted) spectrum intensities (y-axis)
    y_sim_intensity = np.zeros_like(x_grid)
    for nucleus in molecule.nuclei:
        if nucleus.isotope == isotope:
            y_sim_intensity += lorentzian(
                x_grid,
                _get_plot_linewidth(nucleus, effective_linewidths_by_label),
                nucleus.shift.avg,
                1,
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

    glyphs = spec.glyphs
    palette = spec.palette

    # Define plot space
    fig, ax = create_stacked_canvas(
        spec.profile,
        nrows=n_subplots,
        variant="vertical",
        window_title=window_title,
        layout="constrained",
        sharex=True,
    )
    for axis in ax:
        spec.skin_axes(axis)

    # SUBPLOT NUMBER 1 - Simulated spectrum with peak markers and nucleus text-labels
    ax[0].set_xlim(np.max(shift_range), np.min(shift_range))
    ax[0].plot(x_grid, y_sim_intensity, lw=0.8 * glyphs.line_lw, color=palette.primary)
    ax[0].plot(
        shifts,
        sim_peak_heights,
        lw=0,
        color=palette.primary,
        markersize=glyphs.ms,
    )
    _annotate_peaks_with_barrier(
        ax[0],
        x_grid=x_grid,
        y_intensity=y_sim_intensity,
        peak_x=shifts,
        labels=labels,
        shift_range=shift_range,
        spec=spec,
        palette=palette,
        glyphs=glyphs,
        reverse_axis=True,
        label_fontsize=round(spec.typography.label * 0.6),
        line_scale=0.8,
    )

    # Vertical left-side label
    ax[0].text(
        -0.0,
        0.5,
        "Simulation",
        transform=ax[0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=str(spec.typography.annotation - 1),
        clip_on=False,
    )

    # SUBPLOT NUMBER 2 - Deconvoluted (processed experimental) spectrum
    ax[1].plot(
        x_grid,
        y_deconv_intensity,
        lw=0.8 * glyphs.line_lw,
        color=palette.primary,
        alpha=0.7,
    )
    # Overlay raw experimental spectrum if available.
    # If `experiment.exp_reference` is provided (ppm), normalize the raw spectrum
    # so it overlays the deconvoluted spectrum using the strongest deconv peak
    # within ±1 ppm of the reference.
    if experiment.spectrum is not None:
        x_raw = np.asarray(experiment.spectrum[:, 0], dtype=float)
        y_raw = np.asarray(experiment.spectrum[:, 1], dtype=float)

        exp_ref = getattr(experiment, "exp_reference", None)

        if exp_ref is not None:
            exp_ref = float(exp_ref)

            tol_ppm = 1.0

            # Reference height from deconvoluted spectrum: max within [ref - 1, ref + 1]
            m_deconv = (x_grid >= exp_ref - tol_ppm) & (x_grid <= exp_ref + tol_ppm)
            if np.any(m_deconv):
                ref_y_deconv = float(np.max(y_deconv_intensity[m_deconv]))
            else:
                ref_y_deconv = float(np.max(y_deconv_intensity))

            # Reference height from raw spectrum: max within [ref - 1, ref + 1]
            m_raw = (x_raw >= exp_ref - tol_ppm) & (x_raw <= exp_ref + tol_ppm)
            if np.any(m_raw):
                ref_y_raw = float(np.max(y_raw[m_raw]))
            else:
                ref_y_raw = float(np.max(y_raw))

            # Scale raw to match deconvoluted reference height (guard against zeros)
            if ref_y_raw > 0.0:
                scale = ref_y_deconv / ref_y_raw
            else:
                scale = 1.0

            y_raw = y_raw * scale

            # Clip extreme solvent peaks: after normalization, cap raw intensity
            # to the global maximum of the deconvoluted spectrum.
            deconv_max = float(np.max(y_deconv_intensity))
            y_raw = np.clip(y_raw, a_min=None, a_max=deconv_max)

        ax[1].plot(
            x_raw,
            y_raw,
            lw=0.8 * glyphs.line_lw,
            color=palette.highlight,
            alpha=0.35,
        )

    # Try to match exp. to the same LaTeX labels used for the simulated spectrum
    latex_label_map: dict[str, str] = {}
    for nucleus in molecule.nuclei:
        if nucleus.isotope != isotope:
            continue

        # Prefer the plain chemical label if available.
        plain = getattr(nucleus, "chem_label", None)
        latex = getattr(nucleus, "chem_math_label", None)
        if plain and latex:
            latex_label_map[str(plain)] = str(latex)

    def _map_assignment_to_latex(assignment: str) -> str:
        """Map an experimental assignment string to LaTeX labels if possible.

        Supports comma-separated assignments (e.g. "H1,H2"). If no mapping is
        found, the original token is preserved.
        """

        if assignment is None:
            return ""

        tokens = [t.strip() for t in str(assignment).split(",")]
        mapped: list[str] = []
        for tok in tokens:
            mapped.append(latex_label_map.get(tok, tok))
        return ",".join(mapped)

    pm_sorted = sorted(
        [
            (signal.shift, _map_assignment_to_latex(signal.assignment))
            for signal in experiment.signals
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    pm_shifts = [sh for sh, _ in pm_sorted]
    pm_labels = [lab for _, lab in pm_sorted]

    _annotate_peaks_with_barrier(
        ax[1],
        x_grid=x_grid,
        y_intensity=y_deconv_intensity,
        peak_x=pm_shifts,
        labels=pm_labels,
        shift_range=shift_range,
        spec=spec,
        palette=palette,
        glyphs=glyphs,
        reverse_axis=True,
        label_fontsize=round(spec.typography.label * 0.6),
    )

    # Vertical left-side label (instead of a top title)
    ax[1].text(
        0.0,
        0.5,
        "Experiment",
        transform=ax[1].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=str(spec.typography.annotation - 1),
        clip_on=False,
    )

    # Set x-axis at the bottom of the plot
    ax[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[-1].set_xlabel(r"{} $\delta$ (ppm)".format(isotope_format(isotope)))

    # Remove y-axis ticks, labels, and spines for a cleaner stacked-spectra layout
    for axis in ax:
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.spines[["right", "top", "left"]].set_visible(False)

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Spectra saved to %s", f"{save_name}.pdf")

    return fig, ax


def _annotate_peaks_with_barrier(
    ax: plt.Axes,
    *,
    x_grid: np.ndarray,
    y_intensity: np.ndarray,
    peak_x: list[float],
    labels: list[str],
    shift_range: ArrayLike,
    spec: PlotSpec,
    palette,
    glyphs,
    reverse_axis: bool = True,
    barrier_scale: float = 1.1,
    labels_above_barrier_scale: float = 1.05,
    label_mindist_scale: float = 0.03,
    connector_alpha: float = 0.4,
    barrier_alpha: float = 0.7,
    label_fontsize: str | None = None,
    line_scale: float = 1.0,
) -> None:
    """Annotate a spectrum with a horizontal barrier, vertical labels, and connectors.

    The function also resolves label overlaps by shifting label x-positions and
    enforces a monotonic label ordering to prevent connector crossings.

    Args:
        ax: Axis to annotate.
        x_grid: Common x grid for the spectrum.
        y_intensity: Spectrum intensity values on `x_grid`.
        peak_x: Peak x-positions (ppm) to annotate.
        labels: Text labels for each peak.
        shift_range: Two-element sequence specifying min/max ppm.
        spec: Plot specification.
        palette: Plot palette.
        glyphs: Plot glyph specification.
        reverse_axis: If True, treat the x-axis as visually
        reversed (high ppm on the left).
        barrier_scale: Scale factor for the barrier y-position
        relative to max intensity.
        labels_above_barrier_scale: Scale factor for label
        y-position relative to barrier.
        label_mindist_scale: Minimum x-separation between
        labels as a fraction of x-range.
        connector_alpha: Alpha for connector lines.
        barrier_alpha: Alpha for the horizontal barrier line.
    """

    if len(peak_x) == 0:
        return

    # Sort peaks so labels are placed in a consistent visual order.
    order = np.argsort(peak_x)
    if reverse_axis:
        order = order[::-1]

    peak_x_sorted = [peak_x[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    # Peak heights at nearest grid points
    peak_y_sorted = [
        y_intensity[find_index_of_nearest(x_grid, sh)] for sh in peak_x_sorted
    ]

    # Horizontal barrier and label positions in data coordinates
    label_barrier = barrier_scale * float(np.max(y_intensity))
    labels_position_y = labels_above_barrier_scale * label_barrier
    _y_top = labels_position_y * 1.5
    _lw_connector = max(0.2, 0.2 * line_scale * glyphs.line_lw)
    _lw_barrier = max(0.2, 0.5 * line_scale * glyphs.line_lw)
    _fs = label_fontsize or str(spec.typography.label)

    # Draw the static barrier line
    ax.hlines(
        label_barrier,
        np.min(shift_range),
        np.max(shift_range),
        linestyle="-",
        color=palette.primary,
        linewidth=_lw_barrier,
        alpha=barrier_alpha,
    )
    ax.set_ylim(bottom=None, top=_y_top)

    # Container for dynamic annotation artists (cleared on each redraw)
    _annotation_artists = []

    def _redraw_labels(ax_):
        """Redraw labels and connectors for peaks visible in current x range."""
        for artist in _annotation_artists:
            try:
                artist.remove()
            except Exception:
                pass
        _annotation_artists.clear()

        xmin, xmax = ax_.get_xlim()
        # Clamp y-top
        _, ymax = ax_.get_ylim()
        if ymax < _y_top:
            ax_.set_ylim(top=_y_top)

        # Select only peaks visible in current x window
        visible = [
            (px, py, lab)
            for px, py, lab in zip(peak_x_sorted, peak_y_sorted, labels_sorted)
            if min(xmin, xmax) <= px <= max(xmin, xmax)
        ]
        if not visible:
            ax_.figure.canvas.draw_idle()
            return

        vis_px, vis_py, vis_labs = zip(*visible)
        vis_px = list(vis_px)

        # Resolve label overlaps within the visible window
        xrange = abs(xmax - xmin)
        mindist = label_mindist_scale * xrange
        adj = list(vis_px)
        dist = np.subtract.outer(adj, adj)
        np.fill_diagonal(dist, np.inf)
        for _ in range(200):
            overlap = np.where(abs(dist) < mindist)
            if not len(overlap[0]):
                break
            for xi, yi in zip(*overlap):
                if yi > xi:
                    adj[xi] -= mindist / 2
                    adj[yi] += mindist / 2
            dist = np.subtract.outer(adj, adj)
            np.fill_diagonal(dist, np.inf)
        adj = sorted(adj, reverse=reverse_axis)

        for px, py, lx, lab in zip(vis_px, vis_py, adj, vis_labs):
            t = ax_.text(
                lx, labels_position_y, lab,
                fontsize=_fs, rotation="vertical",
                va="bottom", ha="center",
                clip_on=False,
            )
            ln, = ax_.plot(
                [px, px, lx],
                [py, label_barrier, labels_position_y],
                linestyle="--",
                color=palette.primary,
                linewidth=_lw_connector,
                alpha=connector_alpha,
                clip_on=False,
            )
            _annotation_artists.extend([t, ln])

    ax.callbacks.connect("xlim_changed", _redraw_labels)
    ax.callbacks.connect("ylim_changed", _redraw_labels)

    # Initial draw
    _redraw_labels(ax)
