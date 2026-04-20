# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot 1D NMR spectra from predicted and experimental data.

Provides utilities to build 1D spectra from per-nucleus shifts and to compare
predicted spectra with deconvoluted and raw experimental spectra.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np
from numpy.typing import ArrayLike

from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.spectrum.kernels import gaussian, lorentzian
from simpnmr.core.util.arrays import find_index_of_nearest

from simpnmr.io.csv.spec import write_spectrum
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.figure import get_figsize
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

    # Labels — math labels for PDF rendering, plain labels for CSV metadata
    avg_shifts_math = {
        nucleus.chem_math_label: nucleus.shift.avg
        for nucleus in molecule.nuclei
        if nucleus.isotope == isotope
    }
    # plain chem_label keyed separately so CSV lookup matches peak_data_*.csv
    math_to_plain = {
        nucleus.chem_math_label: nucleus.chem_label
        for nucleus in molecule.nuclei
        if nucleus.isotope == isotope
    }

    # Ensure labels match shifts in sorted order
    sorted_shifts_labels = sorted(avg_shifts_math.items(), key=lambda x: x[1])
    sorted_labels = [label for label, _ in sorted_shifts_labels]   # math
    sorted_shifts = [shift for _, shift in sorted_shifts_labels]
    sorted_plain_labels = [math_to_plain[lbl] for lbl in sorted_labels]

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

    # Write spectrum CSV with full metadata so the GUI can reconstruct
    # annotations without needing a Molecule object.
    # Derive name from save_name so multi-isotope runs don't overwrite each
    # other (save_name already carries the isotope suffix, e.g. _1H).
    _stem = Path(save_name).name.replace("pred_spectrum", "shift_vs_intensity")
    csv_path = os.path.join(os.path.dirname(save_name), f"{_stem}.csv")
    write_spectrum(
        csv_path, x_grid, y_intensity,
        isotope=isotope,
        temperature=molecule.susc.temperature,
        peak_labels=sorted_plain_labels,
        peak_shifts=sorted_shifts,
    )

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
    label_colors: dict | None = None,
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

    # Two rows: Simulation (top) and Experiment (bottom)

    # Chem-labels that belong to the requested isotope (for signal filtering)
    _iso_chem_labels = {
        nuc.chem_label
        for nuc in molecule.nuclei
        if nuc.isotope == isotope
    }
    # All isotopes present in the molecule
    _all_isotopes = {nuc.isotope for nuc in molecule.nuclei}
    _single_isotope = len(_all_isotopes) == 1

    # Signals for the deconvoluted spectrum.
    # Priority: use signal.isotope field when present; fall back to
    # assignment-based lookup or single-isotope shortcut.
    def _signal_belongs(s) -> bool:
        """True if signal s belongs to the rendered isotope."""
        if s.isotope is not None:
            return s.isotope == isotope
        if _single_isotope:
            return True
        return s.assignment in _iso_chem_labels

    _iso_signals = [s for s in experiment.signals if _signal_belongs(s)]

    # Connector arrows only for signals assigned to this isotope
    _connector_signals = [
        s for s in experiment.signals
        if s.assignment in _iso_chem_labels
    ]

    # Use union of simulation and experimental ranges to avoid clipping.
    if _iso_signals:
        exp_min = min(s.shift for s in _iso_signals)
        exp_max = max(s.shift for s in _iso_signals)
        range_min = min(shift_range[0], exp_min)
        range_max = max(shift_range[1], exp_max)
        pad = 0.1 * max(abs(exp_min), abs(exp_max))
    else:
        range_min, range_max = shift_range[0], shift_range[1]
        pad = 0.1 * max(abs(range_min), abs(range_max))
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
    y_deconv_intensity = _build_deconv_spectrum(experiment, isotope, x_grid)

    # Normalise both spectra to max = 1 so barriers land at the same
    # data coordinate (1.1) and height_ratios can be computed exactly.
    _sim_max = float(np.max(y_sim_intensity)) or 1.0
    y_sim_intensity = y_sim_intensity / _sim_max
    sim_peak_heights = [h / _sim_max for h in sim_peak_heights]

    _deconv_max = float(np.max(y_deconv_intensity)) or 1.0
    y_deconv_intensity = y_deconv_intensity / _deconv_max

    glyphs = spec.glyphs
    palette = spec.palette

    # After normalisation max=1, so:
    #   top panel y-top  = barrier(1.1) * labels_above(1.05) * headroom(1.5) = 1.7325
    #   bottom panel y-top = barrier(1.1)
    # height_ratios must equal these y-ranges so that the 0→1.1 spectrum
    # region occupies the same physical height on both panels.
    _y_top_sim = 1.1 * 1.05 * 1.5    # 1.7325
    _y_top_exp = 1.1                  # = _label_barrier_exp after normalisation

    # Pre-compute barriers (spectra normalised to max=1)
    _sim_barrier = 1.1
    _label_barrier_exp = 1.1

    # label lookups
    latex_label_map: dict[str, str] = {}
    for nucleus in molecule.nuclei:
        if nucleus.isotope != isotope:
            continue
        plain = getattr(nucleus, "chem_label", None)
        latex = getattr(nucleus, "chem_math_label", None)
        if plain and latex:
            latex_label_map[str(plain)] = str(latex)

    def _map_assignment_to_latex(assignment: str) -> str:
        if assignment is None:
            return ""
        tokens = [t.strip() for t in str(assignment).split(",")]
        return ",".join(latex_label_map.get(tok, tok) for tok in tokens)

    _label_to_pred = dict(zip(labels, shifts))

    _matched_pred_labels = {
        _map_assignment_to_latex(s.assignment)
        for s in _connector_signals
        if _label_to_pred.get(
            _map_assignment_to_latex(s.assignment)
        ) is not None
    }
    # Build chem_label → group color and math_label → group color lookups.
    _chem_to_color: dict[str, str] = {}
    if label_colors:
        for nuc in molecule.nuclei:
            if nuc.isotope == isotope and nuc.chem_label in label_colors:
                _chem_to_color[nuc.chem_label] = label_colors[nuc.chem_label]
    _math_to_color: dict[str, str] = {
        nuc.chem_math_label: _chem_to_color.get(nuc.chem_label, palette.primary)
        for nuc in molecule.nuclei
        if nuc.isotope == isotope
    }

    _sim_label_colors = {
        lab: (
            "red" if lab not in _matched_pred_labels
            else _math_to_color.get(lab, palette.primary)
        )
        for lab in labels
    }

    # Per-group Lorentzian contributions (summed over equivalent nuclei).
    _group_y: dict[str, np.ndarray] = {}
    if _chem_to_color:
        for nuc in molecule.nuclei:
            if nuc.isotope != isotope:
                continue
            contrib = lorentzian(
                x_grid, nuc.shift.lw, nuc.shift.avg, 1
            )
            if nuc.chem_label in _group_y:
                _group_y[nuc.chem_label] += contrib
            else:
                _group_y[nuc.chem_label] = contrib.copy()

    # ------------------------------------------------------------------
    # Axis-break: find spectral segments and build the figure grid
    # ------------------------------------------------------------------
    _all_peaks = shifts + [s.shift for s in _iso_signals]
    _segments = _find_spectral_segments(_all_peaks, shift_range)
    _n = len(_segments)
    _seg_widths = [abs(s[1] - s[0]) for s in _segments]

    fig = plt.figure(
        figsize=get_figsize(spec.profile, "vertical"),
        num=window_title,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0, wspace=0.004)
    _gs = fig.add_gridspec(
        2, _n,
        width_ratios=_seg_widths,
        height_ratios=[_y_top_sim, _y_top_exp],
        hspace=0.02,
    )
    _ax_top = [fig.add_subplot(_gs[0, i]) for i in range(_n)]
    _ax_bot = [
        fig.add_subplot(_gs[1, i], sharex=_ax_top[i])
        for i in range(_n)
    ]
    # Share y within each row so ylim propagates
    for i in range(1, _n):
        _ax_top[i].sharey(_ax_top[0])
        _ax_bot[i].sharey(_ax_bot[0])

    # Set inverted xlim per segment (high ppm on left)
    for i, (slo, shi) in enumerate(_segments):
        _ax_top[i].set_xlim(shi, slo)

    for _ax in _ax_top + _ax_bot:
        spec.skin_axes(_ax)
        _ax.set_yticks([])
        _ax.set_yticklabels([])
        _ax.spines[["right", "top", "left"]].set_visible(False)
    for _at in _ax_top:
        _at.tick_params(labelbottom=False)

    # Hide interior spines at break points
    if _n > 1:
        for i in range(_n):
            for _ax in [_ax_top[i], _ax_bot[i]]:
                if i > 0:
                    _ax.spines["left"].set_visible(False)
                if i < _n - 1:
                    _ax.spines["right"].set_visible(False)
        for i in range(_n - 1):
            _draw_break_markers(_ax_top[i], _ax_top[i + 1])
            _draw_break_markers(_ax_bot[i], _ax_bot[i + 1])

    def _seg_ax(x: float, row: list) -> plt.Axes:
        """Return the axis whose segment contains x (fallback: nearest)."""
        for k, (slo, shi) in enumerate(_segments):
            if min(slo, shi) <= x <= max(slo, shi):
                return row[k]
        dists = [abs(x - (s[0] + s[1]) / 2) for s in _segments]
        return row[int(np.argmin(dists))]

    # ------------------------------------------------------------------
    # Top panel — simulated spectrum
    # ------------------------------------------------------------------
    _lw_line = 0.5 * glyphs.line_lw
    _lw_conn = max(0.2, 0.16 * glyphs.line_lw)
    _lw_barrier = max(0.2, 0.4 * glyphs.line_lw)

    for _at in _ax_top:
        # Individual per-group Lorentzians (drawn first, behind composite)
        for chem_lbl, y_grp in _group_y.items():
            col = _chem_to_color.get(chem_lbl, palette.primary)
            y_grp_norm = y_grp / _sim_max
            _at.fill_between(
                x_grid, y_grp_norm,
                alpha=0.18, color=col, linewidth=0,
            )
            _at.plot(
                x_grid, y_grp_norm,
                lw=_lw_line * 0.65, color=col, alpha=0.75,
            )
        # Composite trace on top
        _at.plot(
            x_grid, y_sim_intensity,
            lw=_lw_line, color=palette.primary,
        )
        _at.plot(
            shifts, sim_peak_heights,
            lw=0, color=palette.primary, markersize=glyphs.ms,
        )
        _annotate_peaks_with_barrier(
            _at,
            x_grid=x_grid,
            y_intensity=y_sim_intensity,
            peak_x=shifts,
            labels=labels,
            shift_range=shift_range,
            spec=spec,
            palette=palette,
            glyphs=glyphs,
            reverse_axis=True,
            label_fontsize=spec.typography.label,
            line_scale=0.8,
            label_colors=_sim_label_colors,
            label_mindist_abs=0.015 * abs(
                shift_range[1] - shift_range[0]
            ),
        )
    _ax_top[0].set_ylim(0, _y_top_sim)

    _blend_sim = mtransforms.blended_transform_factory(
        _ax_top[0].transAxes, _ax_top[0].transData
    )
    _ax_top[0].text(
        0.0, _sim_barrier / 2, "Simulation",
        transform=_blend_sim,
        rotation=90, va="center", ha="right",
        fontsize=spec.typography.axis_label, clip_on=False,
    )

    # ------------------------------------------------------------------
    # Bottom panel — deconvoluted + raw experimental spectrum
    # ------------------------------------------------------------------

    # Individual Lorentzian/Gaussian components (thin dashed, behind sum)
    _lw_comp = max(0.3, _lw_line * 0.7)
    for signal in _iso_signals:
        exp_width_ppm = signal.width / (
            get_nuclear_gamma(isotope) * experiment.magnetic_field
        )
        y_comp = (
            signal.l_to_g
            * lorentzian(x_grid, exp_width_ppm, signal.shift, signal.area)
            + (1 - signal.l_to_g)
            * gaussian(x_grid, exp_width_ppm, signal.shift, signal.area)
        )
        y_comp_norm = y_comp / _deconv_max
        for _ab in _ax_bot:
            _ab.plot(
                x_grid, y_comp_norm,
                lw=_lw_comp, color=palette.primary,
                alpha=0.35, linestyle="--",
            )

    for _ab in _ax_bot:
        _ab.plot(
            x_grid, y_deconv_intensity,
            lw=_lw_line, color=palette.primary, alpha=0.7,
        )

    if experiment.spectrum is not None:
        x_raw = np.asarray(experiment.spectrum[:, 0], dtype=float)
        y_raw = np.asarray(experiment.spectrum[:, 1], dtype=float)

        exp_ref = getattr(experiment, "exp_reference", None)
        if exp_ref is not None:
            exp_ref = float(exp_ref)
            tol_ppm = 1.0
            m_deconv = (
                (x_grid >= exp_ref - tol_ppm)
                & (x_grid <= exp_ref + tol_ppm)
            )
            ref_y_deconv = float(
                np.max(y_deconv_intensity[m_deconv])
                if np.any(m_deconv)
                else np.max(y_deconv_intensity)
            )
            m_raw = (
                (x_raw >= exp_ref - tol_ppm)
                & (x_raw <= exp_ref + tol_ppm)
            )
            ref_y_raw = float(
                np.max(y_raw[m_raw]) if np.any(m_raw) else np.max(y_raw)
            )
            scale = ref_y_deconv / ref_y_raw if ref_y_raw > 0.0 else 1.0
            y_raw = np.clip(
                y_raw * scale,
                a_min=None,
                a_max=float(np.max(y_deconv_intensity)),
            )

        for _ab in _ax_bot:
            _ab.plot(
                x_raw, y_raw,
                lw=_lw_line, color=palette.highlight,
            )

    # Barrier line and ylim on bottom panel
    for _ab in _ax_bot:
        _ab.hlines(
            _label_barrier_exp,
            np.min(shift_range), np.max(shift_range),
            linestyle="-", color=palette.primary,
            linewidth=_lw_barrier, alpha=0.7,
        )
    _ax_bot[0].set_ylim(0, _label_barrier_exp)

    # ------------------------------------------------------------------
    # Connectors: exp peak → barrier → pred position in top panel
    # ------------------------------------------------------------------
    for signal in _connector_signals:
        latex_lab = _map_assignment_to_latex(signal.assignment)
        pred_x = _label_to_pred.get(latex_lab)
        if pred_x is None:
            continue
        exp_x = signal.shift
        peak_y = y_deconv_intensity[find_index_of_nearest(x_grid, exp_x)]
        _ab = _seg_ax(exp_x, _ax_bot)
        _at = _seg_ax(pred_x, _ax_top)

        _ab.plot(
            [exp_x, exp_x], [peak_y, _label_barrier_exp],
            linestyle="--", color=palette.primary,
            linewidth=_lw_conn, alpha=0.4, clip_on=False,
        )
        con = mpatches.ConnectionPatch(
            xyA=(exp_x, _label_barrier_exp), xyB=(pred_x, 0),
            coordsA="data", coordsB="data",
            axesA=_ab, axesB=_at,
            linestyle="--", color=palette.primary,
            linewidth=_lw_conn, alpha=0.4,
        )
        fig.add_artist(con)
        _at.plot(
            [pred_x, pred_x], [0, _sim_barrier],
            linestyle="--", color=palette.primary,
            linewidth=_lw_conn, alpha=0.4, clip_on=False,
        )

    # Unmatched experimental signals
    _matched_assignments = {
        _map_assignment_to_latex(s.assignment)
        for s in _connector_signals
        if _label_to_pred.get(
            _map_assignment_to_latex(s.assignment)
        ) is not None
    }
    for signal in _iso_signals:
        if _map_assignment_to_latex(signal.assignment) in _matched_assignments:
            continue
        exp_x = signal.shift
        peak_y = y_deconv_intensity[find_index_of_nearest(x_grid, exp_x)]
        _ab = _seg_ax(exp_x, _ax_bot)
        _ab.plot(
            exp_x, peak_y, marker="o", color="red",
            markersize=0.4 * glyphs.line_lw + 0.5, lw=0, zorder=5,
        )
        _ab.text(
            exp_x, peak_y, f" {signal.assignment}",
            fontsize=str(round(spec.typography.label * 0.7)),
            color="red", va="bottom", ha="left", clip_on=True,
        )

    # "Experiment" label
    _blend_exp = mtransforms.blended_transform_factory(
        _ax_bot[0].transAxes, _ax_bot[0].transData
    )
    _ax_bot[0].text(
        0.0, _label_barrier_exp / 2, "Experiment",
        transform=_blend_exp,
        rotation=90, va="center", ha="right",
        fontsize=spec.typography.axis_label, clip_on=False,
    )

    # Centred x-label spanning the full figure width (works with axis breaks)
    fig.supxlabel(
        r"{} $\delta$ (ppm)".format(isotope_format(isotope)),
        fontsize=spec.typography.axis_label,
    )
    # Per-segment tick density: at most ~3 major ticks per segment
    for i, (slo, shi) in enumerate(_segments):
        _ab = _ax_bot[i]
        _at = _ax_top[i]
        _nbins = max(2, min(3, int(abs(shi - slo) / 5)))
        _loc = ticker.MaxNLocator(nbins=_nbins, integer=False)
        _ab.xaxis.set_major_locator(_loc)
        _ab.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        _at.xaxis.set_major_locator(ticker.MaxNLocator(
            nbins=_nbins, integer=False
        ))

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save and verbose:
        logger.info("Spectra saved to %s", f"{save_name}.pdf")

    return fig, np.array(_ax_top + _ax_bot)


def _adaptive_grid(
    experiments: list,
    isotope: str,
    x_lo: float,
    x_hi: float,
    pts_per_peak: int = 2000,
    gap_pts: int = 2000,
    half_widths: float = 6.0,
) -> np.ndarray:
    """Build a ppm grid dense around every peak and coarse in the gaps.

    For each signal a local grid spanning ±``half_widths`` linewidths is
    created with ``pts_per_peak`` points. The remainder of the axis is
    covered by a coarse uniform grid with ``gap_pts`` points total. The
    result is sorted and deduplicated.

    Args:
        experiments: All experiments (used to collect peak positions/widths).
        isotope: Isotope to build the grid for.
        x_lo: Left edge of the full axis (low ppm, after padding).
        x_hi: Right edge of the full axis (high ppm, after padding).
        pts_per_peak: Points in the dense window around each peak.
        gap_pts: Total points in the coarse background grid.
        half_widths: Half-width of the dense window in units of linewidth.
    """
    pieces = [np.linspace(x_lo, x_hi, gap_pts)]
    for exp in experiments:
        for signal in exp.signals:
            if signal.isotope is not None and signal.isotope != isotope:
                continue
            lw_ppm = signal.width / (
                get_nuclear_gamma(isotope) * exp.magnetic_field
            )
            lo = max(x_lo, signal.shift - half_widths * lw_ppm)
            hi = min(x_hi, signal.shift + half_widths * lw_ppm)
            if lo < hi:
                pieces.append(np.linspace(lo, hi, pts_per_peak))
    grid = np.unique(np.concatenate(pieces))
    return grid


def _build_deconv_spectrum(
    experiment: Experiment,
    isotope: str,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Return deconvoluted spectrum intensity on ``x_grid`` for ``isotope``."""
    y = np.zeros_like(x_grid)
    for signal in experiment.signals:
        if signal.isotope is not None and signal.isotope != isotope:
            continue
        exp_width_ppm = signal.width / (
            get_nuclear_gamma(isotope) * experiment.magnetic_field
        )
        y += signal.l_to_g * lorentzian(x_grid, exp_width_ppm, signal.shift, signal.area)
        y += (1 - signal.l_to_g) * gaussian(x_grid, exp_width_ppm, signal.shift, signal.area)
    return y


def plot_vt_spectra(
    experiments: list[Experiment],
    isotope: str,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "vt_spectra",
    window_title: str = "VT Spectra",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot all experimental 1D spectra stacked by temperature.

    Uses raw spectrum data when available; otherwise reconstructs from
    deconvoluted signals. Spectra are offset vertically in proportion to
    their temperature relative to the coldest measurement, and coloured
    on a cold-to-hot (blue→red) scale.

    Args:
        experiments: Experiments to include.
        isotope: Isotope to plot (e.g. ``"1H"``).
        spec: Plot style specification.
        save: If True, save the figure to ``save_name``.
        show: If True, display the figure interactively.
        save_name: Output file base name.
        window_title: Figure window title.
        verbose: If True, log the saved path.

    Returns:
        A tuple ``(fig, ax)``, or ``(None, None)`` if no data are available.
    """
    exps = sorted(
        [e for e in experiments if e.spectrum is not None or e.signals],
        key=lambda e: e.temperature,
    )
    if not exps:
        logger.warning("plot_vt_spectra: no experimental data available.")
        return None, None

    temps = [e.temperature for e in exps]
    T_min, T_max = temps[0], temps[-1]
    T_span = T_max - T_min if T_max > T_min else 1.0
    n = len(exps)

    cmap = plt.cm.turbo
    norm_t = plt.Normalize(vmin=T_min, vmax=T_max)

    # Determine ppm range from all available data
    _lo, _hi = float("inf"), float("-inf")
    for e in exps:
        if e.spectrum is not None:
            x = np.asarray(e.spectrum[:, 0], dtype=float)
            _lo = min(_lo, float(x.min()))
            _hi = max(_hi, float(x.max()))
        for s in e.signals:
            if s.isotope is None or s.isotope == isotope:
                _lo = min(_lo, s.shift)
                _hi = max(_hi, s.shift)
    if _lo == float("inf"):
        logger.warning("plot_vt_spectra: no shift data for isotope %s.", isotope)
        return None, None

    ppm_lo, ppm_hi = _lo, _hi
    all_shifts = [s.shift for e in exps for s in e.signals
                  if s.isotope is None or s.isotope == isotope]
    pad = max(5.0, 0.1 * abs(ppm_hi - ppm_lo))
    x_grid = _adaptive_grid(exps, isotope, ppm_lo - pad, ppm_hi + pad)

    # Build (x, y) pairs for each experiment
    traces: list[tuple[np.ndarray, np.ndarray]] = []
    for exp in exps:
        if exp.spectrum is not None:
            x = np.asarray(exp.spectrum[:, 0], dtype=float)
            y = np.asarray(exp.spectrum[:, 1], dtype=float)
        else:
            x = x_grid
            y = _build_deconv_spectrum(exp, isotope, x_grid)
        traces.append((x, y))

    global_max = float(max(np.max(np.abs(y)) for _, y in traces)) or 1.0
    # Offset step: 0.15× the global peak — traces overlap intentionally.
    offset_step = 0.15 * global_max

    _segments = _find_spectral_segments(
        all_shifts,
        (float(x_grid[0]), float(x_grid[-1])),
        padding=pad,
    )
    _n_seg = len(_segments)
    _seg_widths = [abs(s[1] - s[0]) for s in _segments]

    fig_w, fig_h_base = get_figsize(spec.profile, "standard")
    fig_h = fig_h_base + 0.5 * (n - 1)

    fig = plt.figure(figsize=(fig_w, fig_h), num=window_title, layout="constrained")
    fig.get_layout_engine().set(w_pad=0, wspace=0.004)
    _gs = fig.add_gridspec(1, _n_seg, width_ratios=_seg_widths)
    axes = [fig.add_subplot(_gs[0, i]) for i in range(_n_seg)]

    # Share y so ylim is consistent across segments
    for i in range(1, _n_seg):
        axes[i].sharey(axes[0])

    for i, (slo, shi) in enumerate(_segments):
        axes[i].set_xlim(shi, slo)

    glyphs = spec.glyphs

    for ax in axes:
        spec.skin_axes(ax)
        ax.set_yticks([])
        ax.spines[["right", "top", "left"]].set_visible(False)

    # Hide interior spines and draw break markers
    if _n_seg > 1:
        for i in range(_n_seg):
            if i > 0:
                axes[i].spines["left"].set_visible(False)
            if i < _n_seg - 1:
                axes[i].spines["right"].set_visible(False)
        for i in range(_n_seg - 1):
            _draw_break_markers(axes[i], axes[i + 1])

    for exp, (x, y) in zip(exps, traces):
        y_offset = (exp.temperature - T_min) / T_span * offset_step * (n - 1)
        color = cmap(norm_t(exp.temperature))
        for i, (slo, shi) in enumerate(_segments):
            mask = (x >= min(slo, shi)) & (x <= max(slo, shi))
            if np.any(mask):
                axes[i].plot(
                    x[mask], y[mask] + y_offset,
                    lw=glyphs.line_lw * 0.6, color=color,
                )
        axes[0].text(
            0.01, y_offset,
            f"{exp.temperature:.0f} K",
            transform=axes[0].get_yaxis_transform(),
            va="bottom", ha="left",
            fontsize=spec.typography.tick_label,
            color=color,
        )

    fig.supxlabel(
        r"{} $\delta$ (ppm)".format(isotope_format(isotope)),
        fontsize=spec.typography.axis_label,
    )
    for i, (slo, shi) in enumerate(_segments):
        _nbins = max(2, min(3, int(abs(shi - slo) / 5)))
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=_nbins, integer=False))
        axes[i].xaxis.set_minor_locator(ticker.AutoMinorLocator())

    render_figure(fig, save=save, show=show, save_name=save_name, fmt="png")

    if save and verbose:
        logger.info("VT spectra saved to %s", f"{save_name}.png")

    return fig, axes[0]


def _find_spectral_segments(
    peak_positions: list[float],
    x_range: tuple[float, float],
    min_gap: float | None = None,
    padding: float | None = None,
) -> list[tuple[float, float]]:
    """Return x-axis segments that contain peaks, for axis-break layout.

    Gaps between segments larger than ``min_gap`` become axis breaks.
    Each segment is padded by ``padding`` on both sides and clipped to
    ``x_range``.  Segments are returned in descending order (high ppm
    first, matching the inverted NMR x-axis).
    """
    span = abs(x_range[1] - x_range[0])
    if min_gap is None:
        min_gap = max(20.0, 0.15 * span)
    if padding is None:
        padding = max(5.0, 0.03 * span)

    if not peak_positions:
        return [(max(x_range), min(x_range))]

    sorted_peaks = sorted(set(peak_positions))
    xlo, xhi = min(x_range), max(x_range)

    # Group into clusters separated by gaps >= min_gap
    clusters: list[list[float]] = [[sorted_peaks[0]]]
    for p in sorted_peaks[1:]:
        if p - clusters[-1][-1] < min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    # Padded segments clipped to x_range
    segs: list[tuple[float, float]] = []
    for cl in clusters:
        lo = max(xlo, min(cl) - padding)
        hi = min(xhi, max(cl) + padding)
        if lo < hi:
            segs.append((lo, hi))

    # Merge any overlapping segments (can arise from padding)
    segs.sort()
    merged: list[list[float]] = [list(segs[0])]
    for lo, hi in segs[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    # Return descending (high ppm first = left side of NMR axis)
    return sorted(
        [(s[0], s[1]) for s in merged],
        key=lambda s: s[0],
        reverse=True,
    )


def _draw_break_markers(
    ax_l: plt.Axes,
    ax_r: plt.Axes,
    d_pts: float = 3.0,
    lw: float = 0.6,
) -> None:
    """Draw parallel diagonal break markers at y=0 with fixed physical size.

    Using display coordinates ensures identical slope regardless of axis width.
    """
    kw = dict(clip_on=False, color="k", lw=lw, zorder=10)
    for ax, x_anchor in [(ax_l, 1.0), (ax_r, 0.0)]:
        # Anchor point in display (pixel) coordinates
        anchor = ax.transAxes.transform((x_anchor, 0))
        p0 = anchor + np.array([-d_pts, -d_pts])
        p1 = anchor + np.array([+d_pts, +d_pts])
        # Convert back to axes fraction coordinates for plotting
        inv = ax.transAxes.inverted()
        p0_ax = inv.transform(p0)
        p1_ax = inv.transform(p1)
        ax.plot(
            [p0_ax[0], p1_ax[0]], [p0_ax[1], p1_ax[1]],
            transform=ax.transAxes, **kw,
        )


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
    label_mindist_abs: float | None = None,
    connector_alpha: float = 0.4,
    barrier_alpha: float = 0.7,
    label_fontsize: str | None = None,
    line_scale: float = 1.0,
    label_colors: dict[str, str] | None = None,
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

    # Filter to peaks visible in this axis's x range
    xmin, xmax = ax.get_xlim()
    visible = [
        (px, py, lab)
        for px, py, lab in zip(peak_x_sorted, peak_y_sorted, labels_sorted)
        if min(xmin, xmax) <= px <= max(xmin, xmax)
    ]
    if not visible:
        return

    vis_px, vis_py, vis_labs = zip(*visible)
    vis_px = list(vis_px)

    # Resolve label overlaps
    xrange = abs(xmax - xmin)
    mindist = (
        label_mindist_abs
        if label_mindist_abs is not None
        else label_mindist_scale * xrange
    )
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
        _col = (
            label_colors.get(lab, palette.primary)
            if label_colors else palette.primary
        )
        ax.text(
            lx, labels_position_y, lab,
            fontsize=_fs, rotation="vertical",
            va="bottom", ha="center",
            color=_col, clip_on=False,
        )
        ax.plot(
            [px, px, lx],
            [py, label_barrier, labels_position_y],
            linestyle="--", color=_col,
            linewidth=_lw_connector,
            alpha=connector_alpha, clip_on=False,
        )
