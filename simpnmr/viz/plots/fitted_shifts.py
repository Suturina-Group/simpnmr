# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot fitted theoretical-versus-experimental chemical shifts.

Provides the fitted-shift scatter plot together with its label and summary-table
helpers.
"""

import logging

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.constants as constants

from simpnmr.core.const import ptable
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.fitting import models
from simpnmr.core.fitting.vt import compute_curie_prefactor
from simpnmr.viz.layout.canvas import create_header_plot_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.label import resolve_label_layout
from simpnmr.viz.layout.table import render_compact_table
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.uncertainty import format_compact_uncertainty

logger = logging.getLogger(__name__)


def _build_element_legend_handles(
    markers: dict[str, str],
    palette,
    glyphs,
) -> list[mlines.Line2D]:
    """Build legend handles for the element-to-marker mapping."""
    return [
        mlines.Line2D(
            [],
            [],
            lw=0,
            marker=marker,
            color=palette.primary,
            markersize=glyphs.ms,
            markerfacecolor=(0, 0, 0, 0.55),
            markeredgecolor=palette.primary,
            markeredgewidth=0.8,
            label=element,
        )
        for element, marker in markers.items()
    ]


def plot_fitted_shifts(
    molecule: Molecule,
    experiment: Experiment,
    susc_model: models.SusceptibilityModel,
    spec: PlotSpec,
    average: bool = True,
    show_point_labels: bool = True,
    save: bool = True,
    show: bool = True,
    save_name: str = "nmr_shifts.pdf",
    window_title: str = "Fitted Shifts",
    susc_units: str = "A3",
    verbose: bool = True,
    spin: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots theoretical vs experimental shifts for a fitted susceptibility model.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experimental shift data.
        susc_model: Fitted susceptibility model.
        average: If ``True``, averages equivalent nuclei (same chemical label).
        show_point_labels: If ``True``, draws nucleus labels next to markers.
            If ``False``, draws an element-shape legend instead.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        susc_units: Units for reporting susceptibility values in the annotation.
            Supported: ``"A3"``, ``"A3 mol-1"``, ``"cm3"``, ``"cm3 mol-1"``.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    seen = set()
    unique_nuclei = [
        seen.add(nuc.chem_label) or nuc
        for nuc in molecule.nuclei
        if nuc.chem_label not in seen
    ]

    # Only plot nuclei whose chem_label has an experimental signal
    _present_labels = set()
    for nuc in unique_nuclei:
        try:
            experiment[nuc.chem_label]
            _present_labels.add(nuc.chem_label)
        except (KeyError, TypeError):
            logger.warning(
                "chem_label '%s' absent from experiment — skipped in shift plot",
                nuc.chem_label,
            )
    unique_nuclei = [nuc for nuc in unique_nuclei if nuc.chem_label in _present_labels]

    if average:
        # Theoretical shifts, averaged over equivalent nuclei
        calc_shifts = {nuc.chem_label: nuc.shift.avg for nuc in unique_nuclei}
        # Experimental shifts, same order as theoretical
        exp = {label: experiment[label].shift for label in calc_shifts.keys()}
    else:
        # One signal per nucleus
        calc_shifts = {nuc.chem_label: [] for nuc in unique_nuclei}
        for nuc in molecule.nuclei:
            if nuc.chem_label not in _present_labels:
                continue
            calc_shifts[nuc.chem_label].append(nuc.shift.total)

        # Experimental shifts, same order as theoretical
        exp = {
            label: [experiment[label].shift] * len(calc_shifts[label])
            for label in calc_shifts.keys()
        }

    # Element specific markers with consistent order
    _unique_elements = [
        ele for ele in ptable.elements if ele in [nuc.label_nn for nuc in unique_nuclei]
    ]
    _markers = {ele: mrkr for (ele, mrkr) in zip(_unique_elements, ["o", "v", "s"])}

    markers = {
        nuc.chem_label: _markers[nuc.label_nn]
        for nuc in molecule.nuclei
        if nuc.chem_label in _present_labels
    }

    # if math labels are present then use these instead
    if all([len(nuc.chem_math_label) for nuc in molecule.nuclei]):
        for nuc in unique_nuclei:
            calc_shifts[nuc.chem_math_label] = calc_shifts.pop(nuc.chem_label)
            markers[nuc.chem_math_label] = markers.pop(nuc.chem_label)
            exp[nuc.chem_math_label] = exp.pop(nuc.chem_label)

    figure_variant = "vertical_extended" if len(susc_model.VARNAMES) > 3 else "vertical"
    # TODO(viz): Move fitted-shift header/plot ratio tuning into PlotSpec so
    # figure-layout heuristics are configured centrally rather than locally.
    header_ratio, plot_ratio = (
        (1.90, 4.50) if figure_variant == "vertical_extended" else (1.35, 5.05)
    )
    fig, header_ax, ax = create_header_plot_canvas(
        spec.profile,
        variant=figure_variant,
        window_title=window_title,
        layout="constrained",
        header_ratio=header_ratio,
        plot_ratio=plot_ratio,
        hspace=0.02,
    )

    glyphs = spec.glyphs
    palette = spec.palette
    scale = spec.skin_axes(ax)

    fig.patch.set_facecolor(palette.annotation_bg)
    header_ax.set_facecolor(palette.annotation_bg)
    ax.set_facecolor(palette.annotation_bg)

    if susc_units == "A3":
        conv = 1.0
        model_unit_label = "Å³"
    elif susc_units == "A3 mol-1":
        conv = constants.Avogadro
        model_unit_label = "Å³ mol⁻¹"
    elif susc_units == "cm3":
        conv = 1e-24
        model_unit_label = "cm³"
    elif susc_units == "cm3 mol-1":
        conv = 1e-24 * constants.Avogadro / (4 * np.pi)
        model_unit_label = "cm³ mol⁻¹"
    else:
        raise ValueError(
            "Unsupported susc_units. Expected one of: A3, A3 mol-1, cm3, cm3 mol-1."
        )

    fit_lines = [
        f"R²adj: {susc_model.adj_r2:.4f}",
        f"MAE: {susc_model.mae:.1f} ppm",
        f"RMSE: {susc_model.rmse:.1f} ppm",
    ]

    model_lines: list[str] = []

    if spin is not None:
        # Report all susceptibility quantities as dimensionless reduced values
        # χ′T = χ · T / norm_factor  (norm_factor in Å³·K)
        norm_factor = compute_curie_prefactor(spin)  # Å³·K
        T = float(molecule.susc.temperature)
        red_conv = T / norm_factor  # Å³ → dimensionless

        # iso and other non-ax/non-rh_over_ax terms first
        for name in susc_model.VARNAMES:
            if name in ("ax", "rh_over_ax"):
                continue
            val_red = float(susc_model.final_var_values[name]) * red_conv
            label = susc_model.VARNAMES_MM[name]
            if name == "iso":
                label = r"$\chi'_\mathrm{iso}T$"
            err = susc_model.fit_stdev.get(name)
            if name in susc_model.fit_vars and err is not None and err > 0:
                err_red = float(err) * red_conv
                compact_value = format_compact_uncertainty(val_red, err_red)
                model_lines.append(f"{label}: {compact_value}")
            else:
                model_lines.append(f"{label}: {val_red:.4f}")

        # Δχ′_ax T
        dax_red = float(molecule.susc.axiality) * red_conv
        ax_err = susc_model.fit_stdev.get("ax")
        if "ax" in susc_model.fit_vars and ax_err is not None and ax_err > 0:
            dax_red_err = float(ax_err) * red_conv
            dax_str = format_compact_uncertainty(dax_red, dax_red_err)
        else:
            dax_str = f"{dax_red:.4f}"
        model_lines.append(
            f"$\\Delta\\chi'_{{\\mathrm{{ax}}}}T$: {dax_str}"
        )

        # Δχ_rh / Δχ_ax ratio immediately below
        if "rh_over_ax" in susc_model.VARNAMES:
            _roa_val = float(
                susc_model.final_var_values["rh_over_ax"]
            )
            _roa_label = susc_model.VARNAMES_MM["rh_over_ax"]
            _roa_err = susc_model.fit_stdev.get("rh_over_ax")
            if (
                "rh_over_ax" in susc_model.fit_vars
                and _roa_err is not None
                and _roa_err > 0
            ):
                model_lines.append(
                    f"{_roa_label}: "
                    f"{format_compact_uncertainty(_roa_val, float(_roa_err))}"
                )
            else:
                model_lines.append(f"{_roa_label}: {_roa_val:.4f}")

        model_block_header = r"$\chi'\!T$ (dimensionless)"
    else:
        for name in susc_model.VARNAMES:
            val = float(susc_model.final_var_values[name]) * conv
            label = susc_model.VARNAMES_MM[name]
            err = susc_model.fit_stdev.get(name)
            if name in susc_model.fit_vars and err is not None and err > 0:
                err_val = float(err) * conv
                compact_value = format_compact_uncertainty(val, err_val)
                model_lines.append(f"{label}: {compact_value}")
            else:
                model_lines.append(f"{label}: {val:.3f}")

        dax = float(molecule.susc.axiality) * conv
        ax_err = susc_model.fit_stdev.get("ax")
        if "ax" in susc_model.fit_vars and ax_err is not None and ax_err > 0:
            dax_str = format_compact_uncertainty(dax, float(ax_err) * conv)
        else:
            dax_str = f"{dax:.3f}"
        model_lines.append(f"Δχ$_{{ax}}$: {dax_str}")
        model_block_header = f"χ ({model_unit_label})"

    euler_lines = [
        f"α: {int(round(molecule.susc.alpha))}°",
        f"β: {int(round(molecule.susc.beta))}°",
        f"γ: {int(round(molecule.susc.gamma))}°",
    ]

    blocks = [
        ("Fit Statistics", fit_lines),
        (model_block_header, model_lines),
        ("Euler Angles", euler_lines),
    ]

    ax.grid(True, which="major", color=palette.grid, linewidth=1.0)
    ax.grid(True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        ax.plot(
            calc,
            expt,
            lw=0,
            marker=markers[label],
            color=palette.primary,
            markersize=glyphs.ms,
            markerfacecolor=(0, 0, 0, 0.55),
            markeredgecolor=palette.primary,
            markeredgewidth=0.8,
        )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    shared_min = np.min([x_lim, y_lim])
    shared_max = np.max([x_lim, y_lim])
    ax.set_xlim([shared_min, shared_max])
    ax.set_ylim([shared_min, shared_max])
    ax.set_aspect("equal", adjustable="box")
    diag_line = ax.plot(
        [shared_min, shared_max],
        [shared_min, shared_max],
        color=palette.primary,
        lw=0.75,
    )[0]

    ax.set_xlabel("Theoretical Shift (ppm)")
    ax.set_ylabel("Experimental Shift (ppm)")

    label_entries: list[tuple[str, float, float]] = []
    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        if average:
            label_entries.append((label, float(calc), float(expt)))
        else:
            for calc_value, exp_value in zip(calc, expt):
                label_entries.append((label, float(calc_value), float(exp_value)))

    render_compact_table(
        header_ax,
        blocks,
        spec,
        bbox=[0.0, 0.02, 1.0, 0.96],
        cell_align="center",
        remove_outer_frame=True,
    )

    fig.canvas.draw()
    plot_pos = ax.get_position()
    header_pos = header_ax.get_position()
    header_ax.set_position(
        [
            plot_pos.x0,
            header_pos.y0,
            plot_pos.width,
            header_pos.height,
        ]
    )

    ax.invert_xaxis()
    ax.invert_yaxis()
    if show_point_labels:
        resolve_label_layout(
            ax,
            label_entries,
            fontsize=scale.annotation,
            marker_size=glyphs.ms,
            diag_line=diag_line,
        )
    else:
        legend_handles = _build_element_legend_handles(_markers, palette, glyphs)
        ax.legend(handles=legend_handles, loc="best")

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Chemical shift plot saved to %s", f"{save_name}.pdf")

    return fig, ax
