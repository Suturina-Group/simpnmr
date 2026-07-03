# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot fitted theoretical-versus-experimental chemical shifts."""

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
from simpnmr.core.fitting.vt import compute_chi_prefactor
from simpnmr.core.phys.susc import compute_bleaney_params
from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.label import resolve_label_layout
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
    orbit: float | None = None,
    total_J: float | None = None,
    label_colors: dict[str, str] | None = None,
    variant: str = "standard",
    width_scale: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot theoretical vs experimental shifts for a fitted susc model.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experimental shift data.
        susc_model: Fitted susceptibility model.
        average: If ``True``, averages equivalent nuclei (same chemical label).
        show_point_labels: If ``True``, draws nucleus labels next to markers.
            If ``False``, draws an element-shape legend instead.
        save: If ``True``, saves the plot to ``save_name``.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        susc_units: Units for susceptibility values in the annotation.
            Supported: ``"A3"``, ``"A3 mol-1"``, ``"cm3"``,
            ``"cm3 mol-1"``.
        verbose: If ``True``, prints the output file name when saving.
        spin: Spin quantum number S.  Used to normalise susceptibility
            values in the annotation.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the Curie prefactor J(J+1) used for unit
            conversion.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Deduplicate by (chem_label, isotope) so same-name labels from different
    # isotopes are kept separate.
    seen = set()
    unique_nuclei = []
    for nuc in molecule.nuclei:
        key = (nuc.chem_label, nuc.isotope)
        if key not in seen:
            seen.add(key)
            unique_nuclei.append(nuc)

    def _exp_key(nuc):
        return (nuc.chem_label, nuc.isotope) if nuc.isotope is not None else nuc.chem_label

    # Only plot nuclei whose chem_label has an experimental signal
    _present_keys = set()
    for nuc in unique_nuclei:
        try:
            experiment[_exp_key(nuc)]
            _present_keys.add(_exp_key(nuc))
        except (KeyError, TypeError):
            logger.warning(
                "chem_label '%s' absent from experiment — skipped",
                nuc.chem_label,
            )
    unique_nuclei = [
        nuc for nuc in unique_nuclei
        if _exp_key(nuc) in _present_keys
    ]
    _present_labels = {nuc.chem_label for nuc in unique_nuclei}

    if average:
        calc_shifts = {
            nuc.chem_label: nuc.shift.avg for nuc in unique_nuclei
        }
        exp = {
            nuc.chem_label: experiment[_exp_key(nuc)].shift
            for nuc in unique_nuclei
        }
    else:
        calc_shifts = {nuc.chem_label: [] for nuc in unique_nuclei}
        for nuc in molecule.nuclei:
            if nuc.chem_label not in _present_labels:
                continue
            calc_shifts[nuc.chem_label].append(nuc.shift.total)
        exp = {
            nuc.chem_label: [experiment[_exp_key(nuc)].shift] * len(calc_shifts[nuc.chem_label])
            for nuc in unique_nuclei
        }

    # Element-specific markers in periodic-table order.
    # Use all nuclei that pass the _present_labels filter so that every
    # element that will appear in `markers` is guaranteed to be in `_markers`.
    _present_nuclei = [
        nuc for nuc in molecule.nuclei if nuc.chem_label in _present_labels
    ]
    _unique_elements = [
        ele for ele in ptable.elements
        if ele in {nuc.label_nn for nuc in _present_nuclei}
    ]
    _marker_pool = ["o", "v", "s", "^", "D", "p", "h", "X", "*", "P"]
    _markers = {
        ele: _marker_pool[i % len(_marker_pool)]
        for i, ele in enumerate(_unique_elements)
    }
    markers = {
        nuc.chem_label: _markers[nuc.label_nn]
        for nuc in _present_nuclei
    }

    # Assign one color per label.  Use the caller-supplied mapping when
    # available (so colors stay consistent with spectrum plots); fall back
    # to the active Matplotlib color cycle.
    if label_colors is not None:
        _colors: dict[str, str] = {
            nuc.chem_label: label_colors.get(nuc.chem_label, "")
            for nuc in unique_nuclei
        }
    else:
        _color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _colors = {
            nuc.chem_label: _color_cycle[i % len(_color_cycle)]
            for i, nuc in enumerate(unique_nuclei)
        }

    # Prefer math labels when available
    if all(len(nuc.chem_math_label) for nuc in molecule.nuclei):
        for nuc in unique_nuclei:
            calc_shifts[nuc.chem_math_label] = calc_shifts.pop(
                nuc.chem_label
            )
            markers[nuc.chem_math_label] = markers.pop(nuc.chem_label)
            exp[nuc.chem_math_label] = exp.pop(nuc.chem_label)
            _colors[nuc.chem_math_label] = _colors.pop(nuc.chem_label)

    fig, ax = create_canvas(
        spec.profile,
        variant=variant,
        window_title=window_title,
        layout="constrained",
        width_scale=width_scale,
    )

    glyphs = spec.glyphs
    palette = spec.palette
    scale = spec.skin_axes(ax)

    fig.patch.set_facecolor(palette.annotation_bg)
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
            "Unsupported susc_units. "
            "Expected one of: A3, A3 mol-1, cm3, cm3 mol-1."
        )

    def _fmt_val(val: float, fmt: str) -> str:
        return "0" if val == 0.0 else format(val, fmt)

    def _ci_str(val: float, key: str, fmt: str = ".4f") -> str:
        err = susc_model.fit_stdev.get(key)
        if err is not None and np.isfinite(float(err)) and float(err) > 0:
            return format_compact_uncertainty(val, float(err))
        return _fmt_val(val, fmt)

    def _err_scaled(key: str, factor: float) -> float | None:
        err = susc_model.fit_stdev.get(key)
        if err is not None and np.isfinite(float(err)) and float(err) > 0:
            return float(err) * factor
        return None

    def _euler_str(angle_deg: float, key: str) -> str:
        err = susc_model.fit_stdev.get(key)
        if err is not None and np.isfinite(err) and err > 0:
            return f"{format_compact_uncertainty(angle_deg, float(err))}°"
        val_str = "0" if angle_deg == 0.0 else f"{angle_deg:.1f}"
        return f"{val_str}°"

    ax0 = float(molecule.susc.axiality)
    _roa_direct = susc_model.final_var_values.get("rh_over_ax")
    if _roa_direct is not None:
        roa_val = float(_roa_direct)
    else:
        rh0 = float(molecule.susc.rhombicity)
        roa_val = rh0 / ax0 if abs(ax0) > 1e-12 else float("nan")

    model_lines: list[str] = []

    if spin is not None:
        norm_factor = compute_chi_prefactor(spin, total_J)
        T = float(molecule.susc.temperature)
        red_conv = T / norm_factor

        iso_red = float(molecule.susc.iso) * red_conv
        iso_err_red = _err_scaled("iso", red_conv)
        if iso_err_red is not None:
            iso_line = format_compact_uncertainty(iso_red, iso_err_red)
        else:
            iso_line = _fmt_val(iso_red, ".4f")
        model_lines.append(
            f"$\\chi$′$_\\mathrm{{iso}}$T: {iso_line}"
        )

        dax_red = ax0 * red_conv
        ax_err_red = _err_scaled("ax", red_conv)
        if ax_err_red is not None:
            dax_line = format_compact_uncertainty(dax_red, ax_err_red)
        else:
            dax_line = _fmt_val(dax_red, ".4f")
        model_lines.append(
            f"$\\Delta\\chi$′$_{{\\mathrm{{ax}}}}$T: {dax_line}"
        )
        model_lines.append(
            "$\\Delta\\chi_{{\\mathrm{{rh}}}}$"
            "/$\\Delta\\chi_{{\\mathrm{{ax}}}}$: "
            f"{_ci_str(roa_val, 'rh_over_ax')}"
        )
    else:
        iso_val = float(molecule.susc.iso) * conv
        model_lines.append(
            f"$\\chi_{{\\mathrm{{iso}}}}$: "
            f"{_ci_str(iso_val, 'iso', '.3f')} {model_unit_label}"
        )
        dax = ax0 * conv
        ax_err_conv = _err_scaled("ax", conv)
        if ax_err_conv is not None:
            dax_line = format_compact_uncertainty(dax, ax_err_conv)
        else:
            dax_line = _fmt_val(dax, ".3f")
        model_lines.append(
            f"$\\Delta\\chi_{{\\mathrm{{ax}}}}$: {dax_line}"
        )
        model_lines.append(
            "$\\Delta\\chi_{{\\mathrm{{rh}}}}$"
            "/$\\Delta\\chi_{{\\mathrm{{ax}}}}$: "
            f"{_ci_str(roa_val, 'rh_over_ax', '.3f')}"
        )

    _a = _euler_str(molecule.susc.alpha, "alpha")
    _b = _euler_str(molecule.susc.beta, "beta")
    _g = _euler_str(molecule.susc.gamma, "gamma")
    model_lines.append(f"$R$ = ({_a}, {_b}, {_g})")

    if (
        spin is not None
        and orbit is not None and orbit > 0
        and total_J is not None
    ):
        T = float(molecule.susc.temperature)
        # C₀ without g_J² — g_J² enters compute_bleaney_params via g_sq_iso.
        norm_factor = compute_chi_prefactor(spin, total_J)
        b20, b22, scale_ax, scale_rh = compute_bleaney_params(
            chi_ax=float(molecule.susc.axiality) / norm_factor,
            chi_rh=float(molecule.susc.rhombicity) / norm_factor,
            temperature=T,
            spin=spin,
            orbit_L=orbit,
            total_J=total_J,
        )
        b20_err = _err_scaled("ax", scale_ax / norm_factor)
        b22_err = _err_scaled("rh", scale_rh / norm_factor)
        b20_str = (
            format_compact_uncertainty(b20, b20_err)
            if b20_err is not None
            else _fmt_val(b20, ".3f")
        )
        b22_str = (
            format_compact_uncertainty(b22, b22_err)
            if b22_err is not None
            else _fmt_val(b22, ".3f")
        )
        model_lines.append(f"$B^2_0$ = {b20_str} cm$^{{-1}}$")
        if float(molecule.susc.rhombicity) != 0.0:
            model_lines.append(f"$B^2_2$ = {b22_str} cm$^{{-1}}$")

    _bbox = dict(
        boxstyle="square,pad=0.0",
        facecolor="none",
        edgecolor="none",
        alpha=1.0,
    )
    _fs = spec.typography.tick_label

    fit_ann = (
        f"$R^2_\\mathrm{{adj}}$: {susc_model.adj_r2:.4f}\n"
        f"MAE: {susc_model.mae:.1f} ppm\n"
        f"RMSE: {susc_model.rmse:.1f} ppm"
    )
    ax.text(
        0.03, 0.97, fit_ann,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=_fs, color=palette.primary,
        bbox=_bbox,
    )

    ax.text(
        0.97, 0.03, "\n".join(model_lines),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=_fs, color=palette.primary,
        bbox=_bbox,
    )

    ax.grid(False)

    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        ax.plot(
            calc,
            expt,
            lw=0,
            marker=markers[label],
            color=palette.primary,
            markersize=glyphs.ms,
            markerfacecolor=_colors[label],
            markeredgecolor=palette.primary,
        )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    shared_min = np.min([x_lim, y_lim])
    shared_max = np.max([x_lim, y_lim])
    ax.set_xlim([shared_min, shared_max])
    ax.set_ylim([shared_min, shared_max])
    diag_line = ax.plot(
        [shared_min, shared_max],
        [shared_min, shared_max],
        color=palette.primary,
        lw=0.75,
    )[0]

    ax.set_xlabel(r"$\delta_\mathrm{theory}$ (ppm)")
    ax.set_ylabel(r"$\delta_\mathrm{exp}$ (ppm)")

    label_entries: list[tuple[str, float, float]] = []
    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        if average:
            label_entries.append((label, float(calc), float(expt)))
        else:
            for calc_value, exp_value in zip(calc, expt):
                label_entries.append(
                    (label, float(calc_value), float(exp_value))
                )

    ax.invert_xaxis()
    ax.invert_yaxis()

    if show_point_labels:
        resolve_label_layout(
            ax,
            label_entries,
            fontsize=scale.axis_label,
            marker_size=glyphs.ms,
            diag_line=diag_line,
            colors=_colors,
        )
    else:
        legend_handles = _build_element_legend_handles(
            _markers, palette, glyphs
        )
        ax.legend(handles=legend_handles, loc="best")

    render_figure(fig, save=save, show=show, save_name=save_name)

    if save and verbose:
        logger.info(
            "Chemical shift plot saved to %s", f"{save_name}.pdf"
        )

    return fig, ax
