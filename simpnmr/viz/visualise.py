# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

import copy
import logging
import os

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.constants as constants
from numpy.typing import ArrayLike

from simpnmr import utils as ut
from simpnmr.core.constants import periodic_table
from simpnmr.core.constants.gammas import NUCLEAR_GAMMAS
from simpnmr.core.domain.experiment import Experiment
from simpnmr.core.domain.molecule import Molecule, Nucleus
from simpnmr.core.fitting import models
from simpnmr.core.spectrum.kernels import gaussian, lorentzian
from simpnmr.mappers import label_format as lf

logger = logging.getLogger(__name__)

SAFE_COLOURS = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "rgb(0  , 0  , 0)",
    "rgb(230, 159, 0)",
    "rgb(86 , 180, 233)",
    "rgb(0  , 158, 115)",
    "rgb(240, 228, 66)",
    "rgb(0  , 114, 178)",
    "rgb(213, 94 , 0)",
    "rgb(204, 121, 167)",
    "rgb(51 , 34 , 136)",
    "rgb(17 , 119, 51)",
    "rgb(68 , 170, 153)",
    "rgb(136, 204, 238)",
    "rgb(221, 204, 119)",
    "rgb(204, 102, 119)",
    "rgb(170, 68 , 153)",
    "rgb(136, 34 , 85)",
]


def set_violin_colours(violin: dict, color: str) -> None:
    """Sets violin-plot colours.

    Args:
        violin: Dictionary returned by ``Axes.violinplot``.
        color: Matplotlib color string.

    Returns:
        None.
    """
    for name, pc in violin.items():
        if name == "bodies":
            for part in pc:
                part.set_facecolor(color)
                part.set_edgecolor(color)
        else:
            pc.set_edgecolor(color)
    return


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
                hf_components[component][nuc.label] = nuc.A.dip[
                    ut.comp2ind(component[1:])
                ]
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
                hf_components[component][nuc.label] = nuc.A.tensor[
                    ut.comp2ind(component)
                ]
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

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            logger.info("Hyperfine plot saved to %s", save_name)

    if show:
        plt.show()

    return fig, ax


def plot_fitted_shifts(
    molecule: Molecule,
    experiment: Experiment,
    susc_model: models.SusceptibilityModel,
    average: bool = True,
    save: bool = True,
    show: bool = True,
    save_name: str = "nmr_shifts.png",
    window_title: str = "Fitted Shifts",
    susc_units: str = "A3",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots theoretical vs experimental shifts for a fitted susceptibility model.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experimental shift data.
        susc_model: Fitted susceptibility model.
        average: If ``True``, averages equivalent nuclei (same chemical label).
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

    if average:
        # Theoretical shifts, averaged over equivalent nuclei
        calc_shifts = {nuc.chem_label: nuc.shift.avg for nuc in unique_nuclei}
        # Experimental shifts, same order as theoretical
        exp = {label: experiment[label].shift for label in calc_shifts.keys()}
    else:
        # One signal per nucleus
        calc_shifts = {nuc.chem_label: [] for nuc in unique_nuclei}
        for nuc in molecule.nuclei:
            calc_shifts[nuc.chem_label].append(nuc.shift.total)

        # Experimental shifts, same order as theoretical
        exp = {
            label: [experiment[label].shift] * len(calc_shifts[label])
            for label in calc_shifts.keys()
        }

    # Element specific markers with consistent order
    _unique_elements = [
        ele
        for ele in periodic_table.elements
        if ele in [nuc.label_nn for nuc in unique_nuclei]
    ]
    _markers = {
        ele: mrkr for (ele, mrkr) in zip(_unique_elements, ["x", "o", "v", "s", "*"])
    }

    markers = {nuc.chem_label: _markers[nuc.label_nn] for nuc in molecule.nuclei}

    # if math labels are present then use these instead
    if all([len(nuc.chem_math_label) for nuc in molecule.nuclei]):
        for nuc in unique_nuclei:
            calc_shifts[nuc.chem_math_label] = calc_shifts.pop(nuc.chem_label)
            markers[nuc.chem_math_label] = markers.pop(nuc.chem_label)
            exp[nuc.chem_math_label] = exp.pop(nuc.chem_label)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 7), num=window_title)

    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        ax.plot(calc, expt, lw=0, marker=markers[label], color="k")
        if average:
            ax.text(calc, expt, label)
        else:
            for ca, ex in zip(calc, expt):
                ax.text(ca, ex, label)

    ax.set_xlabel("Theoretical Shift (ppm)")
    ax.set_ylabel("Experimental Shift (ppm)")

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="k", lw=0.75)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    ax.set_xlim([np.min([x_lim, y_lim]), np.max([x_lim, y_lim])])
    ax.set_ylim([np.min([x_lim, y_lim]), np.max([x_lim, y_lim])])

    if susc_units == "A3":
        conv = 1.0
        unit_label = r"$\mathregular{\AA^3}$"
        per_line = 3
    elif susc_units == "A3 mol-1":
        conv = constants.Avogadro
        unit_label = r"$\mathregular{\AA^3 \ mol^{-1}}$"
        per_line = 2
    elif susc_units == "cm3":
        conv = 1e-24
        unit_label = r"$\mathregular{cm^3}$"
        per_line = 3
    elif susc_units == "cm3 mol-1":
        conv = 1e-24 * constants.Avogadro / (4 * np.pi)
        unit_label = r"$\mathregular{cm^3 \ mol^{-1}}$"
        per_line = 2

    # Add fitted and fixed parameters to top of plot
    expression = ""
    for it, name in enumerate(susc_model.VARNAMES):
        val = float(susc_model.final_var_values[name]) * conv
        label = susc_model.VARNAMES_MM[name]

        if name in susc_model.fit_vars:
            err = susc_model.fit_stdev.get(name)
            if err is not None and err > 0:
                err_val = float(err) * conv
                par = int(round(err_val * 1000))
                expression += f"{label} = {val:.3f}({par}) "
            else:
                expression += f"{label} = {val:.3f} "
        else:
            expression += f"{label} = {val:.3f} "

        expression += unit_label + "     "
        if (
            not (it + 1) % per_line
            and len(susc_model.final_var_values.keys()) > 2
            and it != len(susc_model.VARNAMES) - 1
        ):
            expression += "\n"

    expression += "\n"

    expression += rf"$R^2_\mathregular{{adj.}}$ = {susc_model.adj_r2:.4f}       "
    expression += rf"$\mathrm{{MAE}} = {susc_model.mae:.3f}\ \mathrm{{ppm}}$       "
    expression += rf"$\mathrm{{RMSE}} = {susc_model.rmse:.3f}\ \mathrm{{ppm}}$"

    expression += f"\n{'-' * 50}\n"

    if not any(["ax" in susc_model.VARNAMES]):
        expression += (
            rf"$\Delta\chi_\mathregular{{ax}}$ = "
            f"{molecule.susc.axiality * conv:.3f} {unit_label}"
        )
        expression += (
            rf"  $\Delta\chi_\mathregular{{rh}}$ = "
            f"{molecule.susc.rhombicity * conv:.3f} {unit_label}"
        )
        expression += "\n"
    expression += rf"$\alpha$ = {molecule.susc.alpha:.2f}"
    expression += rf"  $\beta$ = {molecule.susc.beta:.2f}"
    expression += rf"  $\gamma$ = {molecule.susc.gamma:.2f}"

    ax.text(0.0, 1.02, s=expression, fontsize=11, transform=ax.transAxes)

    fig.tight_layout()

    for ax in fig.get_axes():
        ax.invert_xaxis()
        ax.invert_yaxis()

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            logger.info("Chemical shift plot saved to %s", save_name)

    if show:
        plt.show()

    return fig, ax


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
    closest_y = [
        y_intensity[ut.find_index_of_nearest(x_grid, sh)] for sh in sorted_shifts
    ]

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
        peak_index = ut.find_index_of_nearest(x_grid, shift)
        ax.plot(
            [x_grid[peak_index], x_grid[peak_index], label_x],
            [y_intensity[peak_index], label_barrier, label_y],
            linestyle="--",
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

    ax.set_xlabel(
        r"{} $\delta$ (ppm)".format(ut.isotope_format(isotope)), fontsize="18"
    )

    # Deactivate borders, y axis and y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines[["right", "top", "left"]].set_visible(False)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlim([np.max(shift_range), np.min(shift_range)])

    fig.tight_layout()

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            logger.info("Predicted spectrum saved to %s", save_name)

    if show:
        plt.show()

    # Write spectrum (ppm and normalized intensity) to CSV for external visualization
    df = pd.DataFrame({"shift (ppm)": x_grid, "intensity (a.u.)": y_intensity})
    csv_path = os.path.join(
        os.path.dirname(save_name),
        f"shift_vs_intensity_{molecule.susc.temperature:.2f}_K.csv",
    )
    df.to_csv(csv_path, index=False)

    return fig, ax


def plot_shift_spread(
    molecule: Molecule,
    experiment: Experiment | None = None,
    terms: list[str] = ["pc", "fc", "d"],
    order="ascending",
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_spread.png",
    window_title: str = "Shift Spread",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots the spread of theoretical shifts and selected components.

    Optionally overlays experimental shift values.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experimental data, or ``None`` to disable.
        terms: Shift components to include. Supported values include ``"fc"``
            (Fermi contact), ``"pc"`` (pseudocontact), and ``"d"`` (diamagnetic).
        order: Ordering of columns (``"ascending"`` or ``"descending"``).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    unique_chemlabels = {nuc.chem_math_label for nuc in molecule.nuclei}

    xvals = np.arange(1, len(unique_chemlabels) + 1)

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(terms) + 2)
    widthscaler = 1.0

    # Total theoretical
    total = {nuc.chem_math_label: [] for nuc in molecule.nuclei}
    # Grouped by chem_label
    # Remove diamagnetic part if diamagnetic term not included
    for nuc in molecule.nuclei:
        total[nuc.chem_math_label].append(nuc.shift.total)

    # Order using total theoretical shift
    if experiment is None:
        if order.lower() == "ascending":
            _order = [k for k, _ in sorted(total.items(), key=lambda item: item[1])]
        elif order.lower() == "descending":
            _order = [
                k
                for k, _ in sorted(
                    total.items(), key=lambda item: item[1], reverse=True
                )
            ]
    # or order using experimental shift
    else:
        exps = {
            nuc.chem_math_label: experiment[nuc.chem_label].shift
            for nuc in molecule.nuclei
        }

        # Remove diamagnetic part of experiment if not included in terms list
        if "d" not in terms:
            for nuc in molecule.nuclei:
                exps[nuc.chem_math_label] -= nuc.shift.dia

        # Order by low to high experimental shift
        # and store order as list of chemical math labels
        if order.lower() == "ascending":
            _order = [k for k, _ in sorted(exps.items(), key=lambda item: item[1])]
        elif order.lower() == "descending":
            _order = [
                k
                for k, _ in sorted(exps.items(), key=lambda item: item[1], reverse=True)
            ]

    # Total Theoretical shift violin plot
    _violin = ax.violinplot(
        dataset=[total[o] for o in _order],
        positions=(xvals + width * widthscaler),
        widths=width,
        vert=True,
        showmeans=True,
    )
    set_violin_colours(_violin, "black")
    legend_markers = [
        mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten())
    ]
    legend_labels = ["Total"]

    # Experiment circle marker plot
    if experiment is not None:
        ax.plot(
            (xvals + width * widthscaler),
            [exps[o] for o in _order],
            label="Exp.",
            color="k",
            lw=0,
            marker="o",
            markersize=7,
        )
        legend_markers = [
            lines.Line2D([0], [0], color="k", lw=0, marker="o", markerfacecolor="None")
        ] + legend_markers
        legend_labels = ["Exp."] + legend_labels

    widthscaler += 1

    # Fermi contact shift violin plot
    if "fc" in terms:
        fc = {nuc.chem_math_label: [] for nuc in molecule.nuclei}
        for nuc in molecule.nuclei:
            fc[nuc.chem_math_label].append(nuc.shift.fc)
        _violin = ax.violinplot(
            dataset=[fc[o] for o in _order],
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True,
        )
        widthscaler += 1
        set_violin_colours(_violin, "blue")
        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )
        legend_labels.append("Fermi")

    # Pseudo contact shift violin plot
    if "pc" in terms:
        pc = {nuc.chem_math_label: [] for nuc in molecule.nuclei}
        for nuc in molecule.nuclei:
            pc[nuc.chem_math_label].append(nuc.shift.pc)
        _violin = ax.violinplot(
            dataset=[pc[o] for o in _order],
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True,
        )
        widthscaler += 1
        set_violin_colours(_violin, "red")
        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )
        legend_labels.append("Pseudo")

    # Diamagnetic shift violin plot
    if "d" in terms:
        dia = {nuc.chem_math_label: [] for nuc in molecule.nuclei}
        for nuc in molecule.nuclei:
            dia[nuc.chem_math_label].append(nuc.shift.dia)
        _violin = ax.violinplot(
            dataset=[dia[o] for o in _order],
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True,
        )
        widthscaler += 1
        set_violin_colours(_violin, "green")

        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )
        legend_labels.append("Dia")

    # Add zero line to y axis
    ax.hlines(0.0, 1, len(unique_chemlabels) + 1, color="k", lw=0.5)
    # Add grey gridlinesand ticks on x axis
    ax.grid(axis="x", ls="--", which="minor")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Shift label, specify isotope/nucleus if only one type plotted
    if np.unique([nuc.isotope for nuc in molecule.nuclei]).size == 1:
        ax.set_ylabel(
            r"{} $\delta$ (ppm)".format(ut.isotope_format(molecule.nuclei[0].isotope)),
            fontsize="18",
        )
    else:
        ax.set_ylabel(r"$\delta$ (ppm)")

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::1] + 0.5)
    ax.set_xticklabels(_order, rotation=45, fontsize="18")

    ax.grid(axis="x", ls="--", which="minor")
    ax.set_xlim(0.5, len(_order) + 1.5)
    ax.xaxis.set_tick_params("major", length=0)

    # Manually create custom legend
    # Violin plots dont support label kwarg
    legend = ax.legend(
        legend_markers,
        legend_labels,
        loc="best",
        frameon=True,  # Enable the legend border
        fancybox=True,  # Rounded corners for the legend box (optional)
        framealpha=1.0,  # Fully opaque background
        fontsize="12",  # Adjust the font size if needed
    )
    legend.get_frame().set_facecolor(
        "white"
    )  # Set the background color of the legend to white
    legend.get_frame().set_edgecolor(
        "black"
    )  # Set the border color of the legend to black
    legend.get_frame().set_linewidth(1.2)  # Set the border thickness (optional)

    fig.tight_layout()
    fig.subplots_adjust(right=0.950)

    if save:
        out_dir = os.path.dirname(save_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig.savefig(save_name, dpi=400)
        if verbose:
            logger.info("Shift spread plot saved to %s", save_name)

    if show:
        plt.show()

    return fig, ax


def plot_shift_contrib(
    molecule: Molecule,
    experiment: Experiment | None,
    terms: list[str] = ["pc", "fc", "d"],
    order="ascending",
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_components.png",
    window_title: str = "Shift components",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots shift components alongside total and optional experimental values.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experimental data, or ``None`` to disable.
        terms: Shift components to include. Supported values include ``"fc"``,
            ``"pc"``, and ``"d"``.
        order: Ordering of columns (``"ascending"`` or ``"descending"``).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    # Chemical math label to list of nuclei labels
    cl_to_al = {
        nuc.chem_math_label: len(
            [
                nnuc.label
                for nnuc in molecule.nuclei
                if nnuc.chem_math_label == nuc.chem_math_label
            ]
        )
        for nuc in molecule.nuclei
    }
    xvals = np.arange(len(cl_to_al))

    # Experiment
    if experiment is not None:
        # Take average
        exps = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            exps[nuc.chem_math_label] += (
                experiment[nuc.chem_label].shift / cl_to_al[nuc.chem_math_label]
            )

        if "d" not in terms:
            for nuc in molecule.nuclei:
                exps[nuc.chem_math_label] -= (
                    nuc.shift.dia / cl_to_al[nuc.chem_math_label]
                )

        # Order by low to high experimental shift
        # and store order as list of chemical math labels
        if order.lower() == "ascending":
            order = [k for k, _ in sorted(exps.items(), key=lambda item: item[1])]
        elif order.lower() == "descending":
            order = [
                k
                for k, _ in sorted(exps.items(), key=lambda item: item[1], reverse=True)
            ]

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(terms) + 1)

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    # Chemical math label to list of nuclei labels
    cl_to_al = {
        nuc.chem_math_label: len(
            [
                nnuc.label
                for nnuc in molecule.nuclei
                if nnuc.chem_math_label == nuc.chem_math_label
            ]
        )
        for nuc in molecule.nuclei
    }
    xvals = np.arange(len(cl_to_al))

    widthscaler = 1

    # Total theoretical
    # Take average
    total = dict.fromkeys(cl_to_al, 0)
    for nuc in molecule.nuclei:
        total[nuc.chem_math_label] += nuc.shift.total / cl_to_al[nuc.chem_math_label]

    if "d" not in terms:
        for nuc in molecule.nuclei:
            total[nuc.chem_math_label] -= nuc.shift.dia / cl_to_al[nuc.chem_math_label]

    if experiment is None:
        if order.lower() == "ascending":
            order = [k for k, _ in sorted(total.items(), key=lambda item: item[1])]
        elif order.lower() == "descending":
            order = [
                k
                for k, _ in sorted(
                    total.items(), key=lambda item: item[1], reverse=True
                )
            ]

    ax.plot(
        (xvals + 0.5),
        [total[o] for o in order],
        label="Total",
        color="k",
        lw=0,
        marker="x",
        markersize=7,
    )

    # Fermi contact part
    if "fc" in terms:
        # Take average
        fc = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            fc[nuc.chem_math_label] += nuc.shift.fc / cl_to_al[nuc.chem_math_label]
        ax.bar(
            (xvals + width * widthscaler),
            [fc[o] for o in order],
            width,
            label="Fermi",
            color="b",
        )
        widthscaler += 1

    # Pseudocontact part
    if "pc" in terms:
        # Take average
        pc = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            pc[nuc.chem_math_label] += nuc.shift.pc / cl_to_al[nuc.chem_math_label]
        ax.bar(
            (xvals + width * widthscaler),
            [pc[o] for o in order],
            width,
            label="Pseudo",
            color="r",
        )
        widthscaler += 1

    # Diamagnetic part
    if "d" in terms:
        # Take average
        dia = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            dia[nuc.chem_math_label] += nuc.shift.dia / cl_to_al[nuc.chem_math_label]
        ax.bar(
            (xvals + width * widthscaler),
            [dia[o] for o in order],
            width,
            label="Dia.",
            color="g",
        )
        widthscaler += 1

    if experiment is not None:
        ax.plot(
            (xvals + 0.5),
            [exps[o] for o in order],
            label="Exp.",
            color="k",
            lw=0,
            marker="o",
            fillstyle="none",
            markersize=7,
        )

    ax.hlines(0.0, 0, len(total.values()), color="k", lw=0.5)
    ax.grid(axis="x", ls="--", which="minor")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    if np.unique([nuc.isotope for nuc in molecule.nuclei]).size == 1:
        ax.set_ylabel(
            r"{} $\delta$ (ppm)".format(ut.isotope_format(molecule.nuclei[0].isotope)),
            fontsize="18",
        )
    else:
        ax.set_ylabel(r"$\delta$ (ppm)")

    ax.set_xlim([-0.5, xvals[-1] + 1.5])

    ax.set_xticks(xvals + 0.5)
    ax.set_xticklabels(order, rotation=45, fontsize="18")

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_tick_params("major", length=0)

    legend = ax.legend(
        loc="best",
        frameon=True,  # Enable the legend border
        fancybox=True,  # Rounded corners for the legend box (optional)
        framealpha=1.0,  # Set legend background opacity (1.0 = fully opaque)
        fontsize="12",  # Adjust the font size if needed
    )
    legend.get_frame().set_facecolor(
        "white"
    )  # Set the background color of the legend to white
    legend.get_frame().set_edgecolor(
        "black"
    )  # Set the border color of the legend to black
    legend.get_frame().set_linewidth(1.2)  # Set the border thickness

    fig.tight_layout()
    fig.subplots_adjust(right=0.950)

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            logger.info("Shift component plot saved to %s", save_name)

    if show:
        plt.show()

    return fig, ax


def plot_relax_contrib(
    molecule: Molecule,
    experiment: Experiment | None,
    order="ascending",
    save: bool = True,
    show: bool = True,
    save_name: str = "relaxation_contributions.png",
    window_title: str = "Relaxation Contributions",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots contributions to relaxation rates alongside experimental values.

    Args:
        molecule: Molecule containing theoretical relaxation data.
        experiment: Experimental relaxation data, or ``None`` to disable.
        order: Ordering of columns (``"ascending"`` or ``"descending"``).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.

    Notes:
        This function currently defines the interface and docstring but does not yet
        implement the plotting logic.
    """


def plot_shift_tdep(
    experiments: list[Experiment],
    tdep: str = "",
    save: bool = True,
    show: bool = True,
    save_name: str = "shiftxt_vs_t.png",
    window_title: str = "ShiftxT vs T",
    verbose: bool = True,
    assignment: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes]]:
    """Plots experimental shift temperature dependence.

    By default, plots ``shift * T`` versus ``T`` for each assignment label.

    Args:
        experiments: Experiment objects, one per temperature.
        tdep: Temperature-dependence mode (reserved for future use).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.
        assignment: If ``True``, uses assignment labels for legend entries.

    Returns:
        A tuple ``(fig, ax)``.
    """

    labelfontsize = 13

    # Plot both together and save limits
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    # Group signals of each experiment by assignment label
    labels = {signal.assignment for experiment in experiments for signal in experiment}

    colours = {label: SAFE_COLOURS[it] for it, label in enumerate(labels)}

    # grouped_shifts = {
    #     label: []
    #     for label in labels
    # }

    # for experiment in experiments:
    #     for signal in experiment:
    #         grouped_shifts[signal.assignment].append(signal.shift)

    # temperatures

    for experiment in experiments:
        for signal in experiment.signals:
            ax.plot(
                experiment.temperature,
                signal.shift * experiment.temperature,
                marker="x",
                label=signal.assignment,
                color=colours[signal.assignment],
            )

    ax.spines[["right", "top"]].set_visible(False)

    ax.set_xlabel(r"$T$ $\mathregular{(K)}$", fontsize=labelfontsize)

    ax.set_ylabel(r"$\delta_\mathregular{^1H}T$ (ppm K)", fontsize=labelfontsize)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            logger.info("Shift vs Temperature plots saved to %s", save_name)
    if show:
        plt.show()

    return


def plot_isoaxrho(
    vals: dict,
    errs: dict,
    params: dict | None,
    inv_t: np.ndarray,
    show: bool = True,
    save: bool = True,
    y_label: str = "ChiT",
    save_name: str = "susceptibility_components.pdf",
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
            label="SimpNMR Fit",
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
                caption_lines.append(rf"$Intercept = {p['intercept']:.1f}$")

            if "slope" in p and "slope_err" in p:
                caption_lines.append(
                    rf"$Slope = {p['slope']:.1f} \pm {p['slope_err']:.1f}$"
                )
            elif "slope" in p:
                caption_lines.append(rf"$Slope = {p['slope']:.1f}$")

            if "tip" in p:
                caption_lines.append(rf"$TIP = {p['tip']:.3g}$")

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
        ax.set_xlabel(r"$1/T$ K$^{-1}$", fontsize=14)
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
        top_ax.set_xlabel(r"$T$ K", fontsize=14)
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

        if save:
            root, _ = os.path.splitext(save_name)
            comp_save_name = f"{root}_{component}{'.pdf'}"
            fig.savefig(comp_save_name)
            if verbose:
                logger.info("Temperature dependence plot saved to %s", comp_save_name)

        if show:
            plt.show()


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

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            logger.info("Hyperfine plot saved to %s", save_name)

    if show:
        plt.show()

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
                    nuc.A.dip[ut.comp2ind(component[1:])]
                )
                legend_labels[component] = (
                    rf"$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$"
                )
        else:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.tensor[ut.comp2ind(component)]
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

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            logger.info("Hyperfine spread plot saved to %s", save_name)

    if show:
        plt.show()

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
        y_sim_intensity[ut.find_index_of_nearest(x_grid, sh)] for sh in shifts
    ]

    # Construct deconvoluted (processed experimental) spectrum intensities (y-axis)
    y_deconv_intensity = np.zeros_like(x_grid)

    # Accumulate deconvoluted spectrum intensities
    for signal in experiment.signals:
        # Convert experimental linewidth from Hz to ppm
        exp_width_ppm = signal.width / (
            NUCLEAR_GAMMAS[lf.remove_numbers(isotope)] * experiment.magnetic_field
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
    ax[-1].set_xlabel(r"{} $\delta$ (ppm)".format(ut.isotope_format(isotope)))

    # Remove y-axis ticks, labels, and spines for a cleaner stacked-spectra layout
    for axis in ax:
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.spines[["right", "top", "left"]].set_visible(False)

    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            logger.info("Spectra saved to %s", save_name)

    if show:
        plt.show()

    return fig, ax
