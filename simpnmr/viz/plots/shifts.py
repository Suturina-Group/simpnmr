# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot chemical shifts and shift components.

Provides plotting utilities for fitted shifts, shift component contributions,
shift spreads, and temperature-dependent shift trends.
"""

import logging

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.constants as constants

from simpnmr.core.constants import periodic_table
from simpnmr.core.domain.experiment import Experiment
from simpnmr.core.domain.molecule import Molecule
from simpnmr.core.fitting import fit_models
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.violin import set_violin_colours
from simpnmr.viz.style.palette import SAFE_COLOURS
from simpnmr.viz.utils.format import isotope_format

logger = logging.getLogger(__name__)


def plot_fitted_shifts(
    molecule: Molecule,
    experiment: Experiment,
    susc_model: fit_models.SusceptibilityModel,
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

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Chemical shift plot saved to %s", f"{save_name}.pdf")

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
            r"{} $\delta$ (ppm)".format(isotope_format(molecule.nuclei[0].isotope)),
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

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if save and verbose:
        logger.info("Shift spread plot saved to %s", f"{save_name}.pdf")

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
            r"{} $\delta$ (ppm)".format(isotope_format(molecule.nuclei[0].isotope)),
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

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Shift component plot saved to %s", f"{save_name}.pdf")

    return fig, ax


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

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Shift vs Temperature plots saved to %s", f"{save_name}.pdf")

    return
