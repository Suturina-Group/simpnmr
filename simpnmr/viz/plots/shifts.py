# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot chemical shifts, shift components, and shift-temperature trends.

Provides plotting utilities for fitted shifts, shift component contributions,
shift spreads, and temperature-dependent experimental shift trends.
"""

import logging

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.violin import set_violin_colours
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.fmt import isotope_format

logger = logging.getLogger(__name__)


def plot_shift_spread(
    molecule: Molecule,
    experiment: Experiment | None = None,
    *,
    spec: PlotSpec,
    terms: list[str] = ["pc", "fc", "d"],
    order="ascending",
    isotope_filter: str | None = None,
    label_colors: dict[str, str] | None = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_spread.pdf",
    window_title: str = "Shift Spread",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
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

    _nuclei = [
        n for n in molecule.nuclei
        if isotope_filter is None or n.isotope == isotope_filter
    ]
    if not _nuclei:
        return None, None

    # Make plot
    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )
    glyphs = spec.glyphs
    scale = spec.skin_axes(ax)
    palette = spec.palette
    shift_colours = spec.shift_colours

    # chem_label → math_label mapping for color lookup
    _math_to_color: dict[str, str] = {}
    if label_colors:
        for nuc in _nuclei:
            if nuc.chem_label in label_colors:
                _math_to_color[nuc.chem_math_label] = label_colors[nuc.chem_label]

    # Total theoretical
    total = {nuc.chem_math_label: [] for nuc in _nuclei}
    # Grouped by chem_label
    # Remove diamagnetic part if diamagnetic term not included
    for nuc in _nuclei:
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
        def _exp_key(nuc):
            k = (nuc.chem_label, nuc.isotope) if nuc.isotope is not None else nuc.chem_label
            return k

        exps = {
            nuc.chem_math_label: experiment[_exp_key(nuc)].shift
            for nuc in _nuclei
            if _exp_key(nuc) in experiment
        }

        # Remove diamagnetic part of experiment if not included in terms list
        if "d" not in terms:
            for nuc in _nuclei:
                if _exp_key(nuc) in experiment:
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

    xvals = np.arange(1, len(_order) + 1)

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(terms) + 2)
    widthscaler = 1.0

    # Total Theoretical shift violin plot
    _violin = ax.violinplot(
        dataset=[total[o] for o in _order],
        positions=(xvals + width * widthscaler),
        widths=width,
        vert=True,
        showmeans=True,
    )
    set_violin_colours(_violin, shift_colours.total)
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
            color=palette.primary,
            lw=0,
            marker="o",
            markerfacecolor="none",
            markeredgecolor=palette.primary,
            markersize=(glyphs.ms if glyphs is not None else 7),
        )
        legend_markers = [
            lines.Line2D(
                [0],
                [0],
                color=palette.primary,
                lw=0,
                marker="o",
                markerfacecolor="None",
            )
        ] + legend_markers
        legend_labels = ["Exp."] + legend_labels

    widthscaler += 1

    # Fermi contact shift violin plot
    if "fc" in terms:
        fc = {nuc.chem_math_label: [] for nuc in _nuclei}
        for nuc in _nuclei:
            fc[nuc.chem_math_label].append(nuc.shift.fc)
        _violin = ax.violinplot(
            dataset=[fc[o] for o in _order],
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True,
        )
        widthscaler += 1
        set_violin_colours(_violin, shift_colours.fc)
        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )
        legend_labels.append("FC")

    # Pseudo contact shift violin plot
    if "pc" in terms:
        pc = {nuc.chem_math_label: [] for nuc in _nuclei}
        for nuc in _nuclei:
            pc[nuc.chem_math_label].append(nuc.shift.pc)
        _violin = ax.violinplot(
            dataset=[pc[o] for o in _order],
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True,
        )
        widthscaler += 1
        set_violin_colours(_violin, shift_colours.pc)
        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )
        legend_labels.append("PCS")

    # Diamagnetic shift violin plot
    if "d" in terms:
        dia = {nuc.chem_math_label: [] for nuc in _nuclei}
        for nuc in _nuclei:
            dia[nuc.chem_math_label].append(nuc.shift.dia)
        _violin = ax.violinplot(
            dataset=[dia[o] for o in _order],
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True,
        )
        widthscaler += 1
        set_violin_colours(_violin, shift_colours.dia)

        legend_markers.append(
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),
        )
        legend_labels.append("Dia.")

    # Add zero line to y axis
    ax.axhline(
        0.0,
        color=palette.primary,
        lw=(glyphs.line_lw if glyphs is not None else 0.5),
    )
    # Add grey gridlinesand ticks on x axis
    ax.grid(axis="x", ls="--", which="minor", linewidth=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Shift label, specify isotope/nucleus if only one type plotted
    if np.unique([nuc.isotope for nuc in _nuclei]).size == 1:
        ax.set_ylabel(
            r"{} $\delta$ (ppm)".format(isotope_format(_nuclei[0].isotope))
        )
    else:
        ax.set_ylabel(r"$\delta$ (ppm)")

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::1] + 0.5)
    ax.set_xticklabels(_order, rotation=90)
    if _math_to_color:
        for tick, lab in zip(ax.get_xticklabels(), _order):
            tick.set_color(_math_to_color.get(lab, palette.primary))
    ax.tick_params(axis="x", labelsize=scale.axis_label)

    ax.grid(axis="x", ls="--", which="minor", linewidth=0.2)
    ax.set_xlim(1, len(_order) + 1)
    ax.xaxis.set_tick_params("major", length=0)

    # Manually create custom legend
    # Violin plots dont support label kwarg
    ax.legend(legend_markers, legend_labels, loc="best")

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
    spec: PlotSpec,
    terms: list[str] = ["pc", "fc", "d"],
    order="ascending",
    isotope_filter: str | None = None,
    label_colors: dict[str, str] | None = None,
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_components.pdf",
    window_title: str = "Shift components",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
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

    _nuclei = [
        n for n in molecule.nuclei
        if isotope_filter is None or n.isotope == isotope_filter
    ]
    if not _nuclei:
        return None, None

    # Chemical math label → group size
    cl_to_al = {
        nuc.chem_math_label: len(
            [n for n in _nuclei if n.chem_math_label == nuc.chem_math_label]
        )
        for nuc in _nuclei
    }
    xvals = np.arange(len(cl_to_al))

    # Experiment
    _exp_math_labels: set[str] = set()  # math labels with real experimental data
    if experiment is not None:
        def _exp_key2(nuc):
            k = (nuc.chem_label, nuc.isotope) if nuc.isotope is not None else nuc.chem_label
            return k

        # Take average (skip nuclei absent from experiment)
        exps = dict.fromkeys(cl_to_al, 0)
        for nuc in _nuclei:
            if _exp_key2(nuc) not in experiment:
                continue
            exps[nuc.chem_math_label] += (
                experiment[_exp_key2(nuc)].shift / cl_to_al[nuc.chem_math_label]
            )
            _exp_math_labels.add(nuc.chem_math_label)

        if "d" not in terms:
            for nuc in _nuclei:
                if _exp_key2(nuc) not in experiment:
                    continue
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
    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )
    glyphs = spec.glyphs
    scale = spec.skin_axes(ax)
    palette = spec.palette
    shift_colours = spec.shift_colours

    _math_to_color: dict[str, str] = {}
    if label_colors:
        for nuc in _nuclei:
            if nuc.chem_label in label_colors:
                _math_to_color[nuc.chem_math_label] = label_colors[nuc.chem_label]

    xvals = np.arange(len(cl_to_al))

    widthscaler = 1

    # Total theoretical
    # Take average
    total = dict.fromkeys(cl_to_al, 0)
    for nuc in _nuclei:
        total[nuc.chem_math_label] += nuc.shift.total / cl_to_al[nuc.chem_math_label]

    if "d" not in terms:
        for nuc in _nuclei:
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
        color=shift_colours.total,
        lw=0,
        marker="x",
        markersize=(glyphs.ms if glyphs is not None else 7),
    )

    # Fermi contact part
    if "fc" in terms:
        # Take average
        fc = dict.fromkeys(cl_to_al, 0)
        for nuc in _nuclei:
            fc[nuc.chem_math_label] += nuc.shift.fc / cl_to_al[nuc.chem_math_label]
        ax.bar(
            (xvals + width * widthscaler),
            [fc[o] for o in order],
            width,
            label="FC",
            color=shift_colours.fc,
        )
        widthscaler += 1

    # Pseudocontact part
    if "pc" in terms:
        # Take average
        pc = dict.fromkeys(cl_to_al, 0)
        for nuc in _nuclei:
            pc[nuc.chem_math_label] += nuc.shift.pc / cl_to_al[nuc.chem_math_label]
        ax.bar(
            (xvals + width * widthscaler),
            [pc[o] for o in order],
            width,
            label="PCS",
            color=shift_colours.pc,
        )
        widthscaler += 1

    # Diamagnetic part
    if "d" in terms:
        # Take average
        dia = dict.fromkeys(cl_to_al, 0)
        for nuc in _nuclei:
            dia[nuc.chem_math_label] += nuc.shift.dia / cl_to_al[nuc.chem_math_label]
        ax.bar(
            (xvals + width * widthscaler),
            [dia[o] for o in order],
            width,
            label="Dia.",
            color=shift_colours.dia,
        )
        widthscaler += 1

    if experiment is not None:
        _exp_xpos = [
            i + 0.5 for i, o in enumerate(order) if o in _exp_math_labels
        ]
        _exp_yvals = [exps[o] for o in order if o in _exp_math_labels]
        ax.plot(
            _exp_xpos,
            _exp_yvals,
            label="Exp.",
            color=palette.primary,
            lw=0,
            marker="o",
            fillstyle="none",
            markersize=(glyphs.ms if glyphs is not None else 7),
        )

    ax.axhline(
        0.0,
        color=palette.primary,
        lw=(glyphs.line_lw if glyphs is not None else 0.5),
    )
    ax.grid(axis="x", ls="--", which="minor", linewidth=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    if np.unique([nuc.isotope for nuc in _nuclei]).size == 1:
        ax.set_ylabel(
            r"{} $\delta$ (ppm)".format(isotope_format(_nuclei[0].isotope))
        )
    else:
        ax.set_ylabel(r"$\delta$ (ppm)")

    ax.set_xlim([0, xvals[-1] + 1])

    ax.set_xticks(xvals + 0.5)
    ax.set_xticklabels(order, rotation=90)
    if _math_to_color:
        for tick, lab in zip(ax.get_xticklabels(), order):
            tick.set_color(_math_to_color.get(lab, palette.primary))
    ax.tick_params(axis="x", labelsize=scale.axis_label)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_tick_params("major", length=0)

    ax.legend(loc="best")

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
    spec: PlotSpec,
    tdep: str = "",
    save: bool = True,
    show: bool = True,
    save_name: str = "shiftxt_vs_t.pdf",
    window_title: str = "ShiftxT vs T",
    verbose: bool = True,
    assignment: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes,]]:
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

    # Plot both together and save limits
    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        window_title=window_title,
        layout="constrained",
    )

    glyphs = spec.glyphs
    spec.skin_axes(ax)
    palette = spec.palette
    colour_cycle = (
        palette.secondary,
        palette.highlight,
        palette.primary,
        palette.primary,
    )

    # Group signals of each experiment by assignment label
    labels = {
        signal.assignment
        for experiment in experiments
        for signal in experiment
        if signal.assignment is not None
    }

    colour_cycle_len = len(colour_cycle)
    colours = {
        label: colour_cycle[it % colour_cycle_len]
        for it, label in enumerate(sorted(labels))
    }

    for experiment in experiments:
        for signal in experiment.signals:
            if signal.assignment is None:
                continue
            ax.plot(
                experiment.temperature,
                signal.shift * experiment.temperature,
                lw=0,
                marker="x",
                markersize=(glyphs.ms if glyphs is not None else 7),
                label=signal.assignment,
                color=colours[signal.assignment],
            )

    ax.spines[["right", "top"]].set_visible(False)

    ax.set_xlabel(r"$T$ $\mathregular{(K)}$")

    ax.set_ylabel(r"$\delta_\mathregular{^1H}T$ (ppm K)")

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Shift vs Temperature plots saved to %s", f"{save_name}.pdf")

    return fig, (ax,)
