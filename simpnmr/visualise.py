# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

import copy
import os

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.constants as constants
from numpy.typing import ArrayLike, NDArray

from . import main, models, outputs
from . import utils as ut
from .scripts import fit_vt
from .scripts.coords_tools import atoms

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


def gaussian(x: ArrayLike, fwhm: float, b: float, area: float) -> NDArray:
    """Evaluates a Gaussian peak with a given position, width, and area.

    The functional form is:

        ``g(x) = area/(c*sqrt(2*pi)) * exp(-(x-b)^2/(2*c^2))``

    where ``c = fwhm/(2*sqrt(2*ln(2)))``.

    Args:
        x: Coordinate grid.
        fwhm: Full width at half maximum.
        b: Peak position.
        area: Peak area.

    Returns:
        Array of ``g(x)`` values.
    """

    c = fwhm / (2 * np.sqrt(2 * np.log(2)))

    a = 1.0 / (c * np.sqrt(2 * np.pi))

    gaus = a * np.exp(-((x - b) ** 2) / (2 * c**2))

    gaus *= area

    return gaus


def lorentzian(x: ArrayLike, fwhm, x0, area) -> NDArray:
    """Evaluates a Lorentzian peak with a given position, width, and area.

    The functional form is:

        ``L(x) = (0.5*area*fwhm/pi) / ((x-x0)^2 + (0.5*fwhm)^2)``

    Args:
        x: Coordinate grid.
        fwhm: Full width at half maximum.
        x0: Peak position.
        area: Peak area.

    Returns:
        Array of ``L(x)`` values.
    """

    lor = 0.5 * fwhm / np.pi
    lor *= 1.0 / ((x - x0) ** 2 + (0.5 * fwhm) ** 2)

    lor *= area

    return lor


def plot_hyperfine(
    nuclei: list[main.Nucleus],
    components: list[str],
    save: bool = False,
    show: bool = True,
    save_name: str = "hyperfines.dat",
    verbose: bool = False,
    window_title: str = "Hyperfine data",
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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
                hf_components[component][nuc.label] = nuc.A.dip[0, 0] - nuc.A.dip[1, 1]  # noqa
                complabels[component] = r"$A_\mathregular{dip, ax}$"
        elif component == "rho":
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[0, 0] + nuc.A.dip[1, 1]  # noqa
                complabels[component] = r"$A_\mathregular{dip, rho}$"
        elif "d" in component:
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[
                    ut.comp2ind(component[1:])
                ]  # noqa
                complabels[component] = (
                    rf"$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$"  # noqa
                )
        elif component in ["x", "y", "z"]:  # eigenvalues
            _to_ind = {"x": 0, "y": 1, "z": 2}
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.eigvals[_to_ind[component]]  # noqa
                complabels[component] = rf"$A_{{\mathregular{{{component}}}}}$"  # noqa
        else:
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.tensor[
                    ut.comp2ind(component)
                ]  # noqa
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
            ut.cprint("Hyperfine plot saved to {}".format(save_name), "blue")

    if show:
        plt.show()

    return fig, ax


def plot_fitted_shifts(
    molecule: main.Molecule,
    experiment: main.Experiment,
    susc_model: models.SusceptibilityModel,
    average: bool = True,
    save: bool = True,
    show: bool = True,
    save_name: str = "nmr_shifts.png",
    window_title: str = "Fitted Shifts",
    susc_units: str = "A3",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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
        ele for ele in atoms.elements if ele in [nuc.label_nn for nuc in unique_nuclei]
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
        expression += "{} = {:.3f} ".format(
            susc_model.VARNAMES_MM[name],
            susc_model.final_var_values[name] * conv,
        )
        if name in susc_model.fit_vars.keys():
            expression += r"$\pm$ "
            expression += "{:.3f} ".format(susc_model.fit_stdev[name] * conv)
        expression += unit_label + "     "
        if (
            not (it + 1) % per_line
            and len(susc_model.final_var_values.keys()) > 2
            and it != len(susc_model.VARNAMES) - 1
        ):  # noqa
            expression += "\n"

    expression += "\n"

    expression += rf"$r^2_\mathregular{{adj.}}$ = {susc_model.adj_r2:.5f}       "
    expression += rf"$\mathrm{{MAE}} = {susc_model.mae:.5f}\ \mathrm{{ppm}}$       "
    expression += rf"$\mathrm{{RMSE}} = {susc_model.rmse:.5f}\ \mathrm{{ppm}}$"

    expression += "\n-------------------------------------------------\n"

    if not any(["ax" in susc_model.VARNAMES]):
        expression += rf"$\Delta\chi_\mathregular{{ax}}$ = {molecule.susc.axiality * conv:.3f} {unit_label}"  # noqa
        expression += rf"  $\Delta\chi_\mathregular{{rh}}$ = {molecule.susc.rhombicity * conv:.3f} {unit_label}"  # noqa
        expression += "\n"
    expression += rf"$\alpha$ = {molecule.susc.alpha:.3f}"
    expression += rf"  $\beta$ = {molecule.susc.beta:.3f}"
    expression += rf"  $\gamma$ = {molecule.susc.gamma:.3f}"

    ax.text(0.0, 1.02, s=expression, fontsize=11, transform=ax.transAxes)

    fig.tight_layout()

    for ax in fig.get_axes():
        ax.invert_xaxis()
        ax.invert_yaxis()

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            ut.cprint(f"\n Chemical shift plot saved to \n {save_name}\n", "cyan")

    if show:
        plt.show()

    return fig, ax


def plot_pred_spectrum(
    molecule: main.Molecule,
    isotope: str,
    shift_range: ArrayLike,
    save: bool = True,
    show: bool = True,
    save_name: str = "predicted_spectrum.png",
    window_title: str = "Predicted Spectrum",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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

    ppm_grid = np.linspace(np.min(shift_range), np.max(shift_range), 100000)
    _total = np.zeros(np.shape(ppm_grid))
    for nuc in molecule.nuclei:
        if nuc.isotope == isotope:
            _total += lorentzian(ppm_grid, nuc.shift.lw, nuc.shift.avg, 1)

    # Normalise spectrum
    _total /= np.max(_total)

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    # Spectrum trace
    ax.plot(ppm_grid, _total, color="k")

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
    closest_y = [_total[ut.find_index_of_nearest(ppm_grid, sh)] for sh in sorted_shifts]

    # Marker at shift peak position
    ax.plot(sorted_shifts, closest_y, lw=0, marker="x", color="k", markersize=7)

    # Horizontal line 10% above the highest peak
    hline_y = 1.1 * np.max(_total)
    ax.hlines(
        hline_y,
        np.min(shift_range),
        np.max(shift_range),
        linestyle="-",
        color="black",
        linewidth=0.8,
        alpha=0.7,
    )

    # Minimum acceptable distance between labels
    label_mindist = 0.03 * (np.max(shift_range) - np.min(shift_range))

    # Iteratively shift all label x positions so that they will not touch
    # (i.e. are not within label_mindist)

    # Calculate initial distance matrix
    adj_label_xvals = copy.copy(sorted_shifts)
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

    # Peak label y position (20% above max peak)
    label_y = 1.2 * np.max(_total)

    # Add label and dashed lines
    for shift, label, label_x in zip(sorted_shifts, sorted_labels, adj_label_xvals):  # noqa
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
        peak_index = ut.find_index_of_nearest(ppm_grid, shift)
        ax.plot(
            [ppm_grid[peak_index], ppm_grid[peak_index], label_x],
            [_total[peak_index], hline_y, label_y],
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
            ut.cprint(f"\n Predicted spectrum saved to \n {save_name}\n", "cyan")

    if show:
        plt.show()

    # Write spectrum (ppm and normalized intensity) to CSV for external visualization
    df = pd.DataFrame({"shift (ppm)": ppm_grid, "intensity (a.u.)": _total})
    csv_path = os.path.join(
        os.path.dirname(save_name),
        f"shift_vs_intensity_{molecule.susc.temperature:.2f}_K.csv",
    )
    df.to_csv(csv_path, index=False)

    return fig, ax


def plot_shift_spread(
    molecule: main.Molecule,
    experiment: main.Experiment | None = None,
    terms: list[str] = ["pc", "fc", "d"],
    order="ascending",
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_spread.png",
    window_title: str = "Shift Spread",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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
            fillstyle="none",
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
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),  # noqa
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
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),  # noqa
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
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),  # noqa
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
        fig.savefig(save_name, dpi=400)
        if verbose:
            ut.cprint(f"\n Shift spread plot saved to \n {save_name}\n", "cyan")

    if show:
        plt.show()

    return fig, ax


def plot_shift_contrib(
    molecule: main.Molecule,
    experiment: main.Experiment | None,
    terms: list[str] = ["pc", "fc", "d"],
    order="ascending",
    save: bool = True,
    show: bool = True,
    save_name: str = "shift_components.png",
    window_title: str = "Shift components",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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
            )  # noqa

        if "d" not in terms:
            for nuc in molecule.nuclei:
                exps[nuc.chem_math_label] -= (
                    nuc.shift.dia / cl_to_al[nuc.chem_math_label]
                )  # noqa

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
        total[nuc.chem_math_label] += nuc.shift.total / cl_to_al[nuc.chem_math_label]  # noqa

    if "d" not in terms:
        for nuc in molecule.nuclei:
            total[nuc.chem_math_label] -= nuc.shift.dia / cl_to_al[nuc.chem_math_label]  # noqa

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
            fc[nuc.chem_math_label] += nuc.shift.fc / cl_to_al[nuc.chem_math_label]  # noqa
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
            pc[nuc.chem_math_label] += nuc.shift.pc / cl_to_al[nuc.chem_math_label]  # noqa
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
            dia[nuc.chem_math_label] += nuc.shift.dia / cl_to_al[nuc.chem_math_label]  # noqa
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
            ut.cprint(f"\n Shift component plot saved to \n {save_name}\n", "cyan")

    if show:
        plt.show()

    return fig, ax


def plot_relax_contrib(
    molecule: main.Molecule,
    experiment: main.Experiment | None,
    order="ascending",
    save: bool = True,
    show: bool = True,
    save_name: str = "relaxation_contributions.png",
    window_title: str = "Relaxation Contributions",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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
    experiments: list[main.Experiment],
    tdep: str = "",
    save: bool = True,
    show: bool = True,
    save_name: str = "shiftxt_vs_t.png",
    window_title: str = "ShiftxT vs T",
    verbose: bool = True,
    assignment: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes]]:  # noqa
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
            ut.cprint(f"\n Shift vs T plots saved to \n {save_name}\n", "cyan")
    if show:
        plt.show()

    return


def plot_isoaxrho(
    spin: float,
    molecules: list[main.Molecule],
    save: bool = True,
    method: str | None = None,
    vt_variables: dict | None = None,
    show: bool = True,
    save_name: str = "iso_ax_rho_tdep.png",
    window_title: str = "Isotropic, Axial, and Rhombic susceptibilities",  # noqa
    verbose: bool = True,
    y_mode: str = "chiT",
    out_file: str = "slope_intercept.csv",
    susc_models: list[models.SusceptibilityModel] = [],
    tip_corrections=None,
) -> tuple[plt.Figure, tuple[plt.Axes]]:  # noqa
    """Plots temperature dependence of isotropic/axial/rhombic susceptibility.

    The function supports plotting either ``chi*T`` or ``chi`` versus temperature.

    Args:
        spin: Spin quantum number used by the spin-Hamiltonian model.
        molecules: Molecules containing non-empty `susc` attributes.
        save: If ``True``, saves the plot to `save_name`.
        method: Spin-Hamiltonian fitting method.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.
        y_mode: Quantity to plot on the y-axis (``"chiT"`` or ``"chi"``).
        out_file: Output CSV file name for slope/intercept results when `y_mode` is
            ``"chiT"``.
        susc_models: Optional fitted susceptibility models used to provide parameter
            uncertainties.

    Returns:
        A tuple ``(fig, ax)``.
    """

    def plot_component(
        temperature,
        chi_component,
        y_mode: str,
        ax: plt.Axes,
        bax: plt.Axes,
        name: str,
        use_errors: bool = False,
        chi_errors=None,
        fit_results: dict | None = None,
    ):
        y_labels = {
            "chiT": {
                "iso": r"$\chi_\mathregular{iso}T$ (reduced)",
                "axial": r"$\Delta\chi_\mathregular{ax}T$ (reduced)",
                "rhombic": r"$\Delta\chi_\mathregular{rh}T$ (reduced)",
            },
            "chi": {
                "iso": r"$\chi_\mathregular{iso}$ (reduced)",
                "axial": r"$\Delta\chi_\mathregular{ax}$ (reduced)",
                "rhombic": r"$\Delta\chi_\mathregular{rh}$ (reduced)",
            },
        }

        if not use_errors:
            ax.plot(temperature, chi_component, lw=0, marker="x", ms=5, color="black")
        else:
            ax.errorbar(
                temperature,
                chi_component,
                yerr=chi_errors,
                lw=0,
                elinewidth=1.5,
                fillstyle="none",
                color="black",
                capsize=1.5,
                marker="x",
                ms=5,
            )

        ax.set_xlabel(r"$T$ / K")
        ax.set_ylabel(y_labels[y_mode][name])

        ax.spines[["right", "top"]].set_visible(False)

        if fit_results is not None:
            ax.plot(
                temperature,
                fit_results["intercept"] + fit_results["slope"] / temperature,
            )

            params = (
                rf"$Intercept = {fit_results['intercept']:.1f} "
                rf"\pm {fit_results['intercept_err']:.1f}$"
                + "\n"
                + rf"$Slope = {fit_results['slope']:.1f} "
                rf"\pm {fit_results['slope_err']:.1f}$"
            )  # noqa
            _adj_r2 = fit_results.get("adj_r2")
            if _adj_r2 is None or (isinstance(_adj_r2, float) and np.isnan(_adj_r2)):
                _adj_r2_txt = "N/A"
            else:
                _adj_r2_txt = f"{_adj_r2:.3f}"

            bax.annotate(
                text=rf"$r^2_\mathregular{{adj}} = {_adj_r2_txt}$" + "\n" + params,  # noqa
                xy=(0.1, 0.5),
                xycoords="axes fraction",
            )

        # Add minor ticks
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Remove all bottom plot features to give an effective text box
        bax.spines[["right", "top", "left", "bottom"]].set_visible(False)
        bax.xaxis.set_ticks([])
        bax.yaxis.set_ticks([])

        if fit_results is not None:
            return fit_results["intercept"], fit_results["slope"]
        return 0, 0

    fig, ax = plt.subplots(
        2,
        3,
        figsize=[10, 3.5],
        num=window_title,
        gridspec_kw={"height_ratios": [10, 1]},
    )

    isotropic = np.array([molecule.susc.iso for molecule in molecules])
    temperature = np.array([molecule.susc.temperature for molecule in molecules])  # noqa

    axial = np.array([molecule.susc.axiality for molecule in molecules])
    rhombic = np.array([molecule.susc.rhombicity for molecule in molecules])

    if len(susc_models) and isinstance(susc_models[0], models.IsoAxRhoFitter):
        if "iso" in susc_models[0].fix_vars:
            iso_err = np.zeros(len(susc_models))
        else:
            iso_err = np.array([model.fit_stdev["iso"] for model in susc_models])  # noqa
        if "ax" in susc_models[0].fix_vars:
            axial_err = np.zeros(len(susc_models))
        else:
            axial_err = np.array([model.fit_stdev["ax"] for model in susc_models])  # noqa
        if "rho_over_ax" in susc_models[0].fix_vars:
            rhombic_over_ax_err = np.zeros(len(susc_models))
        else:
            rhombic_over_ax_err = np.array(
                [model.fit_stdev["rho_over_ax"] for model in susc_models]
            )  # noqa
        use_errors = True
    else:
        use_errors = False
        iso_err = np.zeros(len(susc_models))
        axial_err = np.zeros(len(susc_models))
        rhombic_over_ax_err = np.zeros(len(susc_models))

    # Use per-component weighting flags: do not weight if the parameter was fixed
    if len(susc_models) and isinstance(susc_models[0], models.IsoAxRhoFitter):
        _fix = susc_models[0].fix_vars
    else:
        _fix = {}
    use_iso_errors = use_errors and ("iso" not in _fix)
    use_ax_errors = use_errors and ("ax" not in _fix)
    use_rh_errors = use_errors and ("rho_over_ax" not in _fix)

    # Isotropic
    iso_vals, iso_errs, iso_fit = fit_vt.compute_chit_linear_parameters(
        method,
        vt_variables,
        "iso",
        spin,
        temperature,
        isotropic,
        y_mode,
        use_errors=use_iso_errors,
        chi_errors=iso_err,
        tip_corrections=tip_corrections,
    )
    plot_component(
        temperature,
        iso_vals,
        y_mode,
        ax[0, 0],
        ax[1, 0],
        "iso",
        use_errors=use_iso_errors,
        chi_errors=iso_errs,
        fit_results=iso_fit,
    )

    # Axial
    axial_vals, axial_errs, axial_fit = fit_vt.compute_chit_linear_parameters(
        method,
        vt_variables,
        "ax",
        spin,
        temperature,
        axial,
        y_mode,
        use_errors=use_ax_errors,
        chi_errors=axial_err,
        tip_corrections=tip_corrections,
    )
    plot_component(
        temperature,
        axial_vals,
        y_mode,
        ax[0, 1],
        ax[1, 1],
        "axial",
        use_errors=use_ax_errors,
        chi_errors=axial_errs,
        fit_results=axial_fit,
    )

    # Rhombic
    rh_vals, rh_errs, rh_fit = fit_vt.compute_chit_linear_parameters(
        method,
        vt_variables,
        "rho",
        spin,
        temperature,
        rhombic,
        y_mode,
        use_errors=use_rh_errors,
        chi_errors=rhombic_over_ax_err,
        tip_corrections=tip_corrections,
    )
    plot_component(
        temperature,
        rh_vals,
        y_mode,
        ax[0, 2],
        ax[1, 2],
        "rhombic",
        use_errors=use_rh_errors,
        chi_errors=rh_errs,
        fit_results=rh_fit,
    )

    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(f"\n {y_mode} vs T plots saved to \n {save_name}\n", "cyan")
    if show:
        plt.show()

    fits = [iso_fit, axial_fit, rh_fit]

    if y_mode.lower() == "chit":
        outputs.save_slope_intercept(fits, out_file)

    return fig, ax


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
    )  # noqa

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(f"Hyperfine plot saved to {save_name}", "cyan")

    if show:
        plt.show()

    return


def plot_hyperfine_spread(
    nuclei: list[main.Nucleus],
    components: list[str] | None = None,
    save: bool = False,
    show: bool = True,
    save_name: str = "hyperfines.png",
    window_title: str = "Hyperfine Components",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:  # noqa
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
                )  # noqa
                legend_labels[component] = r"$A_\mathregular{dip, ax}$"
        elif component == "rho":
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.dip[0, 0] + nuc.A.dip[1, 1]
                )  # noqa
                legend_labels[component] = r"$A_\mathregular{dip, rho}$"
        elif "d" in component:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.dip[ut.comp2ind(component[1:])]
                )  # noqa
                legend_labels[component] = (
                    rf"$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$"  # noqa
                )
        else:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(
                    nuc.A.tensor[ut.comp2ind(component)]
                )  # noqa
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
            mpatches.Patch(color=_violin["bodies"][0].get_facecolor().flatten()),  # noqa
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
            ut.cprint("Hyperfine spread plot saved to {}".format(save_name), "blue")

    if show:
        plt.show()

    return fig, ax


def plot_raw_deconv_pred(
    molecule: main.Molecule,
    experiment: main.Experiment,
    save: bool = True,
    show: bool = True,
    save_name: str = "iso_ax_rho_tdep.png",
    window_title: str = "Raw, Deconvoluted, and Predicted Spectra",  # noqa
    verbose: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes]]:  # noqa
    """Plots raw, deconvoluted, and predicted spectra.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experiment containing the raw spectrum and deconvolution results.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    fig, ax = plt.subplots(3, 1, figsize=(8, 5.5), num=window_title, sharex=True)

    ax[0].set_title(
        "Simulation",
        loc="left",
        fontdict={"size": "smaller"},
        pad=-6,
    )
    ax[1].set_title(
        "Paramagnetic Signals",
        loc="left",
        fontdict={"size": "smaller"},
        pad=-6,
    )
    ax[2].set_title(
        "Full Spectrum",
        loc="left",
        fontdict={"size": "smaller"},
        pad=-6,
    )

    isotope = experiment.isotope  # e.g. "1H"
    element = "".join(filter(str.isalpha, isotope))  # e.g. "H"

    ppm_min = np.min(experiment.spectrum[::4, 0])
    ppm_max = np.max(experiment.spectrum[::4, 0])
    ppm_grid = np.linspace(ppm_min, ppm_max, 100000)

    sim_total = np.zeros_like(ppm_grid)
    for nucleus in molecule.nuclei:
        if nucleus.isotope == isotope:
            # `nucleus.shift.lw` is already a linewidth in ppm (FWHM)
            sim_total += lorentzian(ppm_grid, nucleus.shift.lw, nucleus.shift.avg, 1)

    if np.max(sim_total) != 0:
        sim_total /= np.max(sim_total)

    # Marker positions for labels (same logic as plot_pred_spectrum)
    avg_shifts = {
        nucleus.chem_math_label: nucleus.shift.avg
        for nucleus in molecule.nuclei
        if nucleus.isotope == isotope
    }

    sorted_shifts_labels = sorted(avg_shifts.items(), key=lambda x: x[1])
    labels = [label for label, _ in sorted_shifts_labels]
    shifts = [shift for _, shift in sorted_shifts_labels]
    closest_y = [sim_total[ut.find_index_of_nearest(ppm_grid, sh)] for sh in shifts]

    ax[0].plot(ppm_grid, sim_total, lw=1, color="k")
    ax[0].plot(shifts, closest_y, lw=0, marker="x", color="k", markersize=7)

    # ---- Label placement/leader lines (match plot_pred_spectrum, but do NOT change y-scale) ----
    # Ensure labels match shifts in sorted order (already sorted above, but keep explicit)
    sorted_shifts_labels = sorted(avg_shifts.items(), key=lambda x: x[1])
    sorted_labels = [label for label, _ in sorted_shifts_labels]
    sorted_shifts = [shift for _, shift in sorted_shifts_labels]

    # Grid y value closest to peak position
    closest_y = [
        sim_total[ut.find_index_of_nearest(ppm_grid, sh)] for sh in sorted_shifts
    ]

    # Marker at shift peak position
    ax[0].plot(sorted_shifts, closest_y, lw=0, marker="x", color="k", markersize=7)

    # Use current y-limits so we do not rescale/compress the spectrum
    y_bottom, y_top = ax[0].get_ylim()

    # Horizontal guide line kept within existing y-limits (so it can't change scaling)
    hline_y = y_bottom + 0.98 * (y_top - y_bottom)
    ax[0].hlines(
        hline_y,
        ppm_min,
        ppm_max,
        linestyle="-",
        color="black",
        linewidth=0.8,
        alpha=0.7,
    )

    # Minimum acceptable distance between labels (in ppm)
    label_mindist = 0.03 * (ppm_max - ppm_min)

    # Iteratively shift all label x positions so that they will not touch
    adj_label_xvals = copy.copy(sorted_shifts)
    distance = np.subtract.outer(adj_label_xvals, adj_label_xvals)
    np.fill_diagonal(distance, np.inf)

    while len(np.where(abs(distance) < label_mindist)[0]):
        xlocs, ylocs = np.where(abs(distance) < label_mindist)
        for x, y in zip(xlocs, ylocs):
            if y > x:
                adj_label_xvals[x] -= label_mindist / 2
                adj_label_xvals[y] += label_mindist / 2

        distance = np.subtract.outer(adj_label_xvals, adj_label_xvals)
        np.fill_diagonal(distance, np.inf)

    # Place labels slightly above the axis (axes fraction > 1) so scaling is unchanged
    label_y_axes = 1.02
    xaxis_transform = ax[0].get_xaxis_transform()  # x in data, y in axes fraction

    for shift, label, label_x in zip(sorted_shifts, sorted_labels, adj_label_xvals):
        if not (ppm_min < shift < ppm_max):
            continue

        # Label text above the axis
        ax[0].text(
            label_x,
            label_y_axes,
            label,
            rotation="vertical",
            ha="center",
            va="bottom",
            fontsize=15,
            transform=xaxis_transform,
            clip_on=False,
        )

        # Leader lines: peak -> hline (data coords), then hline -> label (mixed coords)
        peak_index = ut.find_index_of_nearest(ppm_grid, shift)
        peak_y = sim_total[peak_index]

        # Segment 1: vertical to the hline
        ax[0].plot(
            [ppm_grid[peak_index], ppm_grid[peak_index]],
            [peak_y, hline_y],
            linestyle="--",
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

        # Segment 2: horizontal along the hline to the label x-position
        ax[0].plot(
            [ppm_grid[peak_index], label_x],
            [hline_y, hline_y],
            linestyle="--",
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

        # Segment 3: from hline up to the label (doesn't affect data scaling)
        ax[0].annotate(
            "",
            xy=(label_x, hline_y),
            xycoords="data",
            xytext=(label_x, label_y_axes),
            textcoords=xaxis_transform,
            arrowprops={
                "arrowstyle": "-",
                "linestyle": "--",
                "color": "black",
                "linewidth": 0.8,
                "alpha": 0.6,
            },
            annotation_clip=False,
        )

    deconv_total = np.zeros_like(ppm_grid)
    for signal in experiment.signals:
        # `signal.width` is provided in Hz (FWHM)
        width_ppm = signal.width / (
            ut.NUCLEAR_GAMMAS[element] * experiment.magnetic_field
        )

        deconv_total += signal.l_to_g * lorentzian(
            ppm_grid, width_ppm, signal.shift, signal.area
        )
        deconv_total += (1 - signal.l_to_g) * gaussian(
            ppm_grid, width_ppm, signal.shift, signal.area
        )

    ax[1].plot(ppm_grid, deconv_total, lw=1, color="k")
    ax[2].plot(
        experiment.spectrum[::4, 0], experiment.spectrum[::4, 1], lw=1, color="k"
    )

    for axis in ax:
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.set_xlim(
            [np.max(experiment.spectrum[:, 0]), np.min(experiment.spectrum[:, 0])]
        )
        axis.spines[["right", "top", "left"]].set_visible(False)

    ax[2].set_xlabel(r"{} $\delta$ (ppm)".format(ut.isotope_format(isotope)))
    ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(f"\n Spectra saved to\n {save_name}\n", "cyan")

    if show:
        plt.show()

    return
