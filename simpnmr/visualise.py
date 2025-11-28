import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as lines
from scipy.optimize import curve_fit
import scipy.constants as constants
import numpy as np
from numpy.typing import NDArray, ArrayLike
import xyz_py.atomic as atomic
import copy
import pandas as pd
import os
import math

from .scripts import chi_plot as chi_helpers

from . import utils as ut
from . import main
from . import models


SAFE_COLOURS = [
    'C0',
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'C9',
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
    "rgb(136, 34 , 85)"
]

def set_violin_colours(violin: dict, color: str) -> None:
    '''
    Sets violin plot colours.

    Parameters
    ----------
    violin: dict[str, Collection]
        Returned value of ax.violinplot
    color: str
        Colour name
    '''
    for name, pc in violin.items():
        if name == 'bodies':
            for part in pc:
                part.set_facecolor(color)
                part.set_edgecolor(color)
        else:
            pc.set_edgecolor(color)
    return


def gaussian(x: ArrayLike, fwhm: float, b: float, area: float) -> NDArray:
    """
    Gaussian g(x) with given peak position (b), fwhm, and area

    g(x) = area/(c*sqrt(2pi)) * exp(-(x-b)**2/(2c**2))

    c = fwhm/(2*np.sqrt(2*np.log(2)))

    Parameters
    ----------
    x : array_like
        Continuous variable
    fwhm: float
        Full Width at Half-Maximum
    b : float
        Peak position
    area : float
        Area of Gaussian function

    Return
    ------
    list[float]
        g(x) at each value of x
    """

    c = fwhm / (2 * np.sqrt(2 * np.log(2)))

    a = 1. / (c * np.sqrt(2 * np.pi))

    gaus = a * np.exp(-(x - b)**2 / (2 * c**2))

    gaus *= area

    return gaus


def lorentzian(x: ArrayLike, fwhm, x0, area) -> NDArray:
    """
    Lotenztian L(x) with given peak position (b), fwhm, and area

    L(x) = (0.5*area*fwhm/pi) * 1/((x-x0)**2 + (0.5*fwhm)**2)

    Parameters
    ----------
    x : array_like
        Continuous variable
    fwhm: float
        Full Width at Half-Maximum
    x0 : float
        Peak position
    area : float
        Area of Lorentzian function

    Return
    ------
    list[float]
        L(x) at each value of x
    """

    lor = 0.5 * fwhm / np.pi
    lor *= 1. / ((x - x0)**2 + (0.5 * fwhm)**2)

    lor *= area

    return lor


def plot_hyperfine(nuclei: list[main.Nucleus], components: list[str],
                   save: bool = False, show: bool = True,
                   save_name: str = 'hyperfines.dat', verbose: bool = False,
                   window_title: str = 'Hyperfine data') -> tuple[plt.Figure, list[plt.Axes]]: # noqa
    '''
    Plots Hyperfine coupling components for each nucleus label

    Parameters
    ----------
    nuclei: list[Nucleus]
        Nuclei to plot
    components: list[str]
        Name(s) of hyperfine components to plot\n
        (xx, yy, iso, ax, rho, dxy, ...)
    save: bool, default False
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'hyperfines.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Hyperfine components'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    complabels = {
        component: ''
        for component in components
    }

    hf_components = {
        component: {
            nuc.label: []
            for nuc in nuclei
        }
        for component in components
    }

    for component in components:
        if component == 'iso':
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.iso
                complabels[component] = r'$A_\mathregular{iso}$'
        elif component == 'ax':
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[0, 0] - nuc.A.dip[1, 1] # noqa
                complabels[component] = r'$A_\mathregular{dip, ax}$'
        elif component == 'rho':
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[0, 0] + nuc.A.dip[1, 1] # noqa
                complabels[component] = r'$A_\mathregular{dip, rho}$'
        elif 'd' in component:
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.dip[ut.comp2ind(component[1:])] # noqa
                complabels[component] = fr'$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$'  # noqa
        elif component in ['x', 'y', 'z']:  # eigenvalues
            _to_ind = {'x': 0, 'y': 1, 'z': 2}
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.eigvals[_to_ind[component]] # noqa
                complabels[component] = fr'$A_{{\mathregular{{{component}}}}}$'  # noqa
        else:
            for nuc in nuclei:
                hf_components[component][nuc.label] = nuc.A.tensor[ut.comp2ind(component)] # noqa
                complabels[component] = fr'$A_\mathregular{{{component}}}$'

    fig, ax = plt.subplots(
        1,
        1,
        num=window_title
    )

    n_nuclei = len(nuclei)

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(components) + 1)
    shifts = [width + width * it for it in range(n_nuclei)]

    # Tick positions
    xvals = np.arange(1, n_nuclei + 1)

    unique_nuclabels = np.unique([
        nuc.label
        for nuc in nuclei
    ])

    xvals = np.arange(1, len(unique_nuclabels) + 1)

    for (comp_name, comp_values), shift in zip(hf_components.items(), shifts):
        ax.bar(
            xvals + shift,
            list(comp_values.values()),
            width=width,
            label=complabels[comp_name]
        )

    if len(hf_components) < 11:
        step = 1
    else:
        step = 2

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::step] + 0.5)
    labels = list(unique_nuclabels)
    ax.set_xticklabels(labels[::step])
    ax.grid(axis='x', ls='--', which='minor')
    ax.set_xlim(0.5, len(labels) + 1.5)
    ax.xaxis.set_tick_params('major', length=0)

    ax.hlines(0, 0.5, len(labels) + 1.5, lw=0.5, color='k')

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend()

    ax.set_ylabel(r'Hyperfine Coupling (ppm Å$^\mathregular{-3}$)')
    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(
                'Hyperfine plot saved to {}'.format(save_name),
                'blue'
            )

    if show:
        plt.show()

    return fig, ax


def plot_fitted_shifts(molecule: main.Molecule, experiment: main.Experiment,
                       susc_model: models.SusceptibilityModel,
                       average: bool = True, save: bool = True,
                       show: bool = True, save_name: str = 'nmr_shifts.png',
                       window_title: str = 'Fitted Shifts',
                       susc_units: str = 'A3',
                       verbose: bool = True) -> tuple[plt.Figure, list[plt.Axes]]: # noqa
    '''
    Plots theoretical and experimental shifts against each other

    Parameters
    ----------
    molecule: main.Molecule
        Object containing all theoretical shift data
    experiment: main.Experiment
        Object containing all experimental shift data
    susc_model: models.SusceptibilityModel
        Object containing fit information
    average: bool, default True
        If True, average nuclei with the same chemical label
    save: bool, default True
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'nmr_shifts.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Fitted Shifts'
        Title of figure window, not of plot
    susc_units: str, {'A3', 'A3 mol-1', 'cm3 mol-1', 'cm3'}
        Units to use for susceptibility
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    seen = set()
    unique_nuclei = [
        seen.add(nuc.chem_label) or nuc
        for nuc in molecule.nuclei
        if nuc.chem_label not in seen
    ]

    if average:
        # Theoretical shifts, averaged over equivalent nuclei
        calc_shifts = {
            nuc.chem_label: nuc.shift.avg
            for nuc in unique_nuclei
        }
        # Experimental shifts, same order as theoretical
        exp = {
            label: experiment[label].shift
            for label in calc_shifts.keys()
        }
    else:
        # One signal per nucleus
        calc_shifts = {
            nuc.chem_label: []
            for nuc in unique_nuclei
        }
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
        for ele in atomic.elements
        if ele in [nuc.label_nn for nuc in unique_nuclei]
    ]
    _markers = {
        ele: mrkr for (ele, mrkr) in zip(
            _unique_elements,
            ['x', 'o', 'v', 's', '*']
        )
    }

    markers = {
        nuc.chem_label: _markers[nuc.label_nn] for nuc in molecule.nuclei
    }

    # if math labels are present then use these instead
    if all([len(nuc.chem_math_label) for nuc in molecule.nuclei]):
        for nuc in unique_nuclei:
            calc_shifts[nuc.chem_math_label] = calc_shifts.pop(nuc.chem_label)
            markers[nuc.chem_math_label] = markers.pop(nuc.chem_label)
            exp[nuc.chem_math_label] = exp.pop(nuc.chem_label)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 7), num=window_title)

    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        ax.plot(
            calc,
            expt,
            lw=0,
            marker=markers[label],
            color='k'
        )
        if average:
            ax.text(
                calc,
                expt,
                label
            )
        else:
            for ca, ex in zip(calc, expt):
                ax.text(
                    ca,
                    ex,
                    label
                )

    ax.set_xlabel('Theoretical Shift (ppm)')
    ax.set_ylabel('Experimental Shift (ppm)')

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', lw=.75)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    ax.set_xlim([np.min([x_lim, y_lim]), np.max([x_lim, y_lim])])
    ax.set_ylim([np.min([x_lim, y_lim]), np.max([x_lim, y_lim])])

    if susc_units == 'A3':
        conv = 1.
        unit_label = r'$\mathregular{\AA^3}$'
        per_line = 3
    elif susc_units == 'A3 mol-1':
        conv = constants.Avogadro
        unit_label = r'$\mathregular{\AA^3 \ mol^{-1}}$'
        per_line = 2
    elif susc_units == 'cm3':
        conv = 1E-24
        unit_label = r'$\mathregular{cm^3}$'
        per_line = 3
    elif susc_units == 'cm3 mol-1':
        conv = 1E-24 * constants.Avogadro / (4 * np.pi)
        unit_label = r'$\mathregular{cm^3 \ mol^{-1}}$'
        per_line = 2

    # Add fitted and fixed parameters to top of plot
    expression = ''
    for it, name in enumerate(susc_model.VARNAMES):
        expression += '{} = {:.3f} '.format(
            susc_model.VARNAMES_MM[name],
            susc_model.final_var_values[name] * conv,
        )
        if name in susc_model.fit_vars.keys():
            expression += r'$\pm$ '
            expression += '{:.3f} '.format(susc_model.fit_stdev[name] * conv)
        expression += unit_label + '     '
        if not (it + 1) % per_line and len(susc_model.final_var_values.keys()) > 2 and it != len(susc_model.VARNAMES) - 1: # noqa
            expression += '\n'

    expression += '\n'

    expression += rf'$r^2_\mathregular{{adj.}}$ = {susc_model.adj_r2:.5f}       '
    expression += rf'$\mathrm{{MAE}} = {susc_model.mae:.5f}\ \mathrm{{ppm}}$       '
    expression += rf'$\mathrm{{RMSE}} = {susc_model.rmse:.5f}\ \mathrm{{ppm}}$'

    expression += '\n-------------------------------------------------\n'

    if not any(['ax' in susc_model.VARNAMES]):
        expression += rf'$\Delta\chi_\mathregular{{ax}}$ = {molecule.susc.axiality * conv:.3f} {unit_label}' # noqa
        expression += rf'  $\Delta\chi_\mathregular{{rh}}$ = {molecule.susc.rhombicity * conv:.3f} {unit_label}' # noqa
        expression += '\n'
    expression += rf'$\alpha$ = {molecule.susc.alpha:.3f}'
    expression += rf'  $\beta$ = {molecule.susc.beta:.3f}'
    expression += rf'  $\gamma$ = {molecule.susc.gamma:.3f}'

    ax.text(
        0.0, 1.02, s=expression, fontsize=11, transform=ax.transAxes
    )

    fig.tight_layout()
    
    for ax in fig.get_axes():
        ax.invert_xaxis()
        ax.invert_yaxis()

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            ut.cprint(
                f'\n Chemical shift plot saved to \n {save_name}\n',
                'cyan'
            )

    if show:
        plt.show()

    return fig, ax


def plot_pred_spectrum(molecule: main.Molecule,
                       isotope: str,
                       shift_range: ArrayLike,
                       save: bool = True, show: bool = True,
                       save_name: str = 'predicted_spectrum.png',
                       window_title: str = 'Predicted Spectrum',
                       verbose: bool = True) -> tuple[plt.Figure, list[plt.Axes]]: # noqa
    '''
    Plots predicted spectrum

    Parameters
    ----------
    molecule: main.Molecule
        Object containing all theoretical shift data
    isotope: str
        Isotope to plot spectrum for
    shift_range: array_like
        Upper and lower bounds of chemical shift
    save: bool, default True
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'predicted_spectrum.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Fitted Shifts'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    ppm_grid = np.linspace(
        np.min(shift_range),
        np.max(shift_range),
        100000
    )
    _total = np.zeros(np.shape(ppm_grid))
    for nuc in molecule.nuclei:
        if nuc.isotope == isotope:
            _total += lorentzian(
                ppm_grid,
                nuc.shift.lw,
                nuc.shift.avg,
                1
            )

    # Normalise spectrum
    _total /= np.max(_total)

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    # Spectrum trace
    ax.plot(ppm_grid, _total, color='k')

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
        _total[ut.find_index_of_nearest(ppm_grid, sh)]
        for sh in sorted_shifts
    ]

    # Marker at shift peak position
    ax.plot(
        sorted_shifts,
        closest_y,
        lw=0,
        marker='x',
        color='k',
        markersize=7
    )

    # Horizontal line 10% above the highest peak
    hline_y = 1.1 * np.max(_total)
    ax.hlines(
        hline_y,
        np.min(shift_range),
        np.max(shift_range),
        linestyle='-',
        color='black',
        linewidth=0.8,
        alpha=0.7
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
    for shift, label, label_x in zip(sorted_shifts, sorted_labels, adj_label_xvals): # noqa

        # Add label to plot
        ax.text(
            label_x,
            label_y,
            label,
            rotation='vertical',
            ha='center',
            va='bottom',
            fontsize='18'
        )

        # Draw segmented line from peak to label via horizontal line
        peak_index = ut.find_index_of_nearest(ppm_grid, shift)
        ax.plot(
            [ppm_grid[peak_index], ppm_grid[peak_index], label_x],
            [_total[peak_index], hline_y, label_y],
            linestyle='--',
            color='black',
            linewidth=0.8,
            alpha=0.6
        )

    ax.set_xlabel(r'{} $\delta$ (ppm)'.format(
        ut.isotope_format(isotope)),
        fontsize='18'
    )

    # Deactivate borders, y axis and y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines[['right', 'top', 'left']].set_visible(False)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlim(
        [
            np.max(shift_range),
            np.min(shift_range)
        ]
    )

    fig.tight_layout()

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            ut.cprint(
                f'\n Predicted spectrum saved to \n {save_name}\n',
                'cyan'
            )

    if show:
        plt.show()

    # Write spectrum data (ppm and normalized intensity) to CSV for external visualization
    df = pd.DataFrame({'shift (ppm)': ppm_grid, 'intensity (a.u.)': _total})
    csv_path = os.path.join(os.path.dirname(save_name), f'shift_vs_intensity_{molecule.susc.temperature:.2f}_K.csv')
    df.to_csv(csv_path, index=False)

    return fig, ax


def plot_shift_spread(molecule: main.Molecule,
                      experiment: main.Experiment | None = None,
                      terms: list[str] = ['pc', 'fc', 'd'],
                      order='ascending', save: bool = True, show: bool = True,
                      save_name: str = 'shift_spread.png',
                      window_title: str = 'Shift Spread',
                      verbose: bool = True) -> tuple[plt.Figure, list[plt.Axes]]: # noqa
    '''
    Plots spread of theoretical shift and components, alongside experimental
    shift value

    Parameters
    ----------
    molecule: main.Molecule
        Object containing all theoretical shift data
    experiment: main.Experiment | None
        Object containing all experimental shift data, or None to disable
    terms: list[str]
        String terms 'fc' 'pc' and 'd' to include in plot ()
    order: str {'descending', 'ascending'}
        Plot columns ordered by total calculated shift or experimental shift.\n
        This switches between descending and ascending order.
    save: bool, default True
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'nmr_shifts.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Fitted Shifts'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    unique_chemlabels = {nuc.chem_math_label for nuc in molecule.nuclei}

    xvals = np.arange(1, len(unique_chemlabels) + 1)

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(terms) + 2)
    widthscaler = 1.

    # Total theoretical
    total = {
        nuc.chem_math_label: []
        for nuc in molecule.nuclei
    }
    # Grouped by chem_label
    # Remove diamagnetic part if diamagnetic term not included
    for nuc in molecule.nuclei:
        total[nuc.chem_math_label].append(nuc.shift.total)

    # Order using total theoretical shift
    if experiment is None:
        if order.lower() == 'ascending':
            _order = [
                k for k, _
                in sorted(total.items(), key=lambda item: item[1])
            ]
        elif order.lower() == 'descending':
            _order = [
                k for k, _
                in sorted(
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
        if 'd' not in terms:
            for nuc in molecule.nuclei:
                exps[nuc.chem_math_label] -= nuc.shift.dia

        # Order by low to high experimental shift
        # and store order as list of chemical math labels
        if order.lower() == 'ascending':
            _order = [
                k for k, _
                in sorted(exps.items(), key=lambda item: item[1])
            ]
        elif order.lower() == 'descending':
            _order = [
                k for k, _
                in sorted(exps.items(), key=lambda item: item[1], reverse=True)
            ]

    # Total Theoretical shift violin plot
    _violin = ax.violinplot(
        dataset=[total[o] for o in _order],
        positions=(xvals + width * widthscaler),
        widths=width,
        vert=True,
        showmeans=True
    )
    set_violin_colours(_violin, 'black')
    legend_markers = [
        mpatches.Patch(color=_violin['bodies'][0].get_facecolor().flatten())
    ]
    legend_labels = ['Total']

    # Experiment circle marker plot
    if experiment is not None:
        ax.plot(
            (xvals + width * widthscaler),
            [exps[o] for o in _order], 
            label='Exp.',
            color='k',
            lw=0,
            marker='o',
            fillstyle='none',
            markersize=7
        )
        legend_markers = [
            lines.Line2D(
                [0], [0], color='k', lw=0, marker='o', markerfacecolor='None'
            )
        ] + legend_markers
        legend_labels = ['Exp.'] + legend_labels

    widthscaler += 1

    # Fermi contact shift violin plot
    if 'fc' in terms:
        fc = {
            nuc.chem_math_label: []
            for nuc in molecule.nuclei
        }
        for nuc in molecule.nuclei:
            fc[nuc.chem_math_label].append(nuc.shift.fc)
        _violin = ax.violinplot(
            dataset=[fc[o] for o in _order], 
            positions=(xvals + width * widthscaler),
            widths=width,
            vert=True,
            showmeans=True
        )
        widthscaler += 1
        set_violin_colours(_violin, 'blue')
        legend_markers.append(
            mpatches.Patch(color=_violin['bodies'][0].get_facecolor().flatten()), # noqa
        )
        legend_labels.append('Fermi')

    # Pseudo contact shift violin plot
    if 'pc' in terms:
        pc = {
            nuc.chem_math_label: []
            for nuc in molecule.nuclei
        }
        for nuc in molecule.nuclei:
            pc[nuc.chem_math_label].append(nuc.shift.pc)
        _violin = ax.violinplot(
            dataset=[pc[o] for o in _order], 
            positions=(xvals + width * widthscaler), 
            widths=width,
            vert=True,
            showmeans=True
        )
        widthscaler += 1
        set_violin_colours(_violin, 'red')
        legend_markers.append(
            mpatches.Patch(color=_violin['bodies'][0].get_facecolor().flatten()), # noqa
        )
        legend_labels.append('Pseudo')

    # Diamagnetic shift violin plot
    if 'd' in terms:
        dia = {
            nuc.chem_math_label: []
            for nuc in molecule.nuclei
        }
        for nuc in molecule.nuclei:
            dia[nuc.chem_math_label].append(nuc.shift.dia)
        _violin = ax.violinplot(
            dataset=[dia[o] for o in _order], 
            positions=(xvals + width * widthscaler), 
            widths=width,
            vert=True,
            showmeans=True
        )
        widthscaler += 1
        set_violin_colours(_violin, 'green')

        legend_markers.append(
            mpatches.Patch(color=_violin['bodies'][0].get_facecolor().flatten()), # noqa
        )
        legend_labels.append('Dia')

    # Add zero line to y axis
    ax.hlines(0., 1, len(unique_chemlabels) + 1, color='k', lw=.5)
    # Add grey gridlinesand ticks on x axis
    ax.grid(axis='x', ls='--', which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Shift label, specify isotope/nucleus if only one type plotted
    if np.unique([nuc.isotope for nuc in molecule.nuclei]).size == 1:
        ax.set_ylabel(r'{} $\delta$ (ppm)'.format(
            ut.isotope_format(molecule.nuclei[0].isotope)), fontsize='18'
        )
    else:
        ax.set_ylabel(r'$\delta$ (ppm)')

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::1] + 0.5)
    ax.set_xticklabels(_order, rotation=45, fontsize='18')
    
    ax.grid(axis='x', ls='--', which='minor')
    ax.set_xlim(0.5, len(_order) + 1.5)
    ax.xaxis.set_tick_params('major', length=0)

    # Manually create custom legend
    # Violin plots dont support label kwarg
    legend = ax.legend(
        legend_markers,
        legend_labels,
        loc='best',
        frameon=True,                  # Enable the legend border
        fancybox=True,                 # Rounded corners for the legend box (optional)
        framealpha=1.0,                # Fully opaque background
        fontsize='12'                  # Adjust the font size if needed
    )
    legend.get_frame().set_facecolor('white')    # Set the background color of the legend to white
    legend.get_frame().set_edgecolor('black')    # Set the border color of the legend to black
    legend.get_frame().set_linewidth(1.2)        # Set the border thickness (optional)

    fig.tight_layout()
    fig.subplots_adjust(right=0.950)

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            ut.cprint(
                f'\n Shift spread plot saved to \n {save_name}\n',
                'cyan'
            )

    if show:
        plt.show()

    return fig, ax


def plot_shift_contrib(molecule: main.Molecule,
                       experiment: main.Experiment | None,
                       terms: list[str] = ['pc', 'fc', 'd'],
                       order='ascending',
                       save: bool = True, show: bool = True,
                       save_name: str = 'shift_components.png',
                       window_title: str = 'Shift components',
                       verbose: bool = True) -> tuple[plt.Figure, list[plt.Axes]]: # noqa
    '''
    Plots components of theoretical shift alongside experimental shift value

    Parameters
    ----------
    molecule: main.Molecule
        Object containing all theoretical shift data
    experiment: main.Experiment | None
        Object containing all experimental shift data, or None to disable
    terms: list[str]
        String terms 'fc' 'pc' and 'd'
    order: str {'descending', 'ascending'}
        Plot columns ordered by total calculated shift or experimental shift.\n
        This switches between descending and ascending order.
    save: bool, default True
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'nmr_shifts.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Fitted Shifts'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    # Chemical math label to list of nuclei labels
    cl_to_al = {
        nuc.chem_math_label: len([
            nnuc.label
            for nnuc in molecule.nuclei
            if nnuc.chem_math_label == nuc.chem_math_label
        ])
        for nuc in molecule.nuclei
    }
    xvals = np.arange(len(cl_to_al))

    # Experiment
    if experiment is not None:
        # Take average
        exps = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            exps[nuc.chem_math_label] += experiment[nuc.chem_label].shift / cl_to_al[nuc.chem_math_label] # noqa

        if 'd' not in terms:
            for nuc in molecule.nuclei:
                exps[nuc.chem_math_label] -= nuc.shift.dia / cl_to_al[nuc.chem_math_label] # noqa

        # Order by low to high experimental shift
        # and store order as list of chemical math labels
        if order.lower() == 'ascending':
            order = [
                k for k, _
                in sorted(exps.items(), key=lambda item: item[1])
            ]
        elif order.lower() == 'descending':
            order = [
                k for k, _
                in sorted(exps.items(), key=lambda item: item[1], reverse=True)
            ]

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(terms) + 1)

    # Make plot
    fig, ax = plt.subplots(1, 1, num=window_title, figsize=(8, 5.5))

    # Chemical math label to list of nuclei labels
    cl_to_al = {
        nuc.chem_math_label: len([
            nnuc.label
            for nnuc in molecule.nuclei
            if nnuc.chem_math_label == nuc.chem_math_label
        ])
        for nuc in molecule.nuclei
    }
    xvals = np.arange(len(cl_to_al))

    widthscaler = 1

    # Total theoretical
    # Take average
    total = dict.fromkeys(cl_to_al, 0)
    for nuc in molecule.nuclei:
        total[nuc.chem_math_label] += nuc.shift.total / cl_to_al[nuc.chem_math_label] # noqa

    if 'd' not in terms:
        for nuc in molecule.nuclei:
            total[nuc.chem_math_label] -= nuc.shift.dia / cl_to_al[nuc.chem_math_label] # noqa

    if experiment is None:
        if order.lower() == 'ascending':
            order = [
                k for k, _
                in sorted(total.items(), key=lambda item: item[1])
            ]
        elif order.lower() == 'descending':
            order = [
                k for k, _
                in sorted(
                    total.items(), key=lambda item: item[1], reverse=True
                )
            ]

    ax.plot(
        (xvals + 0.5), 
        [total[o] for o in order],
        label='Total',
        color='k',
        lw=0,
        marker='x',
        markersize=7
    )

    # Fermi contact part
    if 'fc' in terms:
        # Take average
        fc = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            fc[nuc.chem_math_label] += nuc.shift.fc / cl_to_al[nuc.chem_math_label] # noqa
        ax.bar(
            (xvals + width * widthscaler), 
            [fc[o] for o in order],
            width,
            label='Fermi',
            color='b'
        )
        widthscaler += 1

    # Pseudocontact part
    if 'pc' in terms:
        # Take average
        pc = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            pc[nuc.chem_math_label] += nuc.shift.pc / cl_to_al[nuc.chem_math_label] # noqa
        ax.bar(
            (xvals + width * widthscaler), 
            [pc[o] for o in order],
            width,
            label='Pseudo',
            color='r'
        )
        widthscaler += 1

    # Diamagnetic part
    if 'd' in terms:
        # Take average
        dia = dict.fromkeys(cl_to_al, 0)
        for nuc in molecule.nuclei:
            dia[nuc.chem_math_label] += nuc.shift.dia / cl_to_al[nuc.chem_math_label] # noqa
        ax.bar(
            (xvals + width * widthscaler), 
            [dia[o] for o in order],
            width,
            label='Dia.',
            color='g'
        )
        widthscaler += 1

    if experiment is not None:
        ax.plot(
            (xvals + 0.5), 
            [exps[o] for o in order],
            label='Exp.',
            color='k',
            lw=0,
            marker='o',
            fillstyle='none',
            markersize=7
        )

    ax.hlines(0., 0, len(total.values()), color='k', lw=.5)
    ax.grid(axis='x', ls='--', which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    if np.unique([nuc.isotope for nuc in molecule.nuclei]).size == 1:
        ax.set_ylabel(r'{} $\delta$ (ppm)'.format(
            ut.isotope_format(molecule.nuclei[0].isotope)), fontsize='18'
        )
    else:
        ax.set_ylabel(r'$\delta$ (ppm)')

    ax.set_xlim([-0.5, xvals[-1] + 1.5])

    ax.set_xticks(xvals + 0.5)
    ax.set_xticklabels(order, rotation=45, fontsize='18')   

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_tick_params('major', length=0)

    legend = ax.legend(
        loc='best',
        frameon=True,                 # Enable the legend border
        fancybox=True,                # Rounded corners for the legend box (optional)
        framealpha=1.0,               # Set legend background opacity (1.0 = fully opaque)
        fontsize='12'                 # Adjust the font size if needed
    )
    legend.get_frame().set_facecolor('white')   # Set the background color of the legend to white
    legend.get_frame().set_edgecolor('black')   # Set the border color of the legend to black
    legend.get_frame().set_linewidth(1.2)       # Set the border thickness

    fig.tight_layout()
    fig.subplots_adjust(right=0.950)

    if save:
        fig.savefig(save_name, dpi=400)
        if verbose:
            ut.cprint(
                f'\n Shift component plot saved to \n {save_name}\n',
                'cyan'
            )

    if show:
        plt.show()

    return fig, ax


def plot_shift_tdep(experiments: list[main.Experiment], tdep: str = '',
                    save: bool = True, show: bool = True,
                    save_name: str = 'shiftxt_vs_t.png',
                    window_title: str = 'ShiftxT vs T', verbose: bool = True,
                    assignment: bool = True) -> tuple[plt.Figure, tuple[plt.Axes]]: # noqa
    '''
    Plots experimental shift multiplied by temperature against temperature

    Parameters
    ----------
    experiments: list[main.Experiment]
        Experiment objects, one per temperature
    tdep: str {'ShiftT_vs_T', 'Shift_vs_1/T'}
        Type of temperature dependence to plot
    save: bool, default True
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'shiftxt_vs_t.png'
        If save is True, will save plot to this file name
    window_title: str, default 'ShiftxT vs T'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    labelfontsize = 13

    # Plot both together and save limits
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    # Group signals of each experiment by assignment label
    labels = {
        signal.assignment
        for experiment in experiments
        for signal in experiment
    }

    colours = {
        label: SAFE_COLOURS[it]
        for it, label in enumerate(labels)
    }

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
                marker='x',
                label=signal.assignment,
                color=colours[signal.assignment]
            )

    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel(
        r'$T$ $\mathregular{(K)}$',
        fontsize=labelfontsize
    )

    ax.set_ylabel(
        r'$\delta_\mathregular{^1H}T$ (ppm K)',
        fontsize=labelfontsize
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(
                f'\n Shift vs T plots saved to \n {save_name}\n',
                'cyan'
            )
    if show:
        plt.show()

    return


def plot_isoaxrho(molecules: list[main.Molecule], save: bool = True,
                  show: bool = True, save_name: str = 'iso_ax_rho_tdep.png',
    window_title: str = 'Isotropic, Axial, and Rhombic susceptibilities', # noqa
    verbose: bool = True,
    y_mode: str = 'chiT',
    susc_units: str = 'A3',
    out_file: str = 'chi_fits.txt',
    susc_models: list[models.SusceptibilityModel] = []) -> tuple[plt.Figure, tuple[plt.Axes]]: # noqa
    """
    Plot isotropic, axial and rhombic susceptibility components vs inverse temperature
    using the chi_plot weighted linear regression logic.
    """

    weighted_linreg_predict = chi_helpers.weighted_linreg_predict
    _plot_component = chi_helpers._plot_component
    _finalize_axes = chi_helpers._finalize_axes

    def _unit_conv(units: str) -> tuple[float, str]:
        if units == 'A3':
            return 1.0, r'$\mathregular{\AA^3}$'
        if units == 'A3 mol-1':
            return constants.Avogadro, r'$\mathregular{\AA^3 \ mol^{-1}}$'
        if units == 'cm3':
            return 1E-24, r'$\mathregular{cm^3}$'
        if units == 'cm3 mol-1':
            return 1E-24 * constants.Avogadro / (4 * np.pi), r'$\mathregular{cm^3 \ mol^{-1}}$' # noqa
        raise ValueError(f'Unsupported susceptibility unit: {units}')

    if not len(molecules):
        return None, None

    temperatures = np.array([m.susc.temperature for m in molecules], dtype=float)
    isotropic = np.array([m.susc.iso for m in molecules], dtype=float)
    axial = np.array([m.susc.axiality for m in molecules], dtype=float)
    rhombic = np.array([m.susc.rhombicity for m in molecules], dtype=float)

    iso_err = axial_err = rhombic_err = None

    mask = np.isfinite(temperatures) & np.isfinite(isotropic) & np.isfinite(axial) & np.isfinite(rhombic) # noqa
    if not np.any(mask):
        return None, None

    temperatures = temperatures[mask]
    isotropic = isotropic[mask]
    axial = axial[mask]
    rhombic = rhombic[mask]

    inv_t = np.divide(1.0, temperatures, out=np.full_like(temperatures, np.nan), where=temperatures != 0) # noqa

    finite_mask = np.isfinite(inv_t) & np.isfinite(isotropic) & np.isfinite(axial) & np.isfinite(rhombic) # noqa
    temperatures = temperatures[finite_mask]
    inv_t = inv_t[finite_mask]
    isotropic = isotropic[finite_mask]
    axial = axial[finite_mask]
    rhombic = rhombic[finite_mask]

    conv, unit_label = _unit_conv(susc_units)
    isotropic *= conv
    axial *= conv
    rhombic *= conv
    if len(susc_models) and isinstance(susc_models[0], models.IsoAxRhoFitter):
        _fix = susc_models[0].fix_vars
        if 'iso' not in _fix:
            iso_err = [model.fit_stdev['iso'] * conv for model in susc_models]
        if 'ax' not in _fix:
            axial_err = [model.fit_stdev['ax'] * conv for model in susc_models]
        if 'rho_over_ax' not in _fix:
            rhombic_err = [model.fit_stdev['rho_over_ax'] * conv for model in susc_models]

    if iso_err is not None:
        iso_err = [err for err, keep in zip(iso_err, mask) if keep]
    if axial_err is not None:
        axial_err = [err for err, keep in zip(axial_err, mask) if keep]
    if rhombic_err is not None:
        rhombic_err = [err for err, keep in zip(rhombic_err, mask) if keep]

    if iso_err is not None:
        iso_err = [err for err, keep in zip(iso_err, finite_mask) if keep]
    if axial_err is not None:
        axial_err = [err for err, keep in zip(axial_err, finite_mask) if keep]
    if rhombic_err is not None:
        rhombic_err = [err for err, keep in zip(rhombic_err, finite_mask) if keep]

    ylabel_base = rf'$\chi T$ / {unit_label} K' if y_mode.lower() == 'chit' else rf'$\chi$ / {unit_label}' # noqa

    if y_mode.lower() == 'chit':
        isotropic *= temperatures
        axial *= temperatures
        rhombic *= temperatures
        if iso_err is not None:
            iso_err = list(np.array(iso_err, dtype=float) * temperatures)
        if axial_err is not None:
            axial_err = list(np.array(axial_err, dtype=float) * temperatures)
        if rhombic_err is not None:
            rhombic_err = list(np.array(rhombic_err, dtype=float) * temperatures)

    def _fit(comp: np.ndarray, errs: list[float] | None):
        sig = errs if errs is not None and len(errs) == len(comp) else None
        return weighted_linreg_predict(inv_t.tolist(), comp.tolist(), inv_t.tolist(), sigma=sig)

    iso_pred, a_iso, b_iso, iso_std, a_iso_se, b_iso_se = _fit(isotropic, iso_err)
    ax_pred, a_ax, b_ax, ax_std, a_ax_se, b_ax_se = _fit(axial, axial_err)
    rho_pred, a_rho, b_rho, rho_std, a_rho_se, b_rho_se = _fit(rhombic, rhombic_err)

    base, ext = os.path.splitext(save_name)
    ext = ext if ext else '.png'
    iso_out = f'{base}_iso{ext}'
    ax_out = f'{base}_ax{ext}'
    rho_out = f'{base}_rho{ext}'

    if save:
        fig_iso, _ = _plot_component(
            'blue',
            inv_t.tolist(),
            None,
            None,
            None,
            inv_t.tolist(),
            isotropic.tolist(),
            iso_err,
            iso_pred,
            iso_std,
            a_iso,
            b_iso,
            a_iso_se,
            b_iso_se,
            'iso',
            iso_out,
            ylabel_base=ylabel_base
        )
        plt.close(fig_iso)
        fig_ax, _ = _plot_component(
            'green',
            inv_t.tolist(),
            None,
            None,
            None,
            inv_t.tolist(),
            axial.tolist(),
            axial_err,
            ax_pred,
            ax_std,
            a_ax,
            b_ax,
            a_ax_se,
            b_ax_se,
            'ax',
            ax_out,
            ylabel_base=ylabel_base
        )
        plt.close(fig_ax)
        fig_rho, _ = _plot_component(
            'red',
            inv_t.tolist(),
            None,
            None,
            None,
            inv_t.tolist(),
            rhombic.tolist(),
            rhombic_err,
            rho_pred,
            rho_std,
            a_rho,
            b_rho,
            a_rho_se,
            b_rho_se,
            'rho',
            rho_out,
            ylabel_base=ylabel_base
        )
        plt.close(fig_rho)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True, num=window_title)
    ax.plot(inv_t, isotropic, label=r'$\chi_{iso}$', color='blue', marker='o', linestyle='')
    ax.plot(inv_t, axial, label=r'$\chi_{ax}$', color='green', marker='o', linestyle='')
    ax.plot(inv_t, rhombic, label=r'$\chi_{rho}$', color='red', marker='o', linestyle='')

    if iso_pred is not None:
        ax.plot(inv_t, iso_pred, label=r'$\chi_{iso}$ LR', linestyle='-.', linewidth=1.5, color='blue') # noqa
    if ax_pred is not None:
        ax.plot(inv_t, ax_pred, label=r'$\chi_{ax}$ LR', linestyle='-.', linewidth=1.5, color='green') # noqa
    if rho_pred is not None:
        ax.plot(inv_t, rho_pred, label=r'$\chi_{rho}$ LR', linestyle='-.', linewidth=1.5, color='red') # noqa

    if iso_err is not None:
        ax.errorbar(inv_t, isotropic, yerr=iso_err, fmt='none', ecolor='blue', alpha=0.5, capsize=2) # noqa
    if axial_err is not None:
        ax.errorbar(inv_t, axial, yerr=axial_err, fmt='none', ecolor='green', alpha=0.5, capsize=2) # noqa
    if rhombic_err is not None:
        ax.errorbar(inv_t, rhombic, yerr=rhombic_err, fmt='none', ecolor='red', alpha=0.5, capsize=2) # noqa

    legend = _finalize_axes(ax, inv_t.tolist(), inv_t.tolist(), 'All Components', base_ylabel=ylabel_base) # noqa
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent()
    ax_bbox = ax.get_window_extent()
    if legend_bbox.y0 < ax_bbox.y1:
        y_min, y_max = ax.get_ylim()
        y_padding = (y_max - y_min) * 0.10
        ax.set_ylim(y_min, y_max + y_padding)

    if save:
        plt.savefig(save_name, dpi=600)
        if verbose:
            ut.cprint(f'\n {y_mode} vs 1/T plots saved to \n {save_name}\n', 'cyan')
    if show:
        plt.show()

    if y_mode.lower() == 'chit':
        with open(out_file, 'w') as f:
            _fmt = lambda v: f'{v:.6f}' if v is not None else 'nan'
            f.write('isotropic\n')
            f.write(f'a = {_fmt(a_iso)}      b = {_fmt(b_iso)}\n')
            f.write('axial\n')
            f.write(f'a = {_fmt(a_ax)}      b = {_fmt(b_ax)}\n')
            f.write('rhombic\n')
            f.write(f'a = {_fmt(a_rho)}      b = {_fmt(b_rho)}\n')
    return fig, ax


def plot_hyperfine_iso_vs_ax(value_dict: dict[str, float], order: list[int],
                             fig: plt.Figure = None, ax: plt.Axes = None,
                             symbol='x',
                             save: bool = False, show: bool = True,
                             save_name: str = 'hyperfines.dat',
                             verbose: bool = False,
                             window_title: str = 'hyperfine data'):

    if all([fig is None, ax is None]):
        fig, ax = plt.subplots(num=window_title)

    vals = list(value_dict.values())

    ax.plot(
        [vals[o] for o in order],
        lw=0,
        marker=symbol,
        fillstyle='none',
        color='C1'
    )

    ax.xaxis.set_major_locator(
        ticker.FixedLocator(np.arange(len(value_dict)))
    )
    labels = [label for label in value_dict.keys()]
    ax.set_xticklabels([labels[o] for o in order])

    ax.set_ylabel(r'$A_\mathregular{iso} / (A_\mathregular{dip_{xx}} + A_\mathregular{dip_{yy}})$') # noqa

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(f'Hyperfine plot saved to {save_name}', 'cyan')

    if show:
        plt.show()

    return


def plot_hyperfine_spread(nuclei: list[main.Nucleus],
                          components=list[str],
                          save: bool = False,
                          show: bool = True, save_name: str = 'hyperfines.png',
                          window_title: str = 'Hyperfine Components',
                          verbose: bool = True) -> tuple[plt.Figure, list[plt.Axes]]: # noqa
    '''
    Plots spread of hyperfine coupling components for each chem label

    Parameters
    ----------
    nuclei: list[Nucleus]
        Nuclei to plot
    components: list[str]
        Name(s) of hyperfine components to plot\n
        (xx, yy, iso, ax, rho, dxy, ...)
    save: bool, default False
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'hyperfines.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Hyperfine components'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    plt.Axes
        Matplotlib axis object
    '''

    legend_labels = {
        component: ''
        for component in components
    }

    a_comps = {
        component: {
            nuc.chem_math_label: []
            for nuc in nuclei
        }
        for component in components
    }

    for component in components:
        if component == 'iso':
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(nuc.A.iso)
                legend_labels[component] = r'$A_\mathregular{iso}$'
        elif component == 'ax':
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(nuc.A.dip[0, 0] - nuc.A.dip[1, 1]) # noqa
                legend_labels[component] = r'$A_\mathregular{dip, ax}$'
        elif component == 'rho':
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(nuc.A.dip[0, 0] + nuc.A.dip[1, 1]) # noqa
                legend_labels[component] = r'$A_\mathregular{dip, rho}$'
        elif 'd' in component:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(nuc.A.dip[ut.comp2ind(component[1:])]) # noqa
                legend_labels[component] = fr'$A_{{\mathregular{{dip, }}\mathregular{{{component[1:]}}}}}$'  # noqa
        else:
            for nuc in nuclei:
                a_comps[component][nuc.chem_math_label].append(nuc.A.tensor[ut.comp2ind(component)]) # noqa
                legend_labels[component] = fr'$A_\mathregular{{{component}}}$'

    unique_chemlabels = []
    for nuc in nuclei:
        if nuc.chem_math_label not in unique_chemlabels:
            unique_chemlabels.append(nuc.chem_math_label)

    fig, ax = plt.subplots(
        1,
        1,
        num=window_title
    )

    xvals = np.arange(1, len(unique_chemlabels) + 1)

    legend_markers = []
    for comp_values in a_comps.values():
        _violin = ax.violinplot(
            dataset=list(comp_values.values()),
            positions=xvals + 0.5,
            vert=True,
            showmeans=True
        )
        legend_markers.append(
            mpatches.Patch(color=_violin['bodies'][0].get_facecolor().flatten()), # noqa
        )

    if len(a_comps) < 11:
        step = 1
    else:
        step = 2

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks(xvals[::step] + 0.5)
    labels = list(unique_chemlabels)
    ax.set_xticklabels(labels[::step])
    ax.grid(axis='x', ls='--', which='minor')
    ax.set_xlim(0.5, len(labels) + 1.5)
    ax.xaxis.set_tick_params('major', length=0)

    ax.hlines(0, 0.5, len(labels) + 1.5, lw=0.5, color='k')

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend(
        legend_markers,
        [legend_labels[comp] for comp in a_comps.keys()]
    )

    ax.set_ylabel(r'Hyperfine Coupling (ppm Å$^\mathregular{-3}$)')
    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(
                'Hyperfine spread plot saved to {}'.format(save_name),
                'blue'
            )

    if show:
        plt.show()

    return fig, ax


def plot_raw_deconv_pred(molecule: main.Molecule, experiment: main.Experiment,
                         save: bool = True, show: bool = True,
                         save_name: str = 'iso_ax_rho_tdep.png',
                         window_title: str = 'Raw, Deconvoluted, and Predicted Spectra', # noqa
                         verbose: bool = True) -> tuple[plt.Figure, tuple[plt.Axes]]: # noqa
    '''
    Plots raw spectrum, deconvoluted spectrum, and predicted spectra

    Parameters
    ----------
    molecule: list[main.Molecule]
        Molecule containing non-empty Susceptibility attributes
    save: bool, default True
        If True, saves plot to file
    show: bool, default True
        If True, shows plot on screen
    save_name: str, default 'iso_ax_rho_tdep.png'
        If save is True, will save plot to this file name
    window_title: str, default 'Raw, Deconvoluted, and Predicted Spectra'
        Title of figure window, not of plot
    verbose: bool, default True
        If True, plot file location is written to terminal
    '''

    fig, ax = plt.subplots(
        3,
        1,
        figsize=(8, 5.5),
        num=window_title,
        sharex=True
    )

    # Experimental spectrum
    ax[0].plot(
        experiment.spectrum[::4, 0],
        experiment.spectrum[::4, 1],
        lw=1,
        color='k'
    )
    ax[0].set_title('Full spectrum', loc='left', fontdict={'size': 'smaller'})

    # Deconvoluted spectrum (user fit)
    # Convert linewidths into ppm for this spectrometer
    ppm_grid = np.linspace(
        np.min(experiment.spectrum[::4, 0]),
        np.max(experiment.spectrum[::4, 0]),
        10000
    )
    _total = np.zeros(np.shape(ppm_grid))
    for signal in experiment.signals:
        _width = signal.width * 1E-6 * experiment.larmor
        _total += signal.l_to_g * lorentzian(
            ppm_grid,
            _width,
            signal.shift,
            signal.area
        )
        _total += (1 - signal.l_to_g) * gaussian(
            ppm_grid,
            _width,
            signal.shift,
            signal.area
        )
    ax[1].plot(ppm_grid, _total, lw=1, color='k')
    ax[1].set_title(
        'Paramagnetic Signals',
        loc='left',
        fontdict={'size': 'smaller'}
    )

    # Predicted spectrum
    _total = np.zeros(np.shape(ppm_grid))
    for nucleus in molecule.nuclei:
        if nucleus.isotope == experiment.isotope:
            _width = 100 * 1E-6 * experiment.larmor
            _total += lorentzian(
                ppm_grid,
                _width,
                nucleus.shift.avg,
                1
            )

    shifts = list(
        nucleus.shift.avg
        for nucleus in molecule.nuclei
        if nucleus.isotope == experiment.isotope
    )
    labels = list(
        nucleus.chem_math_label
        for nucleus in molecule.nuclei
        if nucleus.isotope == experiment.isotope
    )

    shifts, shift_inds = np.unique(shifts, return_index=True)
    labels = [labels[ind] for ind in shift_inds]

    closest_y = [
        _total[ut.find_index_of_nearest(ppm_grid, sh)]
        for sh in shifts
    ]

    ax[2].plot(
        shifts,
        closest_y,
        lw=0,
        marker='x',
        color='k'
    )

    for shift, y, label in zip(shifts, closest_y, labels):
        if np.min(ppm_grid) < shift < np.max(ppm_grid):
            ax[2].text(shift, y, label)

    ax[2].plot(
        ppm_grid,
        _total,
        lw=1,
        color='k'
    )
    ax[2].set_title('Simulation', loc='left', fontdict={'size': 'smaller'})

    # Axis configuration
    for axis in ax:
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.set_xlim(
            [
                np.max(experiment.spectrum[:, 0]),
                np.min(experiment.spectrum[:, 0])
            ]
        )
        axis.spines[['right', 'top', 'left']].set_visible(False)

    ax[2].set_xlabel(r'{} $\delta$ (ppm)'.format(
        ut.isotope_format(experiment.isotope))
    )
    ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=500)
        if verbose:
            ut.cprint(f'\n Spectra saved to\n {save_name}\n', 'cyan')

    if show:
        plt.show()

    return
