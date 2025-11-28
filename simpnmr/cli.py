'''
This is the command line interface to SimpNMR
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import xyz_py as xyzp
import copy
import sys
from pathos import multiprocessing as mp
import re

from . import main
from . import readers as rdrs
from . import utils as ut
from . import inputs as inps
from . import models
from . import visualise as vis
from . import outputs
from collections import defaultdict
from . import transform as tfm

# Change figure save dialog to use current working directory
mpl.rcParams['savefig.directory'] = ''

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Change matplotlib font size to be larger
mpl.rcParams.update({'font.size': 14})

# Set spawn as default start method - MUCH faster on WSL2 than default fork
# if not using pathos multiprocessing instead
# mp.set_start_method('spawn', force=True)

# Print r2 to terminal for each assignment
ECHO_R2 = False
if os.getenv('pnmr_echo_r2'):
    try:
        if os.getenv('pnmr_echo_r2').lower() == 'true':
            ECHO_R2 = True
    except ValueError:
        ut.cprint('Error in pnmr_echo_r2 environment variable', 'red')
        sys.exit(1)

# Set user specified font name
if os.getenv('pnmr_fontname'):
    try:
        plt.rcParams['font.family'] = os.getenv('pnmr_fontname')
    except ValueError:
        ut.cprint('Error in pnmr_fontname environment variable', 'red')
        sys.exit(1)

# Set user specified plot file format
PFF = '.png'
if os.getenv('pnmr_plot_format'):
    try:
        PFF = os.getenv('pnmr_plot_format')
        if PFF[0] != '.':
            PFF = f'.{PFF}'
    except ValueError:
        ut.cprint('Error in pnmr_plot_format environment variable', 'red')
        sys.exit(1)

# Set user specified plot file format
CSV_DELIMITER = ','
if os.getenv('pnmr_csvdelimiter'):
    try:
        CSV_DELIMITER = os.getenv('pnmr_csvdelimiter')
    except ValueError:
        ut.cprint('Error in pnmr_csvdelimiter environment variable', 'red')
        sys.exit(1)

_SHOW_CONV = {
    'on': True,
    'save': False,
    'show': True,
    'off': False
}

_SAVE_CONV = {
    'on': True,
    'save': True,
    'show': False,
    'off': False
}

_PLOT_ACTIVE = ['on', 'show', 'save']


def extract_dia_func(uargs):
    '''
    Wrapper for extract_dia cli call
    '''
    ut.cprint(
        f'Extracting shifts from {uargs.output_file}',
        'cyan'
    )

    data = rdrs.QCCS.guess_from_file(uargs.output_file)

    if len(uargs.ref_output_file):
        ref_data = rdrs.QCCS.guess_from_file(uargs.ref_output_file)
        ut.cprint(
            f'Extracting reference shifts from {uargs.ref_output_file}',
            'cyan'
        )

        ref_labels = list(ref_data.cs_iso.keys())
        ref_labels_nn = xyzp.remove_label_indices(ref_labels)

        avg_ref_iso = dict.fromkeys(ref_labels_nn, 0)

        for lab, lab_nn in zip(ref_labels, ref_labels_nn):
            avg_ref_iso[lab_nn] += ref_data.cs_iso[lab]

        for lab_nn in np.unique(ref_labels_nn):
            avg_ref_iso[lab_nn] /= ref_labels_nn.count(lab_nn)

        # Subtract from reference shifts based on atom type
        labels = list(data.cs_iso.keys())
        labels_nn = xyzp.remove_label_indices(labels)

        for lab, lab_nn in zip(labels, labels_nn):
            if lab_nn in avg_ref_iso.keys():
                data.cs_iso[lab] = avg_ref_iso[lab_nn] - data.cs_iso[lab]

    # Save diamagnetic shifts to file
    iso_shifts = list(data.cs_iso.values())
    labels = list(data.cs_iso.keys())
    labels = xyzp.remove_label_indices(labels)
    labels = xyzp.add_label_indices(labels)

    out = [
        f'{label}, {value:.6f}'
        for label, value in zip(labels, iso_shifts)
    ]

    np.savetxt(
        'extracted_dia.csv',
        out,
        delimiter=',',
        header='atom_label, shift',
        fmt='%s',
        comments=''
    )

    ut.cprint('Extracted shifts saved to extracted_dia.csv', 'cyan')

    return


def fit_susc_func(uargs):
    '''
    Wrapper for command line call to fit_susc
    '''

    # Parse input file
    config = inps.FitSuscConfig.from_file(uargs.input_file)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Either load hyperfines from DFT output file
    if config.hyperfine_method == 'dft':
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        # Write raw hyperfine data to output file
        qc_hyperfine_data.save_to_csv(
            os.path.join(config.project_name, 'dft_hyperfines.csv'),
            verbose=True,
            delimiter=CSV_DELIMITER,
            comment=f'# Data taken from file {config.hyperfine_file}'
        )

        # Create molecule object from quantum chemical hyperfine data
        # Retain only the atoms that are given in the labels file
        base_molecule = main.Molecule.from_QCA(
            qc_hyperfine_data,
            converter='MHz_to_Ang-3',
            elements=config.nuclei_include
        )
        ut.cprint(f" Group(s)/Atoms included: {config.nuclei_include}", "cyan")
    # generate using point dipole approximation
    elif config.hyperfine_method == 'pdip':

        if os.path.splitext(config.hyperfine_file)[1] == '.xyz':
            labels, coords = xyzp.load_xyz(config.hyperfine_file)
        elif os.path.splitext(config.hyperfine_file)[1] in ['.log', '.out']:
            QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
            labels = QCS.labels
            coords = QCS.coords
        else:
            ut.cprint(f'Specified hyperfine file format {os.path.splitext(uargs.structure_file)[1]} unsupported', 'red') # noqa
            sys.exit(1)

        # Create molecule
        base_molecule = main.Molecule.from_labels_coords(
            labels,
            coords,
            elements=config.nuclei_include
        )

        # Calculate point dipole hyperfine
        base_molecule.calc_pdip(config.hyperfine_pdip_centres)

    # or load from CSV
    elif config.hyperfine_method == 'csv':
        base_molecule = main.Molecule.from_csv(
            config.hyperfine_file,
            elements=config.nuclei_include
        )

    # Add chemical labels
    if len(config.chem_labels_file):
        try:
            base_molecule.add_chem_labels_from_file(
                config.chem_labels_file
            )
        except ValueError as err:
            ut.red_exit(str(err) + '\n Check chem_labels and hyperfine files.')
        except KeyError as err:
            ut.red_exit(str(err))

        # Save xyz file with chemical labels for chemcraft
        base_molecule.save_chemcraft_xyz(
            file_name=os.path.join(
                config.project_name,
                'chemcraft_structure.xyz'
            )
        )

    # Save xyz file with chemical labels for chemcraft
    base_molecule.save_xyz(
        file_name=os.path.join(
            config.project_name,
            'structure.xyz'
        ),
        comment=f'Structure from {config.hyperfine_file}'
    )

    # Apply rotation matrix to all hyperfine tensors
    # if requested
    if len(config.hyperfine_rotate):
        _rot_a = np.loadtxt(config.hyperfine_rotate)
        base_molecule.rotate_hyperfines(_rot_a)

    # Load diamagnetic shift file
    if len(config.diamagnetic_file):
        base_molecule.load_diamagnetic_shifts(
            config.diamagnetic_file,
            config.diamagnetic_method,
            config.diamagnetic_ref_file,
            config.diamagnetic_ref_method
        )

    # Rotationally average hyperfines
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Create experiments
    experiments = main.Experiment.from_file(
        config.experiment_files
    )

    # Check the number of experiments is consistent across the files
    # and issue warning if not
    if len(np.unique([len(exp.signals) for exp in experiments])) > 1:
        ut.cprint(
            'Warning: some experiments have more signals than others!',
            'black_yellowbg'
        )

    # Create a molecule object to accompany each experiment object
    molecules = [
        copy.deepcopy(base_molecule)
        for _ in range(len(experiments))
    ]

    # Obtain fitted and fixed variables
    fit_vars = {
        key: value[1]
        for key, value in config.susc_fit_variables.items()
        if value[0] == 'fit'
    }

    fix_vars = {
        key: value[1]
        for key, value in config.susc_fit_variables.items()
        if value[0] == 'fix'
    }

    name_to_susc_fit: dict[str, models.SusceptibilityModel] = {
        'full': models.FullSuscFitter,
        'split': models.SplitFitter,
        'isoaxrho': models.IsoAxRhoFitter,
        'eigen': models.EigenFitter,
        'isoeigen': models.IsoEigenFitter
    }

    model_to_use = name_to_susc_fit[config.susc_fit_type]

    # Create one susceptibility model per molecule/experiment pair
    susc_models: list[models.SusceptibilityModel] = [
        copy.deepcopy(model_to_use(fit_vars, fix_vars))
        for _ in molecules
    ]

    if uargs.dry_run:
        ut.cprint('All good chief!', 'green')
        sys.exit()

    if len(config.susc_fit_average_shifts):
        if 'all' in config.susc_fit_average_shifts:
            config.susc_fit_average_shifts = list(
                {nuc.chem_label for nuc in base_molecule.nuclei}
            )
        average_labels = [
            [
                nuc.label
                for nuc in base_molecule.nuclei
                if nuc.chem_label == _cl
            ]
            for _cl in config.susc_fit_average_shifts
        ]
    else:
        average_labels = []

    # Shift terms for plots
    # does not affect fit!
    _terms = ['pc', 'fc', 'd']
    if config.hyperfine_method == 'pdip':
        _terms.pop(_terms.index('fc'))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index('d'))

    # Run fit for all experiments
    for molecule, susc_model, experiment in zip(molecules, susc_models, experiments): # noqa

        # If permuting assignments, then first
        # run all assignment permutations to find best one
        if config.assignment_method == 'permute':

            # If no permutation groups provided, permute all
            if not len(config.assignment_groups):
                config.assignment_groups = [
                    list({
                        nuc.chem_label for nuc in molecule.nuclei
                    })
                ]
            # For the current experiment, generate a new set in which
            # the assignment is permuted according to user defined groups
            permed_assignments = main.Experiment.generate_permutations(
                experiment=experiment,
                groups=config.assignment_groups
            )

            ut.cprint(
                f'\n There are {len(permed_assignments):d} possible permutations', # noqa
                'cyan'
            )

            # For each permutation, fit tensor and store r2_adjusted

            # Number of threads
            if config.num_threads == 'auto':
                num_threads = mp.cpu_count() - 1
            else:
                num_threads = config.num_threads

            if num_threads > len(permed_assignments):
                num_threads = len(permed_assignments)

            # Create parallel pool
            pool = mp.Pool(num_threads)
            ut.cprint(f'   ... using {num_threads:d} threads\n', 'cyan')

            iterables = [
                (
                    molecule,
                    permed_assgn,
                    susc_model,
                    copy.deepcopy(experiment),
                    average_labels
                )
                for permed_assgn in permed_assignments
            ]

            # Calculate each assignment's r2 in parallel
            results = pool.starmap(obtain_r2a, iterables)

            # Close Pool and let all the processes complete
            pool.close()
            pool.join()

            # Find assignment with largest r2
            # and use in subsequent (re)fitting
            assignment = permed_assignments[
                np.nanargmax(
                    results
                )
            ]
            opt_r2 = np.nanmax(
                results
            )

            # and swap in new, permuted, assignments
            for it, new in enumerate(assignment):
                experiment.signals[it].assignment = new

            # Save assigned experiment to file
            experiment.to_csv(
                os.path.join(
                    config.project_name,
                    'assigned_experiment_{:.2f}_K.csv'.format(
                        experiment.temperature
                    )
                ),
                delimiter=CSV_DELIMITER,
                comment=f'# Optimal Assignment\n# r2 = {opt_r2:f}\n# T = {experiment.temperature:.2f} K' # noqa
            )

        # Fit susceptibility model to experimental chemical shifts
        # update guess using previous fit
        # susc_model.fit_vars = guess
        susc_model.fit_to(
            molecule, experiment, average_labels=average_labels
        )

        # Skip if fit fails
        if not susc_model.fit_status:
            continue
        # else use best fit as starting guess
        # else:
        #     guess = susc_model.final_var_values

        # Update susceptibility tensor of Molecule using model
        molecule.susc = susc_model.tosusceptibility()
        # print('Taking absolute of DX_ax and DX_rho')
        # molecule.susc.axiality = np.abs(molecule.susc.axiality)
        # molecule.susc.rhombicity = np.abs(molecule.susc.rhombicity)

        # Calculate shifts using new susceptibility tensor
        molecule.calculate_shifts()
        molecule.average_shifts()

        visible = ['show', 'on']

        if uargs.shift_plots in _PLOT_ACTIVE:
            if any(cfg in visible for cfg in [uargs.contrib_plots, uargs.spread_plots]): # noqa
                show = False
            else:
                show = _SHOW_CONV[uargs.shift_plots]
            vis.plot_fitted_shifts(
                molecule,
                experiment,
                susc_model,
                show=show,
                susc_units=uargs.susc_units,
                average=len(config.susc_fit_average_shifts),
                save=_SAVE_CONV[uargs.shift_plots],
                save_name=os.path.join(
                    config.project_name,
                    f'shifts_{experiment.temperature:.2f}_K{PFF}'
                ),
                verbose=True,
                window_title=f'Fitted shifts at {experiment.temperature:.2f} K' # noqa
            )

            visible = ['show', 'on']

            if all(cfg not in visible for cfg in [uargs.contrib_plots, uargs.spread_plots]): # noqa
                plt.close('all')

            if uargs.spread_plots in _PLOT_ACTIVE:

                vis.plot_shift_spread(
                    molecule,
                    experiment,
                    terms=_terms,
                    show=_SHOW_CONV[uargs.spread_plots],
                    save=_SAVE_CONV[uargs.spread_plots],
                    save_name=os.path.join(config.project_name, f'shift_spread_{molecule.susc.temperature:.2f}_K{PFF}'), # noqa
                    verbose=True,
                    window_title=f'Spread of predicted shift components at {experiment.temperature:.2f} K', # noqa
                    order='descending'
                )

            if uargs.contrib_plots in _PLOT_ACTIVE:

                vis.plot_shift_contrib(
                    molecule,
                    experiment,
                    terms=_terms,
                    show=_SHOW_CONV[uargs.contrib_plots],
                    save=_SAVE_CONV[uargs.contrib_plots],
                    save_name=os.path.join(
                        config.project_name,
                        f'mean_components_{experiment.temperature:.2f}_K{PFF}'
                    ),
                    verbose=True,
                    window_title=f'Predicted shift components at {experiment.temperature:.2f} K', # noqa
                    order='descending'
                )

                plt.close('all')

    # Write shift data to file
    _comment = f'# Hyperfines from file {config.hyperfine_file}\n'
    if len(config.diamagnetic_file):
        _comment += f'# Diamagnetic shifts from file {config.diamagnetic_file}\n' # noqa
    if len(config.diamagnetic_ref_file):
        _comment += f'# Diamagnetic reference from file {config.diamagnetic_ref_file}\n' # noqa
    _comment += f'# T = {molecule.susc.temperature:.2f} K'

    for molecule in molecules:
        molecule.to_csv(
            os.path.join(
                config.project_name,
                f'hyperfines_and_shifts_{molecule.susc.temperature:.2f}_K.csv'
            ),
            delimiter=CSV_DELIMITER,
            comment=_comment
        )

    # Write susceptibility tensor with model terms
    outputs.save_susc(
        molecules, os.path.join(
            config.project_name,
            'susceptibility_tensor.csv'
        ),
        susc_models=susc_models,
        susc_units=uargs.susc_units
    )

    if uargs.pcs_isosurface:
        for molecule in molecules:
            # Generate and save PCS isosurface
            molecule.susc.save_pcs_isosurface(
                molecule.labels,
                molecule.coords,
                molecule.labels[0],
                comment='PCS Isosurface',
                file_name=os.path.join(
                    config.project_name,
                    f'pcs_isosurf_{molecule.susc.temperature:.2f}_K.cube'
                )
            )

    if len(experiments) > 1:
        vis.plot_isoaxrho(
            molecules,
            show=False,
            save=_SAVE_CONV[uargs.isoaxrho_plots],
            save_name=os.path.join(
                config.project_name,
                f'susceptibility_components_chiT{PFF}'
            ),
            verbose=True,
            y_mode='chiT',
            window_title='ChiT Susceptibility components',
            susc_units=uargs.susc_units,
            susc_models=susc_models if model_to_use == models.IsoAxRhoFitter else [], # noqa
            out_file=os.path.join(
                config.project_name,
                'isoaxrho_fit.txt'
            )
        )

        vis.plot_isoaxrho(
            molecules,
            show=_SHOW_CONV[uargs.isoaxrho_plots],
            save=_SAVE_CONV[uargs.isoaxrho_plots],
            save_name=os.path.join(
                config.project_name,
                f'susceptibility_components_chi{PFF}'
            ),
            verbose=True,
            y_mode='chi',
            window_title='Susceptibility components',
            susc_units=uargs.susc_units,
            susc_models=susc_models if model_to_use == models.IsoAxRhoFitter else [] # noqa
        )

    return


def obtain_r2a(molecule: main.Molecule, assignment: list[str],
               model: models.SusceptibilityModel, experiment: main.Experiment,
               average_labels: list[list[str]]):
    '''
    Wrapper function for parallel pool.
    Fits a susceptibility model to an experiment for a given assignment
    then stores the adjusted r2 of the fit in the model object
    '''

    # and swap in new, permuted, assignments
    for it, new in enumerate(assignment):
        experiment.signals[it].assignment = new

    # Fit susceptibility model to experimental chemical shifts
    model.fit_to(molecule, experiment, average_labels=average_labels)

    # Print to screen if envvar enabled
    if ECHO_R2:
        print(model.adj_r2)

    return model.adj_r2


def plot_a_func(uargs):
    '''
    Wrapper for call to cli plot_a subprogram
    '''
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(uargs.calculation_data)

    # Create molecule object from quantum chemical hyperfine data
    # to convert units
    molecule = main.Molecule.from_QCA(
        calc_data, converter='MHz_to_Ang-3', elements=uargs.elements
    )

    if uargs.chem_labels is not None:
        molecule.add_chem_labels_from_file(uargs.chem_labels)

    file_head = os.path.splitext(uargs.calculation_data)[0]

    if not (uargs.hide_plots and not uargs.save):
        if uargs.chem_labels is not None:
            vis.plot_hyperfine_spread(
                molecule.nuclei,
                components=uargs.components,
                save=uargs.save,
                show=False,
                save_name=f'hyperfine_spread_{file_head}{PFF}',
                window_title=f'Spread of hyperfine data from {uargs.calculation_data}', # noqa
                verbose=True
            )

        vis.plot_hyperfine(
            molecule.nuclei,
            components=uargs.components,
            save=uargs.save,
            show=False,
            save_name=f'hyperfine_{file_head}{PFF}',
            window_title=f'Hyperfine data from {uargs.calculation_data}',
            verbose=True
        )

        if not uargs.hide_plots:
            plt.show()

    return


def plot_a_iso_ax_func(uargs):
    '''
    Wrapper for call to cli plot_a subprogram
    '''

    config = inps.PlotAConfig.from_file(uargs.input_file)

    symbols = ['x', 'o']
    fig, ax = plt.subplots(1, 1)

    for hf_file, symb in zip(config.hyperfine_file[1:], symbols):

        # Either load hyperfines from DFT output file
        if config.hyperfine_method == 'dft':
            qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
            # Write raw calculation data to output file
            qc_hyperfine_data.save_to_csv(
                os.path.join(config.project_name, 'dft_hyperfines.csv'),
                verbose=True,
                delimiter=CSV_DELIMITER,
                comment=f'# Data taken from file {config.hyperfine_file}'
            )

            # Create molecule object from quantum chemical hyperfine data
            # Retain only the atoms that are given in the labels file
            base_molecule = main.Molecule.from_QCA(
                qc_hyperfine_data, converter='MHz_to_Ang-3',
                elements=config.nuclei_include
            )

        # generate using point dipole approximation
        elif config.hyperfine_method == 'pdip':

            if os.path.splitext(config.hyperfine_file)[1] == '.xyz':
                labels, coords = xyzp.load_xyz(config.hyperfine_file)
            elif os.path.splitext(config.hyperfine_file)[1] in ['.log', '.out']: # noqa
                QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
                labels = QCS.labels
                coords = QCS.coords
            else:
                ut.cprint(f'Specified hyperfine file format {os.path.splitext(uargs.structure_file)[1]} unsupported', 'red') # noqa
                sys.exit(1)

            # Create molecule
            base_molecule = main.Molecule.from_labels_coords(
                labels, coords, elements=config.nuclei_include
            )

            # Calculate point dipole hyperfine
            base_molecule.calc_pdip(config.hyperfine_pdip_centres)

        if len(config.hyperfine_average):
            for av in config.hyperfine_average:
                base_molecule.average_hyperfine(av)

        if len(config.chem_labels_file):
            base_molecule.add_chem_labels_from_file(
                config.chem_labels_file
            )

        file_head = os.path.splitext(hf_file)[0]

        iso_div_ax = {
            nuc.chem_math_label: nuc.A.iso / (nuc.A.dip[0, 0] + nuc.A.dip[1, 1]) # noqa
            for nuc in base_molecule.nuclei
        }

        if symb == 'x':
            order = np.argsort(list(iso_div_ax.values()))

        if not (uargs.hide_plots and not uargs.save):
            vis.plot_hyperfine_iso_vs_ax(
                iso_div_ax,
                order,
                fig=fig,
                ax=ax,
                symbol=symb,
                save=uargs.save,
                show=False,
                save_name=f'hyperfine_iso_ax_{file_head}{PFF}',
                verbose=True,
                window_title=f'Hyperfine data from {hf_file}'
            )

    xlims = ax.get_xlim()

    ax.hlines(0, *xlims, colors='k')

    ax.set_xlim(xlims)
    plt.show()
    return


def extract_a_func(uargs):
    '''
    Wrapper for call to cli extract_a subprogram
    '''
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(uargs.calculation_data)

    # Create molecule object from quantum chemical hyperfine data
    # to convert units
    base = main.Molecule.from_QCA(
        calc_data, converter='MHz_to_Ang-3'
    )

    base.to_csv(
        'hyperfine_{}.csv'.format(uargs.calculation_data),
        verbose=True,
        delimiter=CSV_DELIMITER
    )

    return


def calc_pdip_func(uargs):
    '''
    Wrapper for call to cli calc_pdir subprogram
    '''

    # Parse user specified centres
    centres = [
        centre.lower().capitalize()
        for centre in uargs.centres
    ]

    if os.path.splitext(uargs.structure_file)[1] == '.xyz':
        labels, coords = xyzp.load_xyz(uargs.structure_file)
    elif os.path.splitext(uargs.structure_file)[1] in ['.log', '.out']: # noqa
        QCS = rdrs.QCStructure.guess_from_file(uargs.structure_file)
        labels = QCS.labels
        coords = QCS.coords
    else:
        ut.cprint(f'Specified hyperfine file format {os.path.splitext(uargs.structure_file)[1]} unsupported', 'red') # noqa
        sys.exit(1)

    # Create molecule
    molecule = main.Molecule.from_labels_coords(
        labels, coords, elements=uargs.elements
    )

    # Calculate point dipole A_dip tensor
    molecule.calc_pdip(centres)

    if uargs.chem_labels is not None:
        molecule.add_chem_labels_from_file(uargs.chem_labels)

    # Save hyperfine data to file
    out = np.array(
        [
            '{}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
                nuc.label, nuc.chem_label, *nuc.A.dip[0, :], *nuc.A.dip[1, 1:], nuc.A.dip[2, 2] # noqa
            )
            for nuc in molecule.nuclei
        ]
    )

    # Save to file
    file_head = os.path.splitext(uargs.structure_file)[0]
    file_name = f'point_dipole_A_dip_{file_head}.csv'

    header = 'Label, Adip_xx (ppm Å^-3), Adip_xy (ppm Å^-3), Adip_xz (ppm Å^-3), Adip_yy (ppm Å^-3), Adip_yz (ppm Å^-3), Adip_zz (ppm Å^-3)' # noqa

    np.savetxt(
        file_name,
        out,
        delimiter=',',
        header=header,
        fmt='%s'
    )
    ut.cprint(f'Point dipole dipolar tensors saved to {file_name}', 'cyan')

    if len(uargs.plot_components):
        vis.plot_hyperfine(
            molecule.nuclei,
            uargs.plot_components,
            save=True,
            show=False,
            save_name=f'point_dipole_A_dip_{file_head}{PFF}',
            verbose=True,
            window_title='Point-Dipole Hyperfines'
        )

        if uargs.chem_labels is not None:
            vis.plot_hyperfine_spread(
                molecule.nuclei,
                uargs.plot_components,
                save=True,
                show=False,
                save_name=f'spread_point_dipole_A_dip_{file_head}{PFF}',
                verbose=True,
                window_title='Point-Dipole Hyperfines Spread'
            )

        plt.show()

    return


def calc_pcs_iso_func(uargs):
    '''
    Wrapper for call to cli calc_pcs_iso subprogram
    '''

    if os.path.splitext(uargs.structure_file)[1] == '.xyz':
        labels, coords = xyzp.load_xyz(uargs.structure_file)
    elif os.path.splitext(uargs.structure_file)[1] in ['.log', '.out']:
        QCS = rdrs.QCStructure.guess_from_file(uargs.structure_file)
        labels = QCS.labels
        coords = QCS.coords
    else:
        ut.cprint(f'Specified structure file format {os.path.splitext(uargs.structure_file)[1]} unsupported', 'red') # noqa
        sys.exit(1)

    if uargs.central_atom not in labels:
        ut.red_exit('Specified central atom not present in structure file\n Perhaps try with indexing e.g. Ni1') # noqa

    # Load susceptibility information
    if 'orca' in uargs.susc_format:
        suscs = main.Susceptibility.from_orca(
            uargs.susc_file,
            section=uargs.susc_format.split('orca_')[1]
        )
    elif 'csv' in uargs.susc_format:
        suscs = main.Susceptibility.from_csv(
            uargs.susc_file
        )
    elif 'molcas' in uargs.susc_format:
        ut.red_exit('Molcas files are not currently supported')

    for susc in suscs:
        if susc.temperature in uargs.temperatures:
            # Calculate irreducible representations of susceptibility tensor
            susc.calc_irred()

            # Generate and save PCS isosurface
            susc.save_pcs_isosurface(
                labels,
                coords,
                uargs.central_atom,
                comment=f'PCS Isosurface from {uargs.susc_file} at {susc.temperature:.2f} K', # noqa
                file_name=f'pcs_isosurface_{susc.temperature:.2f}_K.cube'
            )

    return


def predict_func(uargs):
    '''
    Wrapper for call to cli predict subprogram
    '''

    # Parse input file
    config = inps.PredictConfig.from_file(uargs.input_file)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Either load hyperfines from DFT output file
    if config.hyperfine_method == 'dft':
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        # Write raw calculation data to output file
        qc_hyperfine_data.save_to_csv(
            os.path.join(config.project_name, 'dft_hyperfines.csv'),
            verbose=True,
            delimiter=CSV_DELIMITER,
            comment=f'# Data taken from file {config.hyperfine_file}'
        )

        # Create molecule object from quantum chemical hyperfine data
        # Retain only the atoms that are given in the labels file
        base_molecule = main.Molecule.from_QCA(
            qc_hyperfine_data, converter='MHz_to_Ang-3',
            elements=config.nuclei_include
        )

    # generate using point dipole approximation
    elif config.hyperfine_method == 'pdip':

        if os.path.splitext(config.hyperfine_file)[1] == '.xyz':
            labels, coords = xyzp.load_xyz(config.hyperfine_file)
        elif os.path.splitext(config.hyperfine_file)[1] in ['.log', '.out']:
            QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
            labels = QCS.labels
            coords = QCS.coords
        else:
            ut.cprint(f'Specified hyperfine file format {os.path.splitext(uargs.structure_file)[1]} unsupported', 'red') # noqa
            sys.exit(1)

        # Create molecule
        base_molecule = main.Molecule.from_labels_coords(
            labels, coords, elements=config.nuclei_include
        )

        # Calculate point dipole hyperfine
        base_molecule.calc_pdip(config.hyperfine_pdip_centres)

    # or load from CSV
    elif config.hyperfine_method == 'csv':
        base_molecule = main.Molecule.from_csv(
            config.hyperfine_file, elements=config.nuclei_include
        )

    # Add chemical labels
    if len(config.chem_labels_file):
        base_molecule.add_chem_labels_from_file(
            config.chem_labels_file
        )

        # Save xyz file with chemical labels for chemcraft
        base_molecule.save_chemcraft_xyz(
            file_name=os.path.join(
                config.project_name,
                'chemcraft_structure.xyz'
            )
        )

    # Save xyz file with chemical labels for chemcraft
    base_molecule.save_xyz(
        file_name=os.path.join(
            config.project_name,
            'structure.xyz'
        ),
        comment=f'Structure from {config.hyperfine_file}'
    )

    # Load diamagnetic shift file
    if len(config.diamagnetic_file):
        base_molecule.load_diamagnetic_shifts(
            config.diamagnetic_file,
            config.diamagnetic_method,
            config.diamagnetic_ref_file,
            config.diamagnetic_ref_method
        )

    # Rotationally average hyperfines of user selected nuclei:
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Rotate hyperfine tensors from DFT frame into chi eigenframe
    if 'orca' in config.susceptibility_format:
        rot_mat, trans_mat = tfm.get_rotation_and_transformation()
        base_molecule.rotate_hyperfines(rot_mat)

    # Transform HFC coordinates into the chi eigenframe and save
    if 'orca' in config.susceptibility_format:
        tfm.rotate_coords_to_chi_frame(config.project_name)

    # Load susceptibility information
    if 'orca' in config.susceptibility_format:
        suscs = main.Susceptibility.from_orca(
            config.susceptibility_file,
            section=config.susceptibility_format.split('orca_')[1]
            # section = 'auto'
        )
    elif 'csv' in config.susceptibility_format:
        suscs = main.Susceptibility.from_csv(
            config.susceptibility_file
        )
    elif 'molcas' in config.susceptibility_format:
        ut.red_exit('Molcas files are not currently supported')

    suscs = [
        susc
        for susc in suscs
        if susc.temperature in config.susceptibility_temperatures
    ]

    if not len(suscs):
        ut.red_exit(
            'Error: No susceptibility data found for specified temperature(s)'
        )
        
    # Calculate linewidths using user-specified relaxation model (optional)
    if getattr(config, "relaxation_model", None):
        # Solomon linewidths if relaxation model is SBM
        nuclei_labels = config.nuclei_include if isinstance(
            config.nuclei_include, list) else [config.nuclei_include]
        # Use all nuclei in the molecule that match the requested element(s)
        nuclei_coords = {
            nuc.label: nuc.coord
            for nuc in base_molecule.nuclei
            if ut.st.remove_numbers(nuc.label) in nuclei_labels
        }
        electron_coords = config.relaxation_electron_coords
        B0 = config.relaxation_magnetic_field_tesla
        if B0 is None:
            ut.cprint("Warning: relaxation_magnetic_field_tesla not provided — relaxation effects skipped", "black_yellowbg")
        else:
            # Build Aiso, gamma and omega dictionaries for selected nuclei
            # Converts nuclear gyromagnetic ratios from MHz/T to rad/s/T
            # and multiplies Aiso by 1e6 to convert from MHz to Hz
            qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)  # noqa
            A_iso_dict_MHz = qc_hyperfine_data.a_iso  # MHz
            A_iso_dict = {
                nuc.label: A_iso_dict_MHz[nuc.label] * 1e6
                for nuc in base_molecule.nuclei
                if nuc.label in nuclei_coords
            }
            gamma_I_dict = {
                label: ut.NUCLEAR_GAMMAS[ut.st.remove_numbers(
                    label)] * 2 * np.pi * 1e6
                for label in nuclei_coords
            }
            omega_I_dict = {
                label: gamma_I_dict[label] * B0
                for label in nuclei_coords
            }
            omega_S = ut.EGAMMA * B0 * 2 * np.pi * 1e6
            tau_c1 = 1 / ((1 / config.relaxation_tR) +
                          (1 / config.relaxation_T1e))
            tau_c2 = 1 / ((1 / config.relaxation_tR) +
                          (1 / config.relaxation_T2e))
            tau_e1 = config.relaxation_T1e
            tau_e2 = config.relaxation_T2e
            tau_R = config.relaxation_tR

            if config.spin_S is not None:
                spin = config.spin_S
            else:
                spin = rdrs.QCSpin.guess_from_file(config.hyperfine_file).S

            if config.relaxation_model == "sbm":
                # Calculate SBM dipolar rates (R1)
                sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    gamma_I_dict,
                    omega_I_dict,
                    omega_S,
                    tau_c1,
                    tau_c2,
                    spin
                )
                # Calculate SBM contact rates (R1)
                sbm_contact_r1_rates = ut.sbm_r1_contact(
                    list(nuclei_coords.keys()),
                    A_iso_dict,
                    omega_I_dict,
                    omega_S,
                    tau_e2,
                    spin
                )
                # Calculate SBM dipolar rates (R2)
                sbm_dipolar_r2_rates = ut.sbm_r2_dipolar(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    gamma_I_dict,
                    omega_I_dict,
                    omega_S,
                    tau_c1,
                    tau_c2,
                    spin
                )
                # Calculate SBM contact rates (R2)
                sbm_contact_r2_rates = ut.sbm_r2_contact(
                    list(nuclei_coords.keys()),
                    A_iso_dict,
                    omega_I_dict,
                    omega_S,
                    tau_e1,
                    tau_e2,
                    spin
                )
                # Combine rates into a single dictionary
                rates_r1 = {
                    label: sbm_dipolar_r1_rates[label] +
                    sbm_contact_r1_rates[label]
                    for label in nuclei_coords
                }
                rates_r2 = {
                    label: sbm_dipolar_r2_rates[label] +
                    sbm_contact_r2_rates[label]
                    for label in nuclei_coords
                }
            # Curie mechanism only (R1 and R2)
            elif config.relaxation_model == "curie":
                curie_r1_rates = ut.gueron_r1_curie(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    omega_I_dict,
                    config.relaxation_temperature,
                    tau_R,
                    spin
                )
                curie_r2_rates = ut.gueron_r2_curie(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    omega_I_dict,
                    config.relaxation_temperature,
                    tau_R,
                    spin
                )
                rates_r1 = {label: curie_r1_rates[label]
                            for label in nuclei_coords}
                rates_r2 = {label: curie_r2_rates[label]
                            for label in nuclei_coords}
            # Combined SBM and Curie mechanisms
            elif config.relaxation_model == "sbm curie" or config.relaxation_model == "curie sbm":  # noqa
                sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    gamma_I_dict,
                    omega_I_dict,
                    omega_S,
                    tau_c1,
                    tau_c2,
                    spin
                )
                sbm_contact_r1_rates = ut.sbm_r1_contact(
                    list(nuclei_coords.keys()),
                    A_iso_dict,
                    omega_I_dict,
                    omega_S,
                    tau_e1,
                    spin
                )
                sbm_dipolar_r2_rates = ut.sbm_r2_dipolar(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    gamma_I_dict,
                    omega_I_dict,
                    omega_S,
                    tau_c1,
                    tau_c2,
                    spin
                )
                #print("SBM Dipolar Rates:")
                #for label, rate in sbm_dipolar_r2_rates.items():
                #    print(f"{label}: {rate:.3e} s^-1")

                # Calculate SBM contact rates
                sbm_contact_r2_rates = ut.sbm_r2_contact(
                    list(nuclei_coords.keys()),
                    A_iso_dict,
                    omega_I_dict,
                    omega_S,
                    tau_e1,
                    tau_e2,
                    spin
                )
                # print("SBM Contact Rates:")
                # for label, rate in sbm_contact_r2_rates.items():
                #     print(f"{label}: {rate:.3e} s^-1")
                curie_r1_rates = ut.gueron_r1_curie(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    omega_I_dict,
                    config.relaxation_temperature,
                    tau_R,
                    spin
                )
                curie_r2_rates = ut.gueron_r2_curie(
                    list(nuclei_coords.keys()),
                    nuclei_coords,
                    electron_coords,
                    omega_I_dict,
                    config.relaxation_temperature,
                    tau_R,
                    spin
                )
                # print("Curie Rates:")
                # for label, rate in curie_r2_rates.items():
                #     print(f"{label}: {rate:.3e} s^-1")

                rates_r1 = {
                    label: sbm_dipolar_r1_rates[label] + sbm_contact_r1_rates[label] + curie_r1_rates[label]  # noqa
                    for label in nuclei_coords
                }
                rates_r2 = {
                    label: sbm_dipolar_r2_rates[label] + sbm_contact_r2_rates[label] + curie_r2_rates[label]  # noqa
                    for label in nuclei_coords
                }


            # Group rates by chemical label
            r1_by_chem_label = defaultdict(list)
            for nuc in base_molecule.nuclei:
                if nuc.label in rates_r1:
                    r1_by_chem_label[nuc.chem_label].append(
                        rates_r1[nuc.label])
            r2_by_chem_label = defaultdict(list)
            for nuc in base_molecule.nuclei:
                if nuc.label in rates_r2:
                    r2_by_chem_label[nuc.chem_label].append(
                        rates_r2[nuc.label])
            # Calculate average R1 rates for each chemical label
            avg_r1_by_chem_label = {
                chem_label: np.mean(rate_list)
                for chem_label, rate_list in r1_by_chem_label.items()
            }
            # Calculate average R2 rates for each chemical label
            avg_r2_by_chem_label = {
                chem_label: np.mean(rate_list)
                for chem_label, rate_list in r2_by_chem_label.items()
            }
            # Calculate average linewidths for each chemical label (Hz)
            avg_lw_by_chem_label = {
                chem_label: np.mean([rate / np.pi for rate in rate_list])
                for chem_label, rate_list in r2_by_chem_label.items()
            }

            # Optional decomposition of R1 into SBM and Curie components
            avg_dipolar_by_chem_label = None
            avg_contact_by_chem_label = None
            avg_curie_by_chem_label = None

            if 'sbm' in config.relaxation_model:
                dipolar_by_chem_label = defaultdict(list)
                contact_by_chem_label = defaultdict(list)
                for nuc in base_molecule.nuclei:
                    if 'sbm_dipolar_r1_rates' in locals() and nuc.label in sbm_dipolar_r1_rates:
                        dipolar_by_chem_label[nuc.chem_label].append(
                            sbm_dipolar_r1_rates[nuc.label]
                        )
                    if 'sbm_contact_r1_rates' in locals() and nuc.label in sbm_contact_r1_rates:
                        contact_by_chem_label[nuc.chem_label].append(
                            sbm_contact_r1_rates[nuc.label]
                        )
                avg_dipolar_by_chem_label = {
                    chem_label: np.mean(rate_list)
                    for chem_label, rate_list in dipolar_by_chem_label.items()
                }
                avg_contact_by_chem_label = {
                    chem_label: np.mean(rate_list)
                    for chem_label, rate_list in contact_by_chem_label.items()
                }

            if 'curie' in config.relaxation_model:
                curie_by_chem_label = defaultdict(list)
                for nuc in base_molecule.nuclei:
                    if 'curie_r1_rates' in locals() and nuc.label in curie_r1_rates:
                        curie_by_chem_label[nuc.chem_label].append(
                            curie_r1_rates[nuc.label]
                        )
                avg_curie_by_chem_label = {
                    chem_label: np.mean(rate_list)
                    for chem_label, rate_list in curie_by_chem_label.items()
                }

            # Save the relaxation data to CSV
            outputs.save_relaxation_decomposition(
                file_name=os.path.join(
                    config.project_name,
                    "relaxation_decomposition.csv"
                ),
                avg_r1_by_chem_label=avg_r1_by_chem_label,
                avg_r2_by_chem_label=avg_r2_by_chem_label,
                avg_lw_by_chem_label=avg_lw_by_chem_label,
                avg_dipolar_by_chem_label=avg_dipolar_by_chem_label,
                avg_contact_by_chem_label=avg_contact_by_chem_label,
                avg_curie_by_chem_label=avg_curie_by_chem_label,
            )

            for nuc in base_molecule.nuclei:
                if nuc.chem_label in avg_lw_by_chem_label:
                    nuc.shift.lw = avg_lw_by_chem_label[nuc.chem_label] / (abs(omega_I_dict[nuc.label]) / (2 * np.pi)) * 1e6  # noqa
    else:
        ut.cprint("\n No relaxation model specified — linewidths will be fixed at 1 ppm.\n", "cyan")

    # Load experimental data from file into list of experiment objects
    if len(config.experiment_files):
        experiments = main.Experiment.from_file(config.experiment_files)
        for susc, exp in zip(suscs, experiments):
            if susc.temperature != exp.temperature:
                ut.cprint(
                    f'Warning: Mismatch in Susceptibility ({susc.temperature:.2f} K) and Experimental ({exp.temperature:.2f} K) temperatures', # noqa
                    'black_yellowbg'
                )
            if re.sub('[0-9]', '', exp.isotope) not in config.nuclei_include:
                ut.cprint(
                    f'Warning: Experimental isotope ({exp.isotope}) not requested in input file ({config.nuclei_include})', # noqa
                    'black_yellowbg'
                )
    else:
        experiments = [None] * len(suscs)

    # Create a molecule object which accompanies each experiment object
    molecules = [
        copy.deepcopy(base_molecule)
        for _ in range(len(experiments))
    ]

    if len(config.experiment_spectrum_files):
        for experiment, spectrum in zip(experiments, config.experiment_spectrum_files): # noqa
            experiment.load_spectrum_from_file(spectrum)

    _terms = ['pc', 'fc', 'd']

    if config.hyperfine_method == 'pdip':
        _terms.pop(_terms.index('fc'))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index('d'))

    # Try to read the spin from config (YAML)
    spin = config.spin_S

    # If the spin is not provided, try to infer from QC file safely
    if spin is None:
        ext = os.path.splitext(config.hyperfine_file)[1].lower()
        try:
            if config.hyperfine_method == 'dft' or ext in ('.log', '.out'):
                spin_obj = rdrs.QCSpin.guess_from_file(config.hyperfine_file)
                spin = spin_obj.S
        except SystemExit:
            spin = None

    # Update susceptibility tensor of Molecule using model
    for molecule, susc, experiment in zip(molecules, suscs, experiments):
        molecule.susc = susc

        if 'orca' in config.susceptibility_format:
            susc.iso = ut.get_true_iso_susceptibility(uargs, susc.temperature)
        elif spin is not None:
            susc.iso = ut.get_spin_only_susceptibility(uargs, susc.temperature)
        else:
            ut.cprint(
                "\n Spin not specified and could not be inferred — using χ_iso from susceptibility file (no spin-only correction)\n",
                "cyan"
            )

        # Calculate shifts using new susceptibility tensor and rotated hyperfines
        molecule.calculate_shifts()

        # Calculate average shifts
        molecule.average_shifts()

        # Plot theoretical shifts
        # Spread
        vis.plot_shift_spread(
            molecule,
            experiment=experiment,
            save=True,
            show=False,
            terms=_terms,
            save_name=os.path.join(config.project_name, f'pred_shift_spread_{molecule.susc.temperature:.2f}_K{PFF}'), # noqa
            verbose=True,
            window_title=f'Spread of predicted shifts at {susc.temperature:.2f} K', # noqa
            order='descending'
        )

        # Bar chart for means
        vis.plot_shift_contrib(
            molecule,
            experiment=experiment,
            save=True,
            show=False,
            save_name=os.path.join(config.project_name, f'pred_mean_components_{molecule.susc.temperature:.2f}_K{PFF}'), # noqa
            verbose=True,
            window_title=f'Predicted mean shifts and components at {susc.temperature:.2f} K', # noqa
            order='descending'
        )

        shift_range = [
            np.min([nuc.shift.avg for nuc in molecule.nuclei]),
            np.max([nuc.shift.avg for nuc in molecule.nuclei])
        ]

        extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]

        shift_range = [
            shift_range[0] + np.negative(np.max(extras)),
            shift_range[1] + np.positive(np.max(extras))
        ]

        if len(config.experiment_spectrum_files):
            vis.plot_raw_deconv_pred(
                molecule=molecule,
                experiment=experiment,
                save=True,
                show=False,
                save_name=os.path.join(
                    config.project_name,
                    f'pred_and_exp_spectrum_{molecule.susc.temperature:.2f}_K{PFF}' # noqa
                )
            )
        vis.plot_pred_spectrum(
            molecule,
            isotope=molecule.nuclei[0].isotope,
            shift_range=shift_range,
            save=True,
            show=False,
            save_name=os.path.join(
                config.project_name,
                f'pred_spectrum_{molecule.susc.temperature:.2f}_K{PFF}'
            ),
        )

        plt.show()

        plt.close('all')

    # TODO If more than one temperature, then make a stacked plot of spectra

    # Save susceptibility data to file
    outputs.save_susc(
        molecules,
        os.path.join(
            config.project_name,
            'susceptibility_tensor.csv'
        ),
        comment='#Data from {} ({})'.format(
            config.susceptibility_file,
            config.susceptibility_format
        ),
        susc_units=uargs.susc_units
    )

    # Write shift data to file
    for molecule in molecules:
        molecule.to_csv(
            os.path.join(
                config.project_name,
                f'hyperfines_and_shifts_{molecule.susc.temperature:.2f}_K.csv'
            ),
            delimiter=CSV_DELIMITER,
            comment=f'# T = {molecule.susc.temperature:.2f} K'
        )

    return


def plot_shift_tdep_func(uargs):
    '''
    Wrapper for cli call to plot_shift_tdep
    '''

    experiments = main.Experiment.from_file(uargs.experiment_files)

    vis.plot_shift_tdep(
        experiments,
        'ShiftT_vs_T',
        show=True,
        save=True,
        save_name=f'shift_x_T_vs_T{PFF}'
    )

    vis.plot_shift_tdep(
        experiments,
        'Shift_vs_1/T',
        show=True,
        save=True,
        save_name=f'shift_vs_T-1{PFF}'
    )

    return


def read_args(arg_list=None):
    '''
    Reader for command line arguments. Uses subReaders for individual programs

    Parameters
    ----------
        args : argReader object
            command line arguments

    Returns
    -------
        None

    '''

    description = '''
    A package for fitting susceptibility tensors from paramagnetic NMR data
    '''

    epilog = (
        'To display options for a specific program, use\n\n'
        f'      {ut.cstr('simpnmr SUBPROGRAM -h', 'green')}'
    )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog
    )

    parser._positionals.title = 'Subprograms'

    subparsers = parser.add_subparsers(dest='prog')

    extract_dia = subparsers.add_parser(
        'extract_dia',
        description='Extract diamagnetic shifts from quantum chemistry output'
    )
    extract_dia.set_defaults(func=extract_dia_func)

    extract_dia.add_argument(
        'output_file',
        type=str,
        help=(
            'Quantum Chemistry output file containing chemical shift information' # noqa
        )
    )

    extract_dia.add_argument(
        '--ref_output_file',
        metavar='ref_output_file',
        default='',
        type=str,
        help=(
            'Quantum Chemistry output file containing reference chemical shift information' # noqa
        )
    )

    fit_susc = subparsers.add_parser(
        'fit_susc',
        description=(
            'Fit susceptibility tensor using DFT hyperfines and'
            ' experimental peaks'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    fit_susc.set_defaults(func=fit_susc_func)

    fit_susc.add_argument(
        'input_file',
        type=str,
        help=(
            'Input file for fit_susc -- see documentation for format'
        )
    )

    fit_susc.add_argument(
        '--dry_run',
        action='store_true',
        help=(
            'Checks input file, loads data, but quits before simulation'
        )
    )

    fit_susc.add_argument(
        '--pcs_isosurface',
        action='store_true',
        help=(
            'Saves PCS isosurface for each temperature to separate cube files'
        )
    )

    fit_susc.add_argument(
        '--susc_units',
        '-su',
        choices=['cm3 mol-1', 'A3'],
        metavar='<str>',
        type=str,
        default='A3',
        help=(
            'Controls susceptibility units of plots \n'
            '(wrap with "")\n'
            'Default: A3'
        )
    )

    fit_susc.add_argument(
        '--shift_plots',
        choices=['on', 'show', 'save', 'off'],
        metavar='<str>',
        type=str,
        default='save',
        help=(
            'Plot Experimental and Calculated Chemical shifts against each other \n' # noqa
            ' - \'on\' shows and saves the plots\n'
            ' - \'show\' shows the plots\n'
            ' - \'save\' saves the plots\n'
            ' - \'off\' neither shows nor saves\n'
            'Default: save'
        )
    )

    fit_susc.add_argument(
        '--spread_plots',
        choices=['on', 'show', 'save', 'off'],
        metavar='<str>',
        type=str,
        default='save',
        help=(
            'Plot spread of contributions to calculated shifts \n'
            ' - \'on\' shows and saves the plots\n'
            ' - \'show\' shows the plots\n'
            ' - \'save\' saves the plots\n'
            ' - \'off\' neither shows nor saves\n'
            'Default: save'
        )
    )

    fit_susc.add_argument(
        '--contrib_plots',
        choices=['on', 'show', 'save', 'off'],
        metavar='<str>',
        type=str,
        default='save',
        help=(
            'Plot mean of contributions to mean calculated shifts \n'
            ' - \'on\' shows and saves the plots\n'
            ' - \'show\' shows the plots\n'
            ' - \'save\' saves the plots\n'
            ' - \'off\' neither shows nor saves\n'
            'Default: on'
        )
    )

    fit_susc.add_argument(
        '--isoaxrho_plots',
        choices=['on', 'show', 'save', 'off'],
        metavar='<str>',
        type=str,
        default='save',
        help=(
            'Plot Isotropic, Axial, and Rhombic susceptibility as a function of temperature \n' # noqa
            ' - \'on\' shows and saves the plots\n'
            ' - \'show\' shows the plots\n'
            ' - \'save\' saves the plots\n'
            ' - \'off\' neither shows nor saves\n'
            'Default: save'
        )
    )

    extract_a = subparsers.add_parser(
        'extract_a',
        description=(
            'Extract A tensor information from quantum chemistry output'
        )
    )
    extract_a.set_defaults(func=extract_a_func)

    extract_a.add_argument(
        'calculation_data',
        type=str,
        help=(
            'Gaussian log file, or Orca output or property file'
        )
    )

    plot_a = subparsers.add_parser(
        'plot_a',
        description=(
            'Plot A tensor information from quantum chemistry output'
        )
    )
    plot_a.set_defaults(func=plot_a_func)

    plot_a.add_argument(
        'calculation_data',
        type=str,
        help=(
            'Gaussian log file, or Orca output or property file'
        )
    )

    plot_a.add_argument(
        'components',
        choices=[
            'iso', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz',
            'dxx', 'dxy', 'dxz', 'dyx', 'dyy', 'dyz', 'dzx', 'dzy', 'dzz',
            'ax', 'rho'
        ],
        nargs='+',
        help=(
            'Component() to plot'
        )
    )

    plot_a.add_argument(
        '--chem_labels',
        type=str,
        help=(
            'chemical label file (.csv)'
        )
    )

    plot_a.add_argument(
        '--elements',
        type=str,
        nargs='*',
        default='all',
        help=(
            'Elements to include in plot'
        )
    )

    plot_a.add_argument(
        '--save',
        action='store_false',
        help=(
            'Save plot to file'
        )
    )

    plot_a.add_argument(
        '--hide_plots',
        action='store_true',
        help=(
            'Display plot on screen'
        )
    )

    plot_a_iso = subparsers.add_parser(
        'plot_a_iso_ax',
        description=(
            'Plot A tensor information from quantum chemistry output'
        )
    )

    plot_a_iso.set_defaults(func=plot_a_iso_ax_func)

    plot_a_iso.add_argument(
        'input_file',
        type=str,
        help=(
            'simpnmr Input file'
        )
    )

    plot_a_iso.add_argument(
        '--save',
        action='store_true',
        help=(
            'Save plot to file'
        )
    )

    plot_a_iso.add_argument(
        '--hide_plots',
        action='store_true',
        help=(
            'Display plot on screen'
        )
    )

    calc_pdip = subparsers.add_parser(
        'calc_pdip',
        description=(
            'Calculate dipolar Hyperfine tensor using point dipole '
            'approximation'
        )
    )
    calc_pdip.set_defaults(func=calc_pdip_func)

    calc_pdip.add_argument(
        'structure_file',
        type=str,
        help=(
            'File containing molecular structure: .xyz, .log, ORCA .out'
        )
    )

    calc_pdip.add_argument(
        'centres',
        type=str,
        nargs='+',
        help=(
            'Atomic label (with index number) of paramagnetic centre(s)'
        )
    )

    calc_pdip.add_argument(
        '--chem_labels',
        type=str,
        help=(
            'chemical label file (.csv)'
        )
    )

    calc_pdip.add_argument(
        '--elements',
        type=str,
        nargs='*',
        default='all',
        help=(
            'Elements to include in plot'
        )
    )

    calc_pdip.add_argument(
        '--plot_components',
        default=[],
        choices=[
            'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'x', 'y',
            'z', 'ax', 'rho'
        ],
        nargs='+',
        help=(
            'Component(s) to plot'
        )
    )

    calc_pcs_iso = subparsers.add_parser(
        'calc_pcs_iso',
        description=(
            'Calculates PCS isosurface and saves to .cube file'
        )
    )
    calc_pcs_iso.set_defaults(func=calc_pcs_iso_func)

    calc_pcs_iso.add_argument(
        'susc_file',
        help='File containing susceptibility data'
    )

    calc_pcs_iso.add_argument(
        'susc_format',
        help='Susceptibility file format',
        choices=['csv', 'orca_nev', 'orca_cas', 'molcas']
    )

    calc_pcs_iso.add_argument(
        'temperatures',
        nargs='+',
        help='Temperatures for which to produce pcs isosurface plot',
        type=float
    )

    calc_pcs_iso.add_argument(
        'structure_file',
        help=(
            'File containing molecular structure'
            'Either Gaussian .log, ORCA .out, or plain old .xyz'
        )
    )

    calc_pcs_iso.add_argument(
        'central_atom',
        help='Atom on which isosurface is centered. Must include indexing'
    )

    predict = subparsers.add_parser(
        'predict',
        description='Calculate shifts using Hyperfine and Susceptibility'
    )
    predict.set_defaults(func=predict_func)

    predict.add_argument(
        'input_file',
        type=str,
        help=(
            'Input file for predict - see documentation for format'
        )
    )

    predict.add_argument(
        '--susc_units',
        '-su',
        choices=['cm3 mol-1', 'A3'],
        metavar='<str>',
        type=str,
        default='A3',
        help=(
            'Controls susceptibility units of plots and output files \n'
            '(wrap with "")\n'
            'Default: A3'
        )
    )

    plot_shift_tdep = subparsers.add_parser(
        'plot_shift_tdep',
        description='Calculate shifts using Hyperfine and Susceptibility'
    )
    plot_shift_tdep.set_defaults(func=plot_shift_tdep_func)

    plot_shift_tdep.add_argument(
        'experiment_files',
        type=str,
        nargs='+',
        help=(
            'simpnmr experiment.csv files'
        )
    )

    # Read sub-parser and parse arguments
    parser.set_defaults(func=lambda args: parser.print_help())
    args = parser.parse_args(arg_list)
    args.func(args)

    return args


def interface():
    read_args()
    return
