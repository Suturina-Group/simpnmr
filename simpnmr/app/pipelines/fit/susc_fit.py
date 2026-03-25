# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit susceptibility tensors to experimental shift data.

Loads inputs, fits a selected susceptibility model, and writes outputs and plots.
"""

import copy
import logging
import os

import numpy as np
from pathos import multiprocessing as mp

# Application layer
from simpnmr.app.loaders.dia_load import load_diamagnetic_shifts
from simpnmr.app.loaders.elstate_load import load_electronic_state
from simpnmr.app.loaders.exp_load import load_experiments, save_experiment
from simpnmr.app.loaders.hfc_load import load_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_base_molecule
from simpnmr.app.loaders.paramag_centre_load import load_paramagnetic_centre
from simpnmr.app.loaders.sh_load import load_g_tensor_dft
from simpnmr.app.params.options import FitSuscRunOptions
from simpnmr.app.pipelines.fit.vt_fit import fit_vt
from simpnmr.app.policies.assignment import resolve_assignment_search_settings
from simpnmr.app.policies.hfc import has_missing_selected_chem_labels
from simpnmr.app.policies.linewidth import resolve_output_linewidths
from simpnmr.app.policies.susc import resolve_susc_fit_variables

# Core / domain
from simpnmr.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.domain.tensor import Hyperfine
from simpnmr.core.fitting import models
from simpnmr.core.util.strings import remove_numbers
from simpnmr.core.fitting.assign import (
    fit_with_hungarian_assignment,
    generate_assignment_permutations,
)
from simpnmr.core.pcs.isosurf import compute_pcs_isosurface

# IO layer
from simpnmr.io.csv.mol import save_molecule_to_csv
from simpnmr.io.csv.susc import save_susc
from simpnmr.io.cube.pcs_iso_write import write_pcs_cube
from simpnmr.io.xyz import xyz_write
from simpnmr.core.fitting.r6_fit import fit_r6
from simpnmr.viz.plots.fitted_shifts import plot_fitted_shifts
from simpnmr.viz.plots.r6_fit import plot_r6_fit
from simpnmr.viz.plots.spect import plot_raw_deconv_pred
from simpnmr.viz.plots.shift_width_bubble import plot_shift_width_bubble

# Visualisation
from simpnmr.viz.plots.shifts import plot_shift_contrib, plot_shift_spread
from simpnmr.viz.plots.spect import plot_pred_spectrum
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_fit_susc(config, options: FitSuscRunOptions | None = None) -> int:
    """Fit susceptibility tensor(s) defined by a YAML configuration file.

    The pipeline builds a Molecule from the requested hyperfine source, loads
    experimental data, fits the chosen susceptibility model, generates plots, and
    writes outputs into the project directory.

    Args:
        config: FitSuscConfig loaded from YAML.
        options: Runtime options supplied by the CLI.

    Returns:
        Exit code: 0 on success.
    """
    if options is None:
        raise ValueError("FitSuscRunOptions is required")

    delimiter = options.runtime.csv_delimiter

    # Build the resolved plotting contract once per run.
    spec = apply_profile(options.runtime.plot_profile)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Load Molecule
    base_molecule = load_base_molecule(config)

    # Load canonical paramagnetic centre into the molecule domain container
    base_molecule = load_paramagnetic_centre(
        molecule=base_molecule,
        paramagnetic_centre=config.hyperfine_paramagnetic_centre,
    )

    # Load DFT g-tensor (if available)
    base_molecule.sh.g_tensor_dft = load_g_tensor_dft(
        config=config,
    )

    # Load Hyperfines
    base_molecule = load_hyperfines(
        molecule=base_molecule,
        config=config,
    )

    # Load electronic state
    base_molecule.electronic = load_electronic_state(
        spin_S=config.spin_S,
        orbit_L=config.orbit,
        total_J=config.total_momentum_J,
        hyperfine_file=config.hyperfine_file,
        hyperfine_method=config.hyperfine_method,
    )
    spin = base_molecule.electronic.spin_S

    # Add chemical labels
    if len(config.chem_labels_file):
        try:
            al_to_cl, al_to_cml = load_chem_labels_from_csv(config.chem_labels_file)
            if has_missing_selected_chem_labels(base_molecule, al_to_cl):
                logger.warning(
                    "Chemical labels file does not define labels for all selected "
                    "nuclei; missing labels will use atom labels."
                )
            base_molecule.apply_chem_labels(al_to_cl, al_to_cml)
        except ValueError as err:
            raise ValueError(f"{err}\nCheck chem_labels and hyperfine files.")
        except KeyError as err:
            # treat missing labels/keys as a user input error
            raise ValueError(str(err))

        # Save xyz file with chemical labels for chemcraft
        xyz_write.save_chemcraft_xyz(
            file_name=os.path.join(config.project_name, "chemcraft_structure.xyz"),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
            chem_labels={nuc.label: nuc.chem_label for nuc in base_molecule.nuclei},
        )

    # Save xyz file with chemical labels for chemcraft
    xyz_write.save_xyz(
        file_name=os.path.join(config.project_name, "structure.xyz"),
        labels=base_molecule.labels,
        coords=base_molecule.coords,
        comment=f"Structure from {config.hyperfine_file}",
    )

    # Apply rotation matrix to all hyperfine tensors
    # if requested
    if len(config.hyperfine_rotate):
        _rot_a = np.loadtxt(config.hyperfine_rotate)
        base_molecule.rotate_hyperfines(_rot_a)

    # Load diamagnetic shift file
    if len(config.diamagnetic_file):
        dia_by_key, key_kind, ref_avg_by_label_nn = load_diamagnetic_shifts(
            file_name=config.diamagnetic_file,
            file_type=config.diamagnetic_method,
            ref_file_name=config.diamagnetic_ref_file,
            ref_file_type=config.diamagnetic_ref_method,
        )
        base_molecule.apply_diamagnetic_shifts(
            dia_by_key=dia_by_key,
            key_kind=key_kind,
            ref_avg_by_label_nn=ref_avg_by_label_nn,
        )

    # Rotationally average hyperfines
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Create experiments
    experiments = load_experiments(config.experiment_files)

    # Check the number of experiments is consistent across the files
    # and issue warning if not
    if len(np.unique([len(exp.signals) for exp in experiments])) > 1:
        logger.warning("Some experiments have more signals than others!")

    # Create a molecule object to accompany each experiment object
    molecules = [copy.deepcopy(base_molecule) for _ in range(len(experiments))]

    name_to_susc_fit: dict[str, models.SusceptibilityModel] = {
        "full": models.FullSuscFitter,
        "split": models.SplitFitter,
        "isoaxrho": models.IsoAxRhoFitter,
        "eigen": models.EigenFitter,
        "isoeigen": models.IsoEigenFitter,
    }

    model_to_use = name_to_susc_fit[config.susc_fit_type]

    # Create one susceptibility model per molecule/experiment pair. Reduced
    # input units depend on experiment temperature, so normalization happens
    # per experiment here rather than once at config-load time.
    susc_models: list[models.SusceptibilityModel] = []
    for experiment in experiments:
        fit_vars, fix_vars = resolve_susc_fit_variables(
            raw_variables=config.susc_fit_variables,
            input_units=config.susc_fit_input_units,
            temperature=experiment.temperature,
            spin=spin,
        )
        susc_models.append(model_to_use(fit_vars, fix_vars))

    if options.dry_run:
        logger.info("Dry run successful — no computations executed")
        return 0

    if len(config.susc_fit_average_shifts):
        if "all" in config.susc_fit_average_shifts:
            config.susc_fit_average_shifts = list(
                {nuc.chem_label for nuc in base_molecule.nuclei}
            )
        average_labels = [
            [nuc.label for nuc in base_molecule.nuclei if nuc.chem_label == _cl]
            for _cl in config.susc_fit_average_shifts
        ]
    else:
        average_labels = []

    # Shift terms for plots
    # does not affect fit!
    _terms = ["pc", "fc", "d"]
    if config.hyperfine_method == "pdip":
        _terms.pop(_terms.index("fc"))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index("d"))

    # Run fit for all experiments
    for molecule, susc_model, experiment in zip(molecules, susc_models, experiments):
        # If permuting assignments, then first
        # run all assignment permutations to find best one
        if config.assignment_method == "permute":
            # If no permutation groups provided, permute all
            if not len(config.assignment_groups):
                config.assignment_groups = [
                    list({nuc.chem_label for nuc in molecule.nuclei})
                ]
            # For the current experiment, generate a new set in which
            # the assignment is permuted according to user defined groups
            permed_assignments = generate_assignment_permutations(
                experiment=experiment, groups=config.assignment_groups
            )

            logger.info("There are %s possible permutations", len(permed_assignments))

            # For each permutation, fit tensor and store r2_adjusted

            # Number of threads
            if config.num_threads == "auto":
                num_threads = mp.cpu_count() - 1
            else:
                num_threads = config.num_threads

            if num_threads > len(permed_assignments):
                num_threads = len(permed_assignments)

            # Create parallel pool
            pool = mp.Pool(num_threads)
            logger.info(
                "Parallel permutation search: %s worker processes",
                num_threads,
            )

            echo_r2 = options.runtime.echo_r2

            iterables = [
                (
                    molecule,
                    permed_assgn,
                    susc_model,
                    copy.deepcopy(experiment),
                    average_labels,
                    echo_r2,
                )
                for permed_assgn in permed_assignments
            ]

            # Calculate each assignment's r2 in parallel
            results = pool.starmap(_obtain_r2a, iterables)

            # Close Pool and let all the processes complete
            pool.close()
            pool.join()

            # Find assignment with smallest RMSE
            # and use in subsequent (re)fitting
            assignment = permed_assignments[np.nanargmin(results)]
            opt_rmse = np.nanmin(results)
            logger.info("Optimal assignment with RMSE = %.6f", opt_rmse)

            # and swap in new, permuted, assignments
            for it, new in enumerate(assignment):
                experiment.signals[it].assignment = new

            # Save assigned experiment to file
            save_experiment(
                experiment,
                file_name=os.path.join(
                    config.project_name,
                    f"assigned_experiment_{experiment.temperature:.2f}_K.csv",
                ),
                delimiter=delimiter,
                comment=(
                    f"Optimal Assignment\n"
                    f"rmse = {opt_rmse:f}\n"
                    f"T = {experiment.temperature:.2f} K"
                ),
            )

        elif config.assignment_method == "hungarian":
            search_settings = resolve_assignment_search_settings(
                mode=config.assignment_search,
                n_attempts=config.assignment_n_attempts,
                max_iter=config.assignment_max_iter,
                rmse_threshold=config.assignment_rmse_threshold,
            )
            logger.info(
                "Hungarian search policy resolved: mode=%s, n_attempts=%d, "
                "max_iter=%d, rmse_threshold=%.6f",
                search_settings.mode,
                search_settings.n_attempts,
                search_settings.max_iter,
                search_settings.rmse_threshold,
            )

            # Call Hungarian assignment function
            opt_rmse, assignment = fit_with_hungarian_assignment(
                molecule=molecule,
                susc_model=susc_model,
                experiment=experiment,
                average_labels=average_labels,
                n_attempts=search_settings.n_attempts,
                max_iter=search_settings.max_iter,
                rmse_threshold=search_settings.rmse_threshold,
                area_weight=config.assignment_area_weight,
                width_weight=config.assignment_width_weight,
                r1_weight=config.assignment_r1_weight,
            )
            logger.info("Hungarian completed: best RMSE = %.6f", opt_rmse)

            # Save assigned experiment to file
            save_experiment(
                experiment,
                file_name=os.path.join(
                    config.project_name,
                    f"assigned_experiment_{experiment.temperature:.2f}_K.csv",
                ),
                delimiter=delimiter,
                comment=(
                    f"# Optimal Assignment (Hungarian)\n"
                    f"# rmse = {opt_rmse:f}\n"
                    f"# T = {experiment.temperature:.2f} K"
                ),
            )

        # Fit susceptibility model to experimental chemical shifts.
        susc_model.fit_to(molecule, experiment, average_labels=average_labels)

        # Skip if fit fails
        if not susc_model.fit_status:
            continue

        # Update susceptibility tensor of Molecule using model
        molecule.susc = susc_model.tosusceptibility()

        # Calculate shifts using new susceptibility tensor
        molecule.calculate_shifts()
        molecule.average_shifts()

        with spec.context():
            plot_fitted_shifts(
                molecule,
                experiment,
                susc_model,
                spec=spec,
                show=options.runtime.show_plots,
                susc_units=options.susc_units,
                average=len(config.susc_fit_average_shifts),
                save=True,
                save_name=os.path.join(
                    config.project_name,
                    f"shifts_{experiment.temperature:.2f}_K",
                ),
                verbose=True,
                window_title=f"Fitted shifts at {experiment.temperature:.2f} K",
            )

        with spec.context():
            plot_shift_spread(
                molecule,
                experiment,
                spec=spec,
                terms=_terms,
                show=options.runtime.show_plots,
                save=True,
                save_name=os.path.join(
                    config.project_name,
                    f"shift_spread_{molecule.susc.temperature:.2f}_K",
                ),
                verbose=True,
                window_title=(
                    f"Spread of predicted shift components "
                    f"at {experiment.temperature:.2f} K"
                ),
                order="descending",
            )

        with spec.context():
            plot_shift_contrib(
                molecule,
                experiment,
                spec=spec,
                terms=_terms,
                show=options.runtime.show_plots,
                save=True,
                save_name=os.path.join(
                    config.project_name,
                    f"mean_components_{experiment.temperature:.2f}_K",
                ),
                verbose=True,
                window_title=(
                    f"Predicted shift components at {experiment.temperature:.2f} K"
                ),
                order="descending",
            )

        # r^-6 distance-model fits (R1 and linewidth)
        _width_fit_result = None
        _r1_fit_result = None
        for _obs in ("r1", "width"):
            _has_data = any(
                (sig.r1 is not None if _obs == "r1" else sig.width > 0.0)
                for sig in experiment.signals
            )
            if not _has_data:
                continue
            try:
                r6_result = fit_r6(molecule, experiment, observable=_obs)
            except ValueError as err:
                logger.warning("r^-6 fit (%s) skipped: %s", _obs, err)
                continue
            if _obs == "width":
                _width_fit_result = r6_result
            elif _obs == "r1":
                _r1_fit_result = r6_result
            with spec.context():
                plot_r6_fit(
                    r6_result,
                    observable=_obs,
                    spec=spec,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"r6_fit_{_obs}_{experiment.temperature:.2f}_K",
                    ),
                    verbose=True,
                    window_title=(
                        f"r\u207b\u2076 Fit ({_obs}) "
                        f"at {experiment.temperature:.2f} K"
                    ),
                )

        # Shift vs linewidth bubble plot
        with spec.context():
            plot_shift_width_bubble(
                experiment,
                molecule,
                spec=spec,
                observable="width",
                fit_result=_width_fit_result,
                show=options.runtime.show_plots,
                save=True,
                save_name=os.path.join(
                    config.project_name,
                    f"shift_width_bubble_{experiment.temperature:.2f}_K",
                ),
                verbose=True,
                window_title=(
                    f"Shift vs Linewidth at {experiment.temperature:.2f} K"
                ),
            )

        # Shift vs R1 bubble plot (only when R1 data is present)
        if _r1_fit_result is not None or any(
            sig.r1 is not None for sig in experiment.signals
        ):
            with spec.context():
                plot_shift_width_bubble(
                    experiment,
                    molecule,
                    spec=spec,
                    observable="r1",
                    fit_result=_r1_fit_result,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"shift_r1_bubble_{experiment.temperature:.2f}_K",
                    ),
                    verbose=True,
                    window_title=(
                        f"Shift vs R1 at {experiment.temperature:.2f} K"
                    ),
                )

        # Apply r^-6 predicted linewidths to molecule nuclei before spectrum plot
        if _width_fit_result is not None:
            _label_to_lw_loop = dict(
                zip(_width_fit_result["labels"], _width_fit_result["pred"])
            )
            _isotope_loop = molecule.nuclei[0].isotope
            _gamma_loop = NUCLEAR_GAMMAS[remove_numbers(_isotope_loop)]
            _b0_loop = experiment.magnetic_field
            for nuc in molecule.nuclei:
                if nuc.chem_label in _label_to_lw_loop:
                    lw_hz = _label_to_lw_loop[nuc.chem_label]
                    nuc.shift.lw = np.float64(lw_hz / (_gamma_loop * _b0_loop))

        # Predicted + experimental deconvoluted spectrum overlay
        with spec.context():
            plot_raw_deconv_pred(
                molecule=molecule,
                isotope=molecule.nuclei[0].isotope,
                shift_range=[
                    np.min([nuc.shift.avg for nuc in molecule.nuclei]),
                    np.max([nuc.shift.avg for nuc in molecule.nuclei]),
                ],
                experiment=experiment,
                spec=spec,
                save=True,
                show=options.runtime.show_plots,
                save_name=os.path.join(
                    config.project_name,
                    f"pred_and_exp_spectrum_{experiment.temperature:.2f}_K",
                ),
                verbose=True,
                window_title=(
                    f"Predicted and Experimental Spectra"
                    f" at {experiment.temperature:.2f} K"
                ),
            )

    # Write shift data to file
    _comment_base = f"Hyperfines from file {config.hyperfine_file}\n"
    if len(config.diamagnetic_file):
        _comment_base += f"Diamagnetic shifts from file {config.diamagnetic_file}\n"
    if len(config.diamagnetic_ref_file):
        _comment_base += (
            f"Diamagnetic reference from file {config.diamagnetic_ref_file}\n"
        )

    for molecule in molecules:
        comment = _comment_base + f"T = {molecule.susc.temperature:.2f} K"
        save_molecule_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                f"hyperfines_and_fitted_shifts_{molecule.susc.temperature:.2f}_K.csv",
            ),
            delimiter=delimiter,
            comment=comment,
            verbose=True,
        )

    # Write susceptibility tensor with model terms
    save_susc(
        molecules,
        os.path.join(config.project_name, "susceptibility_tensor.csv"),
        susc_models=susc_models,
        susc_units=options.susc_units,
    )

    if options.pcs_isosurface:
        for molecule in molecules:
            # Generate and save PCS isosurface
            molecule.susc.calc_irred()

            labels_arr = np.asarray(molecule.labels)
            coords_arr = np.asarray(molecule.coords, dtype=float)

            center_atom = molecule.labels[0]
            center_idx = np.where(labels_arr == center_atom)[0]
            if center_idx.size == 0:
                raise ValueError(f"Center atom {center_atom} not found in labels")

            coords_bohr = coords_arr * 1.88973
            coords_bohr = coords_bohr - coords_bohr[center_idx[0]]

            values, origin_bohr, step_bohr, grid_shape = compute_pcs_isosurface(
                chi_dtensor=molecule.susc.dtensor,
                labels=labels_arr,
                center_atom=center_atom,
                pdip_fn=Hyperfine.calc_pdip,
            )

            file_name = os.path.join(
                config.project_name,
                f"pcs_isosurf_{molecule.susc.temperature:.2f}_K.cube",
            )

            write_pcs_cube(
                file_name=file_name,
                comment=f"PCS Isosurface (T = {molecule.susc.temperature:.2f} K)",
                labels=labels_arr,
                coords_bohr=coords_bohr,
                origin_bohr=origin_bohr,
                step_bohr=step_bohr,
                grid_shape=grid_shape,
                values=values,
            )

            logger.info("PCS isosurface written to %s", file_name)

    mol = molecules[-1]

    shift_range = [
        np.min([nuc.shift.avg for nuc in mol.nuclei]),
        np.max([nuc.shift.avg for nuc in mol.nuclei]),
    ]

    extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]

    shift_range = [
        shift_range[0] + np.negative(np.max(extras)),
        shift_range[1] + np.positive(np.max(extras)),
    ]
    linewidth_output = resolve_output_linewidths(mol, shift_range)

    if spin is not None:
        temps_fit = np.array([mol.susc.temperature for mol in molecules], dtype=float)
        if temps_fit.size > 1:
            fit_vt(
                config=config,
                molecules=molecules,
                spin=spin,
                susc_models=susc_models,
                plot_profile=options.runtime.plot_profile,
                show_plots=options.runtime.show_plots,
            )

    # Apply r^-6 predicted linewidths to the spectrum if available.
    # Widths from fit_r6 are in Hz (same unit as signal.width); convert to
    # ppm using the Larmor frequency of the last experiment.
    if _width_fit_result is not None:
        _label_to_lw = dict(
            zip(_width_fit_result["labels"], _width_fit_result["pred"])
        )
        _isotope = mol.nuclei[0].isotope
        _gamma = NUCLEAR_GAMMAS[remove_numbers(_isotope)]
        _b0 = experiment.magnetic_field
        for nuc in mol.nuclei:
            if nuc.chem_label in _label_to_lw:
                lw_hz = _label_to_lw[nuc.chem_label]
                nuc.shift.lw = np.float64(lw_hz / (_gamma * _b0))

    with spec.context():
        plot_pred_spectrum(
            mol,
            isotope=mol.nuclei[0].isotope,
            shift_range=shift_range,
            spec=spec,
            effective_linewidths_by_label=linewidth_output.values_by_label,
            save=True,
            show=options.runtime.show_plots,
            save_name=os.path.join(
                config.project_name,
                f"pred_spectrum_{molecule.susc.temperature:.2f}_K",
            ),
        )

    return 0


def _obtain_r2a(
    molecule: Molecule,
    assignment: list[str],
    model: models.SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    echo_r2: bool,
):
    """
    Fit a susceptibility model for a proposed assignment and return RMSE.

    This helper is designed to be run in parallel when searching over assignment
    permutations.

    Args:
        molecule (Molecule): Molecule instance used for shift prediction.
        assignment (list[str]): Proposed assignment list (one per signal).
        model (models.SusceptibilityModel): Model instance to fit.
        experiment (Experiment): Experiment data to fit against.
        average_labels (list[list[str]]): Groups of labels to average during fitting.

    Returns:
        float: RMSE value for this assignment.
    """

    # and swap in new, permuted, assignments
    for it, new in enumerate(assignment):
        experiment.signals[it].assignment = new

    # Fit susceptibility model to experimental chemical shifts
    model.fit_to(molecule, experiment, average_labels=average_labels)

    # Print to screen if envvar enabled
    if echo_r2:
        print(model.rmse)

    return model.rmse
