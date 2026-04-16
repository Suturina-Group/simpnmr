# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit susceptibility tensors to experimental shift data.

Loads inputs, fits a selected susceptibility model, writes outputs and plots.
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
from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.domain.tensor import Hyperfine
from simpnmr.core.fitting import models
from simpnmr.core.util.strings import remove_numbers
from simpnmr.core.fitting.assign import (
    fit_with_hungarian_assignment,
    fit_with_hungarian_assignment_multi,
    generate_assignment_permutations,
)
from simpnmr.core.pcs.isosurf import compute_pcs_isosurface

# IO layer
from simpnmr.io.csv.fit import save_r6_fit
from simpnmr.io.csv.spec import read_spectrum
from simpnmr.io.csv.mol import save_molecule_to_csv
from simpnmr.io.csv.susc import save_susc
from simpnmr.io.cube.pcs_iso_write import write_pcs_cube
from simpnmr.io.xyz import xyz_write
from simpnmr.core.fitting.r6_fit import fit_r6
from simpnmr.viz.plots.fitted_shifts import plot_fitted_shifts
from simpnmr.viz.plots.r6_fit import (
    plot_r6_fit,
    plot_tau_space,
    plot_tau_space_combined,
    plot_tau_space_multitemp,
)
from simpnmr.viz.plots.spect import plot_raw_deconv_pred, plot_vt_spectra
from simpnmr.viz.plots.shift_width_bubble import plot_shift_width_bubble

# Visualisation
from simpnmr.viz.plots.shifts import plot_shift_contrib, plot_shift_spread
from simpnmr.viz.plots.spect import plot_pred_spectrum
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_fit_susc(config, options: FitSuscRunOptions | None = None) -> int:
    """Fit susceptibility tensor(s) defined by a YAML configuration file.

    The pipeline builds a Molecule from the requested hyperfine source, loads
    experimental data, fits the chosen susceptibility model, generates plots,
    and writes outputs into the project directory.

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
            al_to_cl, al_to_cml, al_to_isotope = load_chem_labels_from_csv(
                config.chem_labels_file
            )
            if has_missing_selected_chem_labels(base_molecule, al_to_cl):
                logger.warning(
                    "Chemical labels file does not define labels for all selected "
                    "nuclei; missing labels will use atom labels."
                )
            base_molecule.apply_chem_labels(al_to_cl, al_to_cml, al_to_isotope)
        except ValueError as err:
            raise ValueError(f"{err}\nCheck chem_labels and hyperfine files.")
        except KeyError as err:
            # treat missing labels/keys as a user input error
            raise ValueError(str(err))

        # Save xyz file with chemical labels for chemcraft
        xyz_write.save_chemcraft_xyz(
            file_name=os.path.join(
                config.project_name, "chemcraft_structure.xyz"
            ),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
            chem_labels={
                nuc.label: nuc.chem_label for nuc in base_molecule.nuclei
            },
        )

    # Save xyz file with chemical labels for chemcraft
    xyz_write.save_xyz(
        file_name=os.path.join(config.project_name, "structure.xyz"),
        labels=base_molecule.labels,
        coords=base_molecule.coords,
        comment=f"Structure from {config.hyperfine_file}",
    )

    # Pre-compute τ_R per temperature when hydrodynamic parameters are given
    _tau_r_by_temp: dict[float, float] = {}
    if config.fit_relaxation_tau_r_method is not None:
        from simpnmr.core.phys.tau_c import compute_tau_r

        _struct_xyz = os.path.join(config.project_name, "structure.xyz")
        _temps_seen: set[float] = set()
        for _exp in load_experiments(config.experiment_files):
            if _exp.temperature in _temps_seen:
                continue
            _temps_seen.add(_exp.temperature)
            try:
                _tau_r_result = compute_tau_r(
                    xyz_file=_struct_xyz,
                    temperature=_exp.temperature,
                    solvent=config.fit_relaxation_tau_r_solvent,
                    eta=config.fit_relaxation_tau_r_eta,
                    method=config.fit_relaxation_tau_r_method,
                    shell=config.fit_relaxation_tau_r_shell or 0.0,
                    sigma=config.fit_relaxation_tau_r_sigma or 0.6,
                )
                if _tau_r_result is not None:
                    _tau_r_by_temp[_exp.temperature] = _tau_r_result["tau_iso"]
                    logger.info(
                        "τ_R (%s) at %.1f K: %.1f ps",
                        config.fit_relaxation_tau_r_method,
                        _exp.temperature,
                        _tau_r_result["tau_iso"] * 1e12,
                    )
            except ValueError as err:
                logger.warning("τ_R calculation at %.1f K failed: %s", _exp.temperature, err)

    # Apply rotation matrix to all hyperfine tensors
    # if requested
    if len(config.hyperfine_rotate):
        _rot_a = np.loadtxt(config.hyperfine_rotate)
        base_molecule.rotate_hyperfines(_rot_a)

    # Load diamagnetic shift file
    if len(config.diamagnetic_file):
        dia_by_key, key_kind, ref_avg_by_isotope = load_diamagnetic_shifts(
            file_name=config.diamagnetic_file,
            file_type=config.diamagnetic_method,
            ref_file_name=config.diamagnetic_ref_file,
            ref_file_type=config.diamagnetic_ref_method,
            ref_values=getattr(config, "diamagnetic_ref_values", None),
        )
        base_molecule.apply_diamagnetic_shifts(
            dia_by_key=dia_by_key,
            key_kind=key_kind,
            ref_avg_by_isotope=ref_avg_by_isotope,
        )

    # Rotationally average hyperfines
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Create experiments
    experiments = load_experiments(config.experiment_files)

    # Attach raw spectrum and reference ppm if provided in config
    if config.experiment_spectrum_files:
        for experiment, spectrum_path in zip(
            experiments, config.experiment_spectrum_files
        ):
            experiment.spectrum = read_spectrum(spectrum_path)
            experiment.exp_reference = config.experiment_exp_reference

    # Check the number of experiments is consistent across the files
    # and issue warning if not
    if len(np.unique([len(exp.signals) for exp in experiments])) > 1:
        logger.warning("Some experiments have more signals than others!")

    # Create a molecule object to accompany each experiment object
    molecules = [copy.deepcopy(base_molecule) for _ in range(len(experiments))]

    name_to_susc_fit: dict[str, models.SusceptibilityModel] = {
        "full": models.FullSuscFitter,
        "split": models.SplitFitter,
        "isoaxrh": models.IsoAxRhFitter,
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
    _terms = ["pc", "fc", "d"]
    if config.hyperfine_method == "pdip":
        _terms.pop(_terms.index("fc"))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index("d"))

    # Accumulators for multi-temperature τ-space plots, keyed by isotope
    _r6_records_r1: dict[str, list[dict]] = {}
    _r6_records_width: dict[str, list[dict]] = {}

    # Multi-experiment Hungarian assignment (run once, before the main loop).
    # Finds one shared assignment across all (T, B) conditions by summing
    # cost matrices; each condition retains its own susceptibility parameters.
    _hungarian_done = False
    if config.assignment_method == "hungarian" and len(experiments) > 1:
        search_settings = resolve_assignment_search_settings(
            mode=config.assignment_search,
            n_attempts=config.assignment_n_attempts,
            max_iter=config.assignment_max_iter,
            rmse_threshold=config.assignment_rmse_threshold,
        )
        logger.info(
            "Multi-experiment Hungarian: %d conditions, "
            "mode=%s, n_attempts=%d, max_iter=%d, rmse_threshold=%.6f",
            len(experiments),
            search_settings.mode,
            search_settings.n_attempts,
            search_settings.max_iter,
            search_settings.rmse_threshold,
        )

        # Per-isotope partitions (signal indices from the first experiment)
        _isotopes_multi = list(
            dict.fromkeys(nuc.isotope for nuc in molecules[0].nuclei)
        )
        _multi_iso_partitions = None
        if len(_isotopes_multi) > 1:
            _multi_iso_partitions = []
            for _iso_m in _isotopes_multi:
                _iso_labels_m = {
                    n.chem_label for n in molecules[0].nuclei
                    if n.isotope == _iso_m
                }
                _iso_sig_idxs_m = [
                    i for i, s in enumerate(experiments[0].signals)
                    if (
                        s.isotope == _iso_m
                        if s.isotope is not None
                        else s.assignment in _iso_labels_m
                    )
                ]
                if _iso_sig_idxs_m:
                    _multi_iso_partitions.append(
                        (_iso_sig_idxs_m, _iso_labels_m)
                    )

        _multi_records = [
            {
                "molecule": mol,
                "susc_model": sm,
                "experiment": exp,
                "average_labels": average_labels,
            }
            for mol, sm, exp in zip(molecules, susc_models, experiments)
        ]
        _multi_rmse, _ = fit_with_hungarian_assignment_multi(
            records=_multi_records,
            n_attempts=search_settings.n_attempts,
            max_iter=search_settings.max_iter,
            rmse_threshold=search_settings.rmse_threshold,
            area_weight=config.assignment_area_weight,
            width_weight=config.assignment_width_weight,
            r1_weight=config.assignment_r1_weight,
            isotope_partitions=_multi_iso_partitions,
        )
        logger.info(
            "Multi-experiment Hungarian completed: mean RMSE = %.6f",
            _multi_rmse,
        )
        # Save each assigned experiment
        for exp in experiments:
            save_experiment(
                exp,
                file_name=os.path.join(
                    config.project_name,
                    f"assigned_experiment_{exp.temperature:.2f}_K.csv",
                ),
                delimiter=delimiter,
                comment=(
                    f"# Optimal Assignment (Hungarian, multi-experiment)\n"
                    f"# mean rmse = {_multi_rmse:f}\n"
                    f"# T = {exp.temperature:.2f} K"
                ),
            )
        _hungarian_done = True

    # Run fit for all experiments
    for molecule, susc_model, experiment in zip(
        molecules, susc_models, experiments
    ):
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
            # Build a view of the experiment containing only fittable
            # signals (those whose assignment matches a molecule label) so
            # that unmatched signals do not enter the permutation space.
            import copy as _copy
            _mol_labels_perm = {nuc.chem_label for nuc in molecule.nuclei}
            _perm_exp = _copy.copy(experiment)
            _perm_exp._signals = [
                s for s in experiment.signals
                if s.assignment in _mol_labels_perm
            ]
            permed_assignments = generate_assignment_permutations(
                experiment=_perm_exp, groups=config.assignment_groups
            )

            logger.info(
                "There are %s possible permutations", len(permed_assignments)
            )

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

            # Only update signals whose assignment matches a molecule label;
            # unmatched signals are frozen (their assignment is unchanged).
            _mol_labels = {nuc.chem_label for nuc in molecule.nuclei}
            _fit_sigs = [
                s for s in experiment.signals
                if s.assignment in _mol_labels
            ]
            for sig, new in zip(_fit_sigs, assignment):
                sig.assignment = new

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

        elif config.assignment_method == "hungarian" and not _hungarian_done:
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

            # Build per-isotope partitions so the tensor is fitted jointly
            # using all signals and nuclei, but each isotope's signals only
            # compete for labels belonging to that isotope.
            _isotopes_hung = list(
                dict.fromkeys(nuc.isotope for nuc in molecule.nuclei)
            )
            _iso_partitions = None
            if len(_isotopes_hung) > 1:
                _iso_partitions = []
                for _iso_hung in _isotopes_hung:
                    _iso_labels = {
                        n.chem_label for n in molecule.nuclei
                        if n.isotope == _iso_hung
                    }
                    _iso_sig_idxs = [
                        i for i, s in enumerate(experiment.signals)
                        if (
                            s.isotope == _iso_hung
                            if s.isotope is not None
                            else s.assignment in _iso_labels
                        )
                    ]
                    if _iso_sig_idxs:
                        _iso_partitions.append(
                            (_iso_sig_idxs, _iso_labels)
                        )
                        logger.info(
                            "Hungarian partition: isotope %s — "
                            "%d signals, %d labels",
                            _iso_hung,
                            len(_iso_sig_idxs),
                            len(_iso_labels),
                        )

            opt_rmse, _ = fit_with_hungarian_assignment(
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
                isotope_partitions=_iso_partitions,
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
                window_title=(
                    f"Fitted shifts at {experiment.temperature:.2f} K"
                ),
                spin=spin,
            )

        # Unique isotopes in molecule (insertion-ordered)
        _isotopes_mol = list(
            dict.fromkeys(nuc.isotope for nuc in molecule.nuclei)
        )

        for _iso in _isotopes_mol:
            _T = experiment.temperature
            with spec.context():
                plot_shift_spread(
                    molecule,
                    experiment,
                    spec=spec,
                    terms=_terms,
                    isotope_filter=_iso,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"shift_spread_{_iso}_{_T:.2f}_K",
                    ),
                    verbose=True,
                    window_title=(
                        f"Spread of predicted shift components "
                        f"({_iso}) at {_T:.2f} K"
                    ),
                    order="descending",
                )

            with spec.context():
                plot_shift_contrib(
                    molecule,
                    experiment,
                    spec=spec,
                    terms=_terms,
                    isotope_filter=_iso,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"mean_components_{_iso}_{_T:.2f}_K",
                    ),
                    verbose=True,
                    window_title=(
                        f"Predicted shift components ({_iso}) at {_T:.2f} K"
                    ),
                    order="descending",
                )

        # r^-6 distance-model fits (R1 and linewidth) — one fit per isotope
        _width_fit_result_by_iso: dict = {}
        _r1_fit_result_by_iso: dict = {}
        _relaxation_model = getattr(config, "relaxation_model", "sbm curie")
        _omega_S_r6 = -EGAMMA * experiment.magnetic_field * 2 * np.pi * 1e6

        for _iso_r6 in _isotopes_mol:
            _gamma_I_r6 = get_nuclear_gamma(_iso_r6) * 2 * np.pi * 1e6
            _omega_I_r6 = -_gamma_I_r6 * experiment.magnetic_field

            for _obs in ("r1", "width"):
                _has_data = any(
                    (
                        sig.r1 is not None
                        if _obs == "r1"
                        else sig.width > 0.0
                    )
                    for sig in experiment.signals
                )
                if not _has_data:
                    continue
                try:
                    r6_result = fit_r6(
                        molecule,
                        experiment,
                        observable=_obs,
                        tau_e=config.fit_relaxation_tau_e,
                        isotope_filter=_iso_r6,
                    )
                except ValueError as err:
                    logger.warning(
                        "r^-6 fit (%s, %s) skipped: %s",
                        _obs, _iso_r6, err,
                    )
                    continue

                if _obs == "width":
                    _width_fit_result_by_iso[_iso_r6] = r6_result
                elif _obs == "r1":
                    _r1_fit_result_by_iso[_iso_r6] = r6_result

                _r6_rec = {
                    "fit_result": r6_result,
                    "temperature": experiment.temperature,
                    "omega_I": _omega_I_r6,
                    "omega_S": _omega_S_r6,
                    "gamma_I": _gamma_I_r6,
                }
                if _obs == "r1":
                    _r6_records_r1.setdefault(_iso_r6, []).append(_r6_rec)
                else:
                    _r6_records_width.setdefault(_iso_r6, []).append(_r6_rec)

                save_r6_fit(
                    r6_result,
                    observable=_obs,
                    temperature=experiment.temperature,
                    magnetic_field=experiment.magnetic_field,
                    isotope=_iso_r6,
                    file_name=os.path.join(
                        config.project_name,
                        f"r6_fit_{_obs}_{_iso_r6}_"
                        f"{experiment.temperature:.2f}_K.csv",
                    ),
                    verbose=True,
                    tau_e=config.fit_relaxation_tau_e,
                )
                with spec.context():
                    plot_r6_fit(
                        r6_result,
                        observable=_obs,
                        spec=spec,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"r6_fit_{_obs}_{_iso_r6}_"
                            f"{experiment.temperature:.2f}_K",
                        ),
                        verbose=True,
                        window_title=(
                            f"r\u207b\u2076 Fit ({_obs}, {_iso_r6})"
                            f" at {experiment.temperature:.2f} K"
                        ),
                    )
                _tau_R_fixed = (
                    _tau_r_by_temp.get(experiment.temperature)
                    or config.fit_relaxation_tau_r_fixed
                )
                with spec.context():
                    plot_tau_space(
                        r6_result,
                        observable=_obs,
                        omega_I=_omega_I_r6,
                        omega_S=_omega_S_r6,
                        gamma_I=_gamma_I_r6,
                        spin=spin,
                        orbit=config.orbit,
                        total_momentum_J=config.total_momentum_J,
                        temperature=experiment.temperature,
                        relaxation_model=_relaxation_model,
                        spec=spec,
                        tau_e_range=config.fit_relaxation_tau_e_range,
                        tau_r_range=config.fit_relaxation_tau_r_range,
                        tau_R_fixed=_tau_R_fixed,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"r6_tau_space_{_obs}_{_iso_r6}_"
                            f"{experiment.temperature:.2f}_K",
                        ),
                        verbose=True,
                        window_title=(
                            f"\u03c4 space ({_obs}, {_iso_r6})"
                            f" at {experiment.temperature:.2f} K"
                        ),
                    )

            # Combined τ-space plot per isotope when both obs succeeded
            _r1_rec_iso = (
                _r6_records_r1.get(_iso_r6) or []
            )
            _lw_rec_iso = (
                _r6_records_width.get(_iso_r6) or []
            )
            _r1_iso = _r1_rec_iso[-1]["fit_result"] if _r1_rec_iso else None
            _lw_iso = _lw_rec_iso[-1]["fit_result"] if _lw_rec_iso else None
            if _r1_iso is not None and _lw_iso is not None:
                with spec.context():
                    plot_tau_space_combined(
                        r1_fit_result=_r1_iso,
                        width_fit_result=_lw_iso,
                        omega_I=_omega_I_r6,
                        omega_S=_omega_S_r6,
                        gamma_I=_gamma_I_r6,
                        spin=spin,
                        orbit=config.orbit,
                        total_momentum_J=config.total_momentum_J,
                        temperature=experiment.temperature,
                        relaxation_model=_relaxation_model,
                        spec=spec,
                        tau_e_range=config.fit_relaxation_tau_e_range,
                        tau_r_range=config.fit_relaxation_tau_r_range,
                        tau_R_fixed=_tau_R_fixed,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"r6_tau_space_combined_{_iso_r6}_"
                            f"{experiment.temperature:.2f}_K",
                        ),
                        verbose=True,
                        window_title=(
                            f"\u03c4 space (combined, {_iso_r6})"
                            f" at {experiment.temperature:.2f} K"
                        ),
                    )

        # Bubble plots — one per isotope
        _iso_suffix = len(_isotopes_mol) > 1
        for _iso_bubble in _isotopes_mol:
            _suffix = f"_{_iso_bubble}" if _iso_suffix else ""
            _iso_mol = copy.copy(molecule)
            _iso_mol.nuclei = [
                nuc for nuc in molecule.nuclei
                if nuc.isotope == _iso_bubble
            ]

            # Shift vs linewidth bubble plot
            with spec.context():
                plot_shift_width_bubble(
                    experiment,
                    _iso_mol,
                    spec=spec,
                    observable="width",
                    fit_result=_width_fit_result_by_iso.get(_iso_bubble),
                    isotope_filter=_iso_bubble,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"shift_width_bubble_{experiment.temperature:.2f}_K{_suffix}",
                    ),
                    verbose=True,
                    window_title=(
                        f"Shift vs Linewidth at {experiment.temperature:.2f} K"
                    ),
                )

            # Shift vs R1 bubble plot (only when R1 data is present)
            if _r1_fit_result_by_iso.get(_iso_bubble) is not None or any(
                sig.r1 is not None and sig.isotope == _iso_bubble
                for sig in experiment.signals
            ):
                with spec.context():
                    plot_shift_width_bubble(
                        experiment,
                        _iso_mol,
                        spec=spec,
                        observable="r1",
                        fit_result=_r1_fit_result_by_iso.get(_iso_bubble),
                        isotope_filter=_iso_bubble,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"shift_r1_bubble_{experiment.temperature:.2f}_K{_suffix}",
                        ),
                        verbose=True,
                        window_title=(
                            f"Shift vs R1 at {experiment.temperature:.2f} K"
                        ),
                    )

        # Apply r^-6 predicted linewidths to molecule nuclei before spectrum
        _b0_loop = experiment.magnetic_field
        for _iso_lw, _width_fit_result in _width_fit_result_by_iso.items():
            _label_to_lw_loop = dict(
                zip(_width_fit_result["labels"], _width_fit_result["pred"])
            )
            for nuc in molecule.nuclei:
                if nuc.isotope == _iso_lw and nuc.chem_label in _label_to_lw_loop:
                    lw_hz = _label_to_lw_loop[nuc.chem_label]
                    _gamma_loop = get_nuclear_gamma(nuc.isotope)
                    nuc.shift.lw = np.float64(lw_hz / (_gamma_loop * _b0_loop))

        # Predicted + experimental deconvoluted spectrum overlay — one per isotope
        _avgs = [
            nuc.shift.avg
            for nuc in molecule.nuclei
            if nuc.shift.avg is not None
        ]
        if _avgs and experiment.signals:
            _isotopes_present = list(
                dict.fromkeys(nuc.isotope for nuc in molecule.nuclei)
            )
            for _iso in _isotopes_present:
                _avgs_iso = [
                    nuc.shift.avg
                    for nuc in molecule.nuclei
                    if nuc.isotope == _iso and nuc.shift.avg is not None
                ]
                if not _avgs_iso:
                    continue
                with spec.context():
                    plot_raw_deconv_pred(
                        molecule=molecule,
                        isotope=_iso,
                        shift_range=[np.min(_avgs_iso), np.max(_avgs_iso)],
                        experiment=experiment,
                        spec=spec,
                        save=True,
                        show=options.runtime.show_plots,
                        save_name=os.path.join(
                            config.project_name,
                            f"pred_and_exp_spectrum_{_iso}_"
                            f"{experiment.temperature:.2f}_K",
                        ),
                        verbose=True,
                        window_title=(
                            f"Predicted and Experimental Spectra ({_iso})"
                            f" at {experiment.temperature:.2f} K"
                        ),
                    )

    # VT stacked experimental spectra — one figure per isotope
    if len(experiments) > 1 and any(
        e.spectrum is not None or e.signals for e in experiments
    ):
        _vt_isotopes = list(dict.fromkeys(
            s.isotope
            for e in experiments
            for s in e.signals
            if s.isotope is not None
        )) or list(dict.fromkeys(
            nuc.isotope for nuc in molecules[0].nuclei
        ))
        for _iso in _vt_isotopes:
            with spec.context():
                plot_vt_spectra(
                    experiments=experiments,
                    isotope=_iso,
                    spec=spec,
                    save=True,
                    show=options.runtime.show_plots,
                    save_name=os.path.join(
                        config.project_name,
                        f"vt_spectra_{_iso}",
                    ),
                    verbose=True,
                    window_title=f"VT Spectra ({_iso})",
                )

    # Write shift data to file
    _comment_base = f"Hyperfines from file {config.hyperfine_file}\n"
    if len(config.diamagnetic_file):
        _comment_base += (
            f"Diamagnetic shifts from file {config.diamagnetic_file}\n"
        )
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
                "hyperfines_and_fitted_shifts_"
                f"{molecule.susc.temperature:.2f}_K.csv",
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

            coords_bohr = coords_arr * 1.88973
            if molecule.paramagnetic_centre is not None:
                centre_bohr = (
                    np.asarray(molecule.paramagnetic_centre, dtype=float)
                    * 1.88973
                )
            else:
                centre_bohr = coords_bohr[0]

            (
                values, origin_bohr_rel, step_bohr, grid_shape
            ) = compute_pcs_isosurface(
                chi_dtensor=molecule.susc.dtensor,
                pdip_fn=Hyperfine.calc_pdip,
            )
            origin_bohr = tuple(
                float(centre_bohr[i]) + origin_bohr_rel[i] for i in range(3)
            )

            file_name = os.path.join(
                config.project_name,
                f"pcs_isosurf_{molecule.susc.temperature:.2f}_K.cube",
            )

            write_pcs_cube(
                file_name=file_name,
                comment=(
                    f"PCS Isosurface (T = {molecule.susc.temperature:.2f} K)"
                ),
                labels=labels_arr,
                coords_bohr=coords_bohr,
                origin_bohr=origin_bohr,
                step_bohr=step_bohr,
                grid_shape=grid_shape,
                values=values,
            )

            logger.info("PCS isosurface written to %s", file_name)

    # Multi-temperature τ-space plots (only when >1 experiment) — per isotope
    _relaxation_model_mt = getattr(config, "relaxation_model", "sbm curie")
    for _obs_mt, _records_by_iso in (
        ("r1", _r6_records_r1),
        ("width", _r6_records_width),
    ):
        for _iso_mt, _records_mt in _records_by_iso.items():
            if len(_records_mt) > 1:
                with spec.context():
                    plot_tau_space_multitemp(
                        records=_records_mt,
                        observable=_obs_mt,
                        spin=spin,
                        orbit=config.orbit,
                        total_momentum_J=config.total_momentum_J,
                        relaxation_model=_relaxation_model_mt,
                        spec=spec,
                        tau_e_range=config.fit_relaxation_tau_e_range,
                        tau_r_range=config.fit_relaxation_tau_r_range,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"r6_tau_space_multitemp_{_obs_mt}_{_iso_mt}",
                        ),
                        verbose=True,
                        window_title=(
                            f"\u03c4 space multi-T ({_obs_mt}, {_iso_mt})"
                        ),
                    )

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
        temps_fit = np.array(
            [mol.susc.temperature for mol in molecules], dtype=float
        )
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
        _b0 = experiment.magnetic_field
        for nuc in mol.nuclei:
            if nuc.chem_label in _label_to_lw:
                lw_hz = _label_to_lw[nuc.chem_label]
                _gamma = get_nuclear_gamma(nuc.isotope)
                nuc.shift.lw = np.float64(lw_hz / (_gamma * _b0))

    _isotopes_final = list(
        dict.fromkeys(nuc.isotope for nuc in mol.nuclei)
    )
    for _iso_final in _isotopes_final:
        _avgs_iso_final = [
            nuc.shift.avg
            for nuc in mol.nuclei
            if nuc.isotope == _iso_final and nuc.shift.avg is not None
        ]
        if not _avgs_iso_final:
            continue
        with spec.context():
            plot_pred_spectrum(
                mol,
                isotope=_iso_final,
                shift_range=[
                    np.min(_avgs_iso_final),
                    np.max(_avgs_iso_final),
                ],
                spec=spec,
                effective_linewidths_by_label=linewidth_output.values_by_label,
                save=True,
                show=options.runtime.show_plots,
                save_name=os.path.join(
                    config.project_name,
                    f"pred_spectrum_{_iso_final}_"
                    f"{mol.susc.temperature:.2f}_K",
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

    This helper is designed to be run in parallel when searching over
    assignment permutations.

    Args:
        molecule (Molecule): Molecule instance used for shift prediction.
        assignment (list[str]): Proposed assignment list (one per signal).
        model (models.SusceptibilityModel): Model instance to fit.
        experiment (Experiment): Experiment data to fit against.
        average_labels (list[list[str]]): Groups of labels to average during
            fitting.

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
