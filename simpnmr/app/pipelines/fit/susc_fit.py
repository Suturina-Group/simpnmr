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
from simpnmr.app.loaders.exp_load import load_experiments, save_experiments
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
from simpnmr.app.policies.relax import apply_relaxation_decomposition
from simpnmr.app.policies.susc import resolve_susc_fit_variables

# Core / domain
from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.domain.tensor import Hyperfine
from simpnmr.core.fitting import models
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
from simpnmr.io.csv.peaks import save_peak_data_to_csv
from simpnmr.io.csv.susc import save_susc
from simpnmr.io.cube.pcs_iso_write import write_pcs_cube
from simpnmr.io.xyz import xyz_write
from simpnmr.core.fitting.r6_fit import fit_r6, solve_tau_e_from_p1
from simpnmr.viz.plots.fitted_shifts import plot_fitted_shifts
from simpnmr.viz.plots.r6_fit import (
    plot_r6_fit,
    plot_tau_space,
    plot_tau_space_combined,
    plot_tau_space_multitemp,
)
from simpnmr.viz.plots.spect import plot_raw_deconv_pred, plot_vt_spectra
from simpnmr.viz.plots.shift_width_bubble import plot_shift_width_bubble
from simpnmr.viz.plots.param_covariance import plot_param_covariance

# Visualisation
from simpnmr.viz.plots.shifts import plot_shift_contrib, plot_shift_spread
from simpnmr.viz.plots.spect import plot_pred_spectrum
from simpnmr.viz.style.theme import apply_profile

try:
    from simpnmr.gui.molecule_view import (
        assign_label_colors, _CPK_COLORS, parse_xyz,
    )
    _HAS_VIEWER = True
except ImportError:
    _HAS_VIEWER = False

logger = logging.getLogger(__name__)


def _colors_from_xyz(xyz_path: str) -> "dict[str, str] | None":
    """Return the exact chem-label → color mapping the GUI viewer computes.

    Reads `xyz_path` with the same `parse_xyz` → `assign_label_colors`
    pipeline the GUI uses, so graph colors are guaranteed to match.
    Returns None when the viewer modules are unavailable or the file is missing.
    """
    if not _HAS_VIEWER:
        return None
    try:
        mol = parse_xyz(xyz_path)
    except (FileNotFoundError, ValueError):
        return None
    groups = mol.grouped_by_label()
    if not groups:
        return None
    unlabelled_elems = {a.element for a in mol.atoms if not a.label}
    reserved = {_CPK_COLORS[e] for e in unlabelled_elems if e in _CPK_COLORS}
    return assign_label_colors(list(groups.keys()), reserved_colors=reserved)


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

    # Compute label → color once by re-running the viewer's exact parse_xyz
    # pipeline on the chemcraft XYZ we just wrote, so graph colors always
    # match the GUI molecular viewer.
    _xyz_path = os.path.join(config.project_name, "chemcraft_structure.xyz")
    _mol_label_colors_all: dict[str, str] | None = _colors_from_xyz(_xyz_path)

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
                logger.warning(
                    "τ_R calculation at %.1f K failed: %s",
                    _exp.temperature, err,
                )

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
        "split": models.SplitFitter,
        "isoaxrh": models.IsoAxRhFitter,
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
            total_J=base_molecule.electronic.total_J,
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
    _assignment_run = False
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
            [iso for iso in dict.fromkeys(nuc.isotope for nuc in molecules[0].nuclei) if iso is not None]
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
        _hungarian_done = True
        _assignment_run = True

    # Shared permute assignment (run once, before the main loop).
    # Finds one permutation that minimises the sum of RMSEs across all
    # temperatures; each condition is then fitted independently with that
    # fixed assignment.
    _permute_done = False
    if (
        config.assignment_method == "permute"
        and getattr(config, "assignment_shared", False)
        and len(experiments) > 1
    ):
        _ref_mol = molecules[0]
        if not len(config.assignment_groups):
            config.assignment_groups = [
                list({nuc.chem_label for nuc in _ref_mol.nuclei})
            ]
        _mol_labels_shared = {nuc.chem_label for nuc in _ref_mol.nuclei}
        _perm_exp_ref = copy.deepcopy(experiments[0])
        _perm_exp_ref._signals = [
            s for s in experiments[0].signals
            if s.assignment in _mol_labels_shared
        ]
        _correlations_shared = getattr(config, "assignment_correlations", [])
        _sig_allowed_shared = (
            _signal_allowed_from_correlations(
                config.assignment_groups, _correlations_shared,
                _ref_mol, experiments[0],
            )
            if _correlations_shared else None
        )
        permed_assignments_shared = generate_assignment_permutations(
            experiment=_perm_exp_ref,
            groups=config.assignment_groups,
            signal_allowed=_sig_allowed_shared,
            correlations=_correlations_shared,
        )
        if not permed_assignments_shared:
            logger.warning(
                "No valid permutations after applying correlation constraints "
                "— skipping shared permute assignment"
            )
            _permute_done = False
        logger.info(
            "Shared permute: %d permutations × %d temperatures",
            len(permed_assignments_shared), len(experiments),
        )
        num_threads = (
            mp.cpu_count() - 1
            if config.num_threads == "auto"
            else config.num_threads
        )
        num_threads = min(num_threads, len(permed_assignments_shared))
        # Ship the shared molecule/experiment/model lists once per worker via
        # the initializer; only the permutation is sent per task.
        pool = mp.Pool(
            num_threads,
            initializer=_perm_worker_init,
            initargs=(
                {
                    "molecules": molecules,
                    "susc_models": susc_models,
                    "experiments": experiments,
                    "average_labels": average_labels,
                    "echo_r2": options.runtime.echo_r2,
                },
            ),
        )
        results_shared = pool.map(
            _obtain_r2a_multi_perm, permed_assignments_shared
        )
        pool.close()
        pool.join()
        best_perm = permed_assignments_shared[np.nanargmin(results_shared)]
        opt_rmse_shared = np.nanmin(results_shared)
        logger.info(
            "Shared permute completed: best total RMSE = %.6f", opt_rmse_shared
        )
        # Apply the best assignment to all experiments
        for experiment in experiments:
            _mol_labels = {nuc.chem_label for nuc in molecules[0].nuclei}
            _fit_sigs = [
                s for s in experiment.signals
                if s.assignment in _mol_labels
            ]
            for sig, new in zip(_fit_sigs, best_perm):
                sig.assignment = new
        _permute_done = True
        _assignment_run = True

    # Run fit for all experiments
    for molecule, susc_model, experiment in zip(
        molecules, susc_models, experiments
    ):
        # If permuting assignments, then first
        # run all assignment permutations to find best one
        if config.assignment_method == "permute" and not _permute_done:
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
            _correlations = getattr(config, "assignment_correlations", [])
            _sig_allowed = (
                _signal_allowed_from_correlations(
                    config.assignment_groups, _correlations, molecule, experiment
                )
                if _correlations else None
            )
            permed_assignments = generate_assignment_permutations(
                experiment=_perm_exp,
                groups=config.assignment_groups,
                signal_allowed=_sig_allowed,
                correlations=_correlations,
            )

            logger.info(
                "There are %s valid permutations", len(permed_assignments)
            )
            if not permed_assignments:
                logger.warning(
                    "No valid permutations after applying correlation constraints "
                    "— skipping permute assignment for this experiment"
                )
                continue

            # For each permutation, fit tensor and store r2_adjusted

            # Number of threads
            if config.num_threads == "auto":
                num_threads = mp.cpu_count() - 1
            else:
                num_threads = config.num_threads

            if num_threads > len(permed_assignments):
                num_threads = len(permed_assignments)

            # Create parallel pool. Heavy objects are shipped once per worker
            # via the initializer; only the lightweight assignment list is sent
            # per task, so no large list of deep copies is materialised here.
            pool = mp.Pool(
                num_threads,
                initializer=_perm_worker_init,
                initargs=(
                    {
                        "molecule": molecule,
                        "susc_model": susc_model,
                        "experiment": experiment,
                        "average_labels": average_labels,
                        "echo_r2": options.runtime.echo_r2,
                    },
                ),
            )
            logger.info(
                "Parallel permutation search: %s worker processes",
                num_threads,
            )

            # Calculate each assignment's r2 in parallel (order preserved)
            results = pool.map(_obtain_r2a_perm, permed_assignments)

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

            _assignment_run = True

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
                [iso for iso in dict.fromkeys(nuc.isotope for nuc in molecule.nuclei) if iso is not None]
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
            _assignment_run = True

        # Fit susceptibility model to experimental chemical shifts.
        susc_model.fit_to(molecule, experiment, average_labels=average_labels)

        # Skip if fit fails
        if not susc_model.fit_status:
            continue

        # Update susceptibility tensor of Molecule using model. tosusceptibility
        # records the fitted isotropic value as the g-corrected channel
        # (chi.iso_g_corr); calculate_shifts then adds the spin-only channel and
        # splits the Fermi contact into spin-only and g-correction parts.
        molecule.susc = susc_model.tosusceptibility()

        # Calculate shifts using new susceptibility tensor
        molecule.calculate_shifts()
        molecule.average_shifts()

        _mol_label_colors_fit = _mol_label_colors_all

        if config.susc_fit_figure("fitted_shifts"):
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
                    variant=config.susc_fit_shifts_format,
                    width_scale=config.susc_fit_shifts_width_scale,
                    show_point_labels=config.susc_fit_shifts_labels,
                    window_title=(
                        f"Fitted shifts at {experiment.temperature:.2f} K"
                    ),
                    spin=spin,
                    orbit=base_molecule.electronic.orbit_L,
                    total_J=base_molecule.electronic.total_J,
                    label_colors=_mol_label_colors_fit,
                )

        # Parameter covariance contour plot (optional)
        _cov_params = getattr(config, "susc_fit_covariance_params", [])
        if len(_cov_params) == 2:
            logger.info(
                "Computing parameter covariance grid (%s vs %s) — this may take a minute",
                _cov_params[0], _cov_params[1],
            )
            try:
                with spec.context():
                    plot_param_covariance(
                        molecule,
                        experiment,
                        susc_model,
                        param_x=_cov_params[0],
                        param_y=_cov_params[1],
                        spec=spec,
                        average_labels=average_labels,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"covariance_{_cov_params[0]}_{_cov_params[1]}"
                            f"_{experiment.temperature:.2f}_K",
                        ),
                        window_title=(
                            f"Covariance {_cov_params[0]} vs {_cov_params[1]}"
                            f" at {experiment.temperature:.2f} K"
                        ),
                    )
            except Exception as _cov_exc:
                logger.warning(
                    "Covariance plot skipped: %s", _cov_exc
                )

        # Unique NMR isotopes in molecule (insertion-ordered, skip None).
        _isotopes_mol = [
            iso for iso in dict.fromkeys(nuc.isotope for nuc in molecule.nuclei)
            if iso is not None
        ]

        if config.susc_fit_figure("shift_components"):
            for _iso in _isotopes_mol:
                _T = experiment.temperature
                _iso_chem_labels_sc = {
                    nuc.chem_label
                    for nuc in molecule.nuclei
                    if nuc.isotope == _iso
                }
                _label_colors_sc = (
                    {
                        lbl: c for lbl, c in _mol_label_colors_fit.items()
                        if lbl in _iso_chem_labels_sc
                    }
                    if _mol_label_colors_fit else None
                )
                with spec.context():
                    plot_shift_spread(
                        molecule,
                        experiment,
                        spec=spec,
                        terms=_terms,
                        isotope_filter=_iso,
                        label_colors=_label_colors_sc,
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
                        label_colors=_label_colors_sc,
                        show=options.runtime.show_plots,
                        save=True,
                        save_name=os.path.join(
                            config.project_name,
                            f"mean_components_{_iso}_{_T:.2f}_K",
                        ),
                        verbose=True,
                        variant=config.susc_fit_shifts_format,
                        width_scale=config.susc_fit_shifts_width_scale,
                        window_title=(
                            f"Predicted shift components"
                            f" ({_iso}) at {_T:.2f} K"
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
                        distance_power=getattr(
                            config, "fit_relaxation_distance_power", 0.0
                        ) or 0.0,
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
                if config.susc_fit_figure("r6_fit"):
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
                if config.susc_fit_figure("tau_space"):
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
                if config.susc_fit_figure("tau_space"):
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

            # Shift vs linewidth bubble plot (only when r⁻⁶ width fit ran)
            if (
                config.susc_fit_figure("bubble_plots")
                and _width_fit_result_by_iso.get(_iso_bubble) is not None
            ):
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
                            "shift_width_bubble_"
                            f"{experiment.temperature:.2f}_K{_suffix}",
                        ),
                        verbose=True,
                        window_title=(
                            f"Shift vs Linewidth at"
                            f" {experiment.temperature:.2f} K"
                        ),
                    )

            # Shift vs R1 bubble plot (only when R1 data is present)
            if config.susc_fit_figure("bubble_plots") and (
                _r1_fit_result_by_iso.get(_iso_bubble) is not None or any(
                    sig.r1 is not None and sig.isotope == _iso_bubble
                    for sig in experiment.signals
                )
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
                            "shift_r1_bubble_"
                            f"{experiment.temperature:.2f}_K{_suffix}",
                        ),
                        verbose=True,
                        window_title=(
                            f"Shift vs R1 at {experiment.temperature:.2f} K"
                        ),
                    )

        # Derive a single molecular τ_e from the most trustworthy r⁻⁶
        # relaxation fit (across isotopes and observables), then compute the
        # full R1/R2 decomposition for the whole molecule. τ_R and τ_e are
        # molecular (isotope-independent); only the *source* of τ_e is chosen
        # by fit trustworthiness (number of points + p1 relative uncertainty).
        _tau_R_relax = (
            _tau_r_by_temp.get(experiment.temperature)
            or config.fit_relaxation_tau_r_fixed
        )
        if _tau_R_relax:
            _tau_e_ref = config.fit_relaxation_tau_e or 1e-12
            _tau_e_candidates = []  # (score, tau_e, iso, obs, n, rel_err)
            for _iso_c in _isotopes_mol:
                _gamma_c = get_nuclear_gamma(_iso_c) * 2 * np.pi * 1e6
                _omega_I_c = -_gamma_c * experiment.magnetic_field
                for _obs_c, _by_iso in (
                    ("width", _width_fit_result_by_iso),
                    ("r1", _r1_fit_result_by_iso),
                ):
                    _res_c = _by_iso.get(_iso_c)
                    if not _res_c:
                        continue
                    _te = solve_tau_e_from_p1(
                        _res_c["p1"], _tau_R_relax,
                        omega_I=_omega_I_c, omega_S=_omega_S_r6,
                        gamma_I=_gamma_c, spin=spin, orbit=config.orbit,
                        total_momentum_J=config.total_momentum_J,
                        temperature=experiment.temperature,
                        observable=_obs_c,
                        relaxation_model=_relaxation_model,
                        tau_e_range=config.fit_relaxation_tau_e_range,
                        tau_e_ref=_tau_e_ref,
                    )
                    if _te is None:
                        continue
                    _n_c = len(_res_c.get("labels", []))
                    _p1_c = abs(_res_c.get("p1", 0.0))
                    _p1e_c = _res_c.get("p1_err")
                    _rel_c = (
                        abs(_p1e_c) / _p1_c
                        if (_p1_c > 0 and _p1e_c is not None)
                        else float("inf")
                    )
                    _score_c = _n_c / max(_rel_c, 1e-6)
                    _tau_e_candidates.append(
                        (_score_c, _te, _iso_c, _obs_c, _n_c, _rel_c)
                    )

            if _tau_e_candidates:
                _tau_e_candidates.sort(key=lambda c: c[0], reverse=True)
                _best_c = _tau_e_candidates[0]
                _tau_e_chosen = _best_c[1]
                logger.info(
                    "Relaxation decomposition: τ_R = %.1f ps, τ_e = %.3f ps "
                    "(from %s %s fit: %d points, p1 rel. unc. %.1f%%)",
                    _tau_R_relax * 1e12, _tau_e_chosen * 1e12,
                    _best_c[2], _best_c[3], _best_c[4],
                    _best_c[5] * 100 if np.isfinite(_best_c[5]) else float("nan"),
                )
                for _alt in _tau_e_candidates[1:]:
                    logger.info(
                        "  alt τ_e = %.3f ps (from %s %s: %d points, "
                        "rel. unc. %.1f%%)",
                        _alt[1] * 1e12, _alt[2], _alt[3], _alt[4],
                        _alt[5] * 100 if np.isfinite(_alt[5]) else float("nan"),
                    )
                try:
                    apply_relaxation_decomposition(
                        molecule,
                        relaxation_model=_relaxation_model,
                        temperature=experiment.temperature,
                        magnetic_field_tesla=experiment.magnetic_field,
                        tau_R=_tau_R_relax,
                        tau_e1=_tau_e_chosen,
                        tau_e2=_tau_e_chosen,
                        hyperfine_method=getattr(
                            config, "hyperfine_method", None
                        ),
                        min_linewidth_hz=getattr(
                            config, "relaxation_min_linewidth_hz", 0.0
                        ) or 0.0,
                    )
                except Exception as _relax_err:
                    logger.warning(
                        "Relaxation decomposition skipped: %s", _relax_err
                    )

        # Fall back to the raw r⁻⁶-fitted linewidths only when no relaxation
        # decomposition was computed above.
        if getattr(molecule, "relaxation", None) is None:
            # Apply r^-6 predicted linewidths to molecule nuclei before spectrum
            _b0_loop = experiment.magnetic_field
            for _iso_lw, _width_fit_result in _width_fit_result_by_iso.items():
                _label_to_lw_loop = dict(
                    zip(_width_fit_result["labels"], _width_fit_result["pred"])
                )
                for nuc in molecule.nuclei:
                    if (
                        nuc.isotope == _iso_lw
                        and nuc.chem_label in _label_to_lw_loop
                    ):
                        lw_hz = _label_to_lw_loop[nuc.chem_label]
                        _gamma_loop = get_nuclear_gamma(nuc.isotope)
                        nuc.shift.lw = np.float64(
                            lw_hz / (_gamma_loop * _b0_loop)
                        )

        # Predicted + experimental spectrum overlay — one per isotope
        _avgs = [
            nuc.shift.avg
            for nuc in molecule.nuclei
            if nuc.shift.avg is not None
        ]
        if _avgs and experiment.signals and config.susc_fit_figure("spectra"):
            _mol_label_colors = _mol_label_colors_all

            _isotopes_present = list(
                [iso for iso in dict.fromkeys(nuc.isotope for nuc in molecule.nuclei) if iso is not None]
            )
            for _iso in _isotopes_present:
                _avgs_iso = [
                    nuc.shift.avg
                    for nuc in molecule.nuclei
                    if nuc.isotope == _iso and nuc.shift.avg is not None
                ]
                if not _avgs_iso:
                    continue
                _iso_chem_labels = {
                    nuc.chem_label
                    for nuc in molecule.nuclei
                    if nuc.isotope == _iso
                }
                _label_colors = (
                    {
                        lbl: c for lbl, c in _mol_label_colors.items()
                        if lbl in _iso_chem_labels
                    }
                    if _mol_label_colors else None
                )
                _iso_shift_range = [np.min(_avgs_iso), np.max(_avgs_iso)]
                _iso_lw_output = resolve_output_linewidths(
                    molecule, _iso_shift_range
                )
                with spec.context():
                    plot_raw_deconv_pred(
                        molecule=molecule,
                        isotope=_iso,
                        shift_range=_iso_shift_range,
                        experiment=experiment,
                        spec=spec,
                        effective_linewidths_by_label=_iso_lw_output.values_by_label,
                        save=True,
                        show=options.runtime.show_plots,
                        save_name=os.path.join(
                            config.project_name,
                            f"pred_and_exp_spectrum_{_iso}_"
                            f"{experiment.temperature:.2f}_K",
                        ),
                        verbose=True,
                        window_title=(
                            f"Predicted and Experimental Spectra"
                            f" ({_iso}) at"
                            f" {experiment.temperature:.2f} K"
                        ),
                        label_colors=_label_colors,
                        axis_break=config.susc_fit_spectra_break,
                    )

    # VT stacked experimental spectra — one figure per isotope
    if config.susc_fit_figure("spectra") and len(experiments) > 1 and any(
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

    # Write assigned experiments in one wide-format file (preserves isotope
    # column and multi-temperature/field layout matching the input format).
    if _assignment_run:
        save_experiments(
            experiments,
            file_name=os.path.join(
                config.project_name, "assigned_experiment.csv"
            ),
            delimiter=delimiter,
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

    for molecule, experiment in zip(molecules, experiments):
        _T = molecule.susc.temperature
        comment = _comment_base + f"T = {_T:.2f} K"
        save_molecule_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                "hyperfines_and_fitted_shifts_"
                f"{_T:.2f}_K.csv",
            ),
            delimiter=delimiter,
            comment=comment,
            verbose=True,
        )

        # Peak data file — same format as the predict workflow's
        # peak_data_*.csv, built from the fitted shifts and output linewidths.
        # The magnetic field is encoded in the name because the linewidths
        # (relaxation) are field-dependent.
        _peak_avgs = [
            nuc.shift.avg for nuc in molecule.nuclei
            if nuc.shift.avg is not None
        ]
        if _peak_avgs:
            # Guard the final peak-data write so a late failure never discards
            # an otherwise-complete fit.
            try:
                _peak_range = [
                    float(np.min(_peak_avgs)), float(np.max(_peak_avgs))
                ]
                _peak_lw = resolve_output_linewidths(molecule, _peak_range)
                _b0 = getattr(experiment, "magnetic_field", None)
                _field_tag = f"_{_b0:.2f}_T" if _b0 is not None else ""
                _peak_comment = f"T = {_T:.2f} K"
                if _b0 is not None:
                    _peak_comment += f", B0 = {_b0:.2f} T"
                save_peak_data_to_csv(
                    molecule=molecule,
                    file_name=os.path.join(
                        config.project_name,
                        f"peak_data_{_T:.2f}_K{_field_tag}.csv",
                    ),
                    linewidth_by_label=_peak_lw.values_by_label,
                    linewidth_column_name=_peak_lw.column_name,
                    comment=_peak_comment,
                    verbose=True,
                )
            except Exception as _peak_err:
                logger.warning(
                    "peak_data file not written for %.2f K: %s",
                    _T, _peak_err,
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

    # Multi-temperature τ-space plots (only when >1 experiment), per isotope
    _relaxation_model_mt = getattr(config, "relaxation_model", "sbm curie")
    for _obs_mt, _records_by_iso in (
        ("r1", _r6_records_r1),
        ("width", _r6_records_width),
    ):
        for _iso_mt, _records_mt in _records_by_iso.items():
            if len(_records_mt) > 1 and config.susc_fit_figure("tau_space"):
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
                save_chi_t=config.susc_fit_figure("chi_t"),
            )

    # Apply r^-6 predicted linewidths to the spectrum if available.
    # Widths from fit_r6 are in Hz (same unit as signal.width); convert to
    # ppm using the Larmor frequency of the last experiment.
    _b0 = experiment.magnetic_field
    for _iso_lw_final, _width_fit_result in _width_fit_result_by_iso.items():
        _label_to_lw = dict(
            zip(_width_fit_result["labels"], _width_fit_result["pred"])
        )
        for nuc in mol.nuclei:
            if nuc.isotope == _iso_lw_final and nuc.chem_label in _label_to_lw:
                lw_hz = _label_to_lw[nuc.chem_label]
                _gamma = get_nuclear_gamma(nuc.isotope)
                nuc.shift.lw = np.float64(lw_hz / (_gamma * _b0))

    _isotopes_final = list(
        [iso for iso in dict.fromkeys(nuc.isotope for nuc in mol.nuclei) if iso is not None]
    )
    for _iso_final in _isotopes_final:
        _avgs_iso_final = [
            nuc.shift.avg
            for nuc in mol.nuclei
            if nuc.isotope == _iso_final and nuc.shift.avg is not None
        ]
        if not _avgs_iso_final:
            continue
        if not config.susc_fit_figure("spectra"):
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


_HSQC_CUTOFF_ANG = 1.7   # direct H–C bond
_HMBC_CUTOFF_ANG = 4.5   # 2–3 bond H–C through-bond distance


def _are_connected(
    h_chem_label: str,
    c_chem_label: str,
    molecule,
    corr_type: str,
    cutoff: float | None = None,
) -> bool:
    """Return True if any nucleus with h_chem_label is within cutoff Å of any
    nucleus with c_chem_label.

    Default cutoffs: 1.7 Å for HSQC (1-bond H–C), 4.5 Å for HMBC (2–3 bond).
    """
    if cutoff is None:
        cutoff = _HSQC_CUTOFF_ANG if corr_type == "hsqc" else _HMBC_CUTOFF_ANG
    h_coords = [
        np.asarray(nuc.coord, dtype=float)
        for nuc in molecule.nuclei
        if nuc.chem_label == h_chem_label
    ]
    c_coords = [
        np.asarray(nuc.coord, dtype=float)
        for nuc in molecule.nuclei
        if nuc.chem_label == c_chem_label
    ]
    if not h_coords or not c_coords:
        return True  # unknown label — don't filter
    for h in h_coords:
        for c in c_coords:
            if float(np.linalg.norm(h - c)) <= cutoff:
                return True
    return False


def _signal_allowed_from_correlations(
    groups: list[list[str]],
    correlations: list[dict],
    molecule,
    experiment,
) -> dict[str, set[str]]:
    """Precompute allowed chem_labels per signal from HMBC/HSQC constraints.

    Returns a dict ``{signal_label: set_of_allowed_chem_labels}`` for every
    signal that appears in at least one correlation constraint.  Signals with
    no constraints are absent (meaning all group labels are allowed).
    """
    # C signal label → chem_label (fixed, not permuted)
    c_label_map: dict[str, str] = {
        s.assignment: s.assignment for s in experiment.signals
    }
    # Build set of chem_labels per group for fast lookup
    label_to_group: dict[str, list[str]] = {}
    for group in groups:
        for lbl in group:
            label_to_group[lbl] = group

    # For each constrained H signal, intersect allowed sets across constraints
    signal_allowed: dict[str, set[str]] = {}
    for corr in correlations:
        h_exp = corr["h"]
        c_exp = corr["c"]
        c_chem = c_label_map.get(c_exp, c_exp)
        corr_type = corr.get("type", "hsqc")
        cutoff = corr.get("cutoff")
        group = label_to_group.get(h_exp)
        if group is None:
            continue  # signal not in any permuted group
        # Which chem_labels in this group are valid for h_exp given this constraint?
        valid = {
            lbl for lbl in group
            if _are_connected(lbl, c_chem, molecule, corr_type, cutoff)
        }
        if h_exp in signal_allowed:
            signal_allowed[h_exp] &= valid   # intersect multiple constraints
        else:
            signal_allowed[h_exp] = valid
    return signal_allowed


def _filter_by_correlations(
    permed_assignments: list,
    fittable_signals: list,
    correlations: list[dict],
    molecule,
    experiment,
) -> list:
    """Remove permutations inconsistent with HMBC/HSQC correlation constraints.

    For each correlation ``{h, c, type}``:
    - ``h``: current assignment label of the experimental H signal being permuted.
    - ``c``: current assignment label of the experimental C signal (fixed, not permuted).

    A permutation is kept only if, for every constraint, the proposed H chem_label
    is within the expected distance of the C chem_label still assigned to the C signal.
    """
    if not correlations:
        return permed_assignments

    # Map current H signal assignment → index in fittable_signals list
    sig_to_idx: dict[str, int] = {
        s.assignment: i for i, s in enumerate(fittable_signals)
    }
    # Map C signal assignment label → its current chem_label (unchanged)
    c_label_map: dict[str, str] = {
        s.assignment: s.assignment
        for s in experiment.signals
        if s.isotope is None or s.isotope not in ("1H",)
    }

    valid = []
    for assignment in permed_assignments:
        ok = True
        for corr in correlations:
            h_exp = corr["h"]
            c_exp = corr["c"]
            idx = sig_to_idx.get(h_exp)
            if idx is None:
                continue  # H signal not in permuted set — skip
            h_chem = assignment[idx]
            c_chem = c_label_map.get(c_exp, c_exp)
            if not _are_connected(
                h_chem, c_chem, molecule, corr["type"],
                corr.get("cutoff"),
            ):
                ok = False
                break
        if ok:
            valid.append(assignment)
    return valid


def _obtain_r2a_multi(
    molecules: list,
    assignment: list[str],
    model_list: list,
    experiments: list,
    average_labels: list[list[str]],
    echo_r2: bool,
) -> float:
    """Sum of RMSEs across all temperatures for a shared permutation assignment.

    Used to find the single best permutation that minimises the total residual
    across all experimental temperatures simultaneously.
    """
    total = 0.0
    for molecule, model, experiment in zip(molecules, model_list, experiments):
        exp_copy = copy.deepcopy(experiment)
        for it, new in enumerate(assignment):
            exp_copy.signals[it].assignment = new
        model_copy = copy.deepcopy(model)
        model_copy.fit_to(molecule, exp_copy, average_labels=average_labels)
        rmse = model_copy.rmse if model_copy.fit_status else float("nan")
        total += rmse
    if echo_r2:
        print(total)
    return total


# Worker-local context for the parallel permutation search. Populated once per
# worker process via the Pool ``initializer`` so the (potentially heavy)
# molecule / experiment / model objects are pickled once per worker rather than
# once per permutation — avoiding building thousands of deep copies up front.
_PERM_CTX: dict = {}


def _perm_worker_init(ctx: dict) -> None:
    """Pool initializer: stash shared, read-only search data in each worker.

    A single mutable working copy of the experiment is created per worker; its
    signal assignments are overwritten on every task (so it can be reused
    without re-copying the spectral data it may carry).
    """
    _PERM_CTX.clear()
    _PERM_CTX.update(ctx)
    if "experiment" in ctx:
        _PERM_CTX["work_exp"] = copy.deepcopy(ctx["experiment"])


def _obtain_r2a_perm(assignment: list[str]) -> float:
    """Single-temperature permutation task using the worker-local context.

    The per-worker experiment copy is reused (assignments are fully overwritten
    each call); a fresh model copy is taken per task because ``fit_to`` mutates
    the model and must start from the same initial guess every time.
    """
    c = _PERM_CTX
    experiment = c["work_exp"]
    for it, new in enumerate(assignment):
        experiment.signals[it].assignment = new
    model = copy.deepcopy(c["susc_model"])
    model.fit_to(c["molecule"], experiment, average_labels=c["average_labels"])
    if c["echo_r2"]:
        print(model.rmse)
    return model.rmse


def _obtain_r2a_multi_perm(assignment: list[str]) -> float:
    """Shared (multi-temperature) permutation task using the worker context.

    Delegates to :func:`_obtain_r2a_multi`, which deep-copies each experiment
    and model internally, so the shared lists held in the context are never
    mutated.
    """
    c = _PERM_CTX
    return _obtain_r2a_multi(
        c["molecules"],
        assignment,
        c["susc_models"],
        c["experiments"],
        c["average_labels"],
        c["echo_r2"],
    )
