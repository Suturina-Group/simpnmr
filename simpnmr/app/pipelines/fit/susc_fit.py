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
from simpnmr.app.loaders.hfc_load import load_base_molecule_from_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.susc_load import load_susceptibilities
from simpnmr.app.params import plot_cfg as pl
from simpnmr.app.params.options import FitSuscRunOptions
from simpnmr.app.pipelines.fit.assign import generate_assignment_permutations

# Core / domain
from simpnmr.core.build.susc import build_ab_initio_chit_series
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.domain.tensor import Hyperfine
from simpnmr.core.fitting import models, vt
from simpnmr.core.pcs.isosurf import compute_pcs_isosurface

# IO layer
from simpnmr.io.csv import fit
from simpnmr.io.csv.mol import save_molecule_to_csv
from simpnmr.io.csv.susc import save_susc
from simpnmr.io.cube.pcs_iso_write import write_pcs_cube
from simpnmr.io.qc import gateway as rdrs
from simpnmr.io.xyz import xyz_write

# Visualisation
from simpnmr.viz.plots.shifts import (
    plot_fitted_shifts,
    plot_shift_contrib,
    plot_shift_spread,
)
from simpnmr.viz.plots.spect import plot_pred_spectrum
from simpnmr.viz.plots.susc import plot_exp_vs_ab_initio, plot_isoaxrho
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

    # Load hyperfines / construct base molecule
    base_molecule = load_base_molecule_from_hyperfines(
        config=config, delimiter=delimiter
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

    # Obtain fitted and fixed variables
    fit_vars = {
        key: value[1]
        for key, value in config.susc_fit_variables.items()
        if value[0] == "fit"
    }

    fix_vars = {
        key: value[1]
        for key, value in config.susc_fit_variables.items()
        if value[0] == "fix"
    }

    name_to_susc_fit: dict[str, models.SusceptibilityModel] = {
        "full": models.FullSuscFitter,
        "split": models.SplitFitter,
        "isoaxrho": models.IsoAxRhoFitter,
        "eigen": models.EigenFitter,
        "isoeigen": models.IsoEigenFitter,
    }

    model_to_use = name_to_susc_fit[config.susc_fit_type]

    # Create one susceptibility model per molecule/experiment pair
    susc_models: list[models.SusceptibilityModel] = [
        copy.deepcopy(model_to_use(fit_vars, fix_vars)) for _ in molecules
    ]

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

            # Find assignment with largest r2
            # and use in subsequent (re)fitting
            assignment = permed_assignments[np.nanargmax(results)]
            opt_r2 = np.nanmax(results)

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
                    f"# Optimal Assignment\n"
                    f"# r2 = {opt_r2:f}\n"
                    f"# T = {experiment.temperature:.2f} K"
                ),
            )

        # Fit susceptibility model to experimental chemical shifts
        # update guess using previous fit
        # susc_model.fit_vars = guess
        susc_model.fit_to(molecule, experiment, average_labels=average_labels)

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

    # Write shift data to file
    _comment_base = f"# Hyperfines from file {config.hyperfine_file}\n"
    if len(config.diamagnetic_file):
        _comment_base += f"# Diamagnetic shifts from file {config.diamagnetic_file}\n"
    if len(config.diamagnetic_ref_file):
        _comment_base += (
            f"# Diamagnetic reference from file {config.diamagnetic_ref_file}\n"
        )

    for molecule in molecules:
        comment = _comment_base + f"# T = {molecule.susc.temperature:.2f} K"
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

    if config.susc_fit_type == "isoaxrho" and spin is not None:
        fit_isoaxrho_vt(
            config=config,
            molecules=molecules,
            spin=spin,
            susc_models=susc_models,
            plot_mode=options.isoaxrho_plots,
            susc_units=options.susc_units,
            plot_profile=options.runtime.plot_profile,
            show_plots=options.runtime.show_plots,
        )

    with spec.context():
        plot_pred_spectrum(
            molecule,
            isotope=mol.nuclei[0].isotope,
            shift_range=shift_range,
            spec=spec,
            save=True,
            show=options.runtime.show_plots,
            save_name=os.path.join(
                config.project_name,
                f"pred_spectrum_{molecule.susc.temperature:.2f}_K",
            ),
        )

    return 0


def fit_isoaxrho_vt(
    config,
    molecules,
    spin,
    susc_models,
    plot_mode: pl.PlotMode,
    susc_units: str,
    plot_profile: str,
    show_plots: bool,
) -> None:
    # Define the components to fit
    fit_component = ["iso", "ax", "rho"]

    # Default to high-temperature limit unless the user explicitly requests vt_2nd_order
    method = config.susc_vt_method or "ht_limit"

    # Read VT variables (may be None if not provided)
    susc_vt_variables = config.susc_vt_variables

    if method == "vt_2nd_order":
        assert susc_vt_variables is not None

    # Load the optional susceptibility-model input used for TIP extraction
    tip_type = config.susc_vt_tip_type

    # Load temperatures from the fitted susceptibility tensors
    temps_fit = np.array([mol.susc.temperature for mol in molecules])

    # Load the fitted Iso/Ax/Rho components
    chi_vals = {
        "iso": np.array([mol.susc.iso for mol in molecules]),
        "ax": np.array([mol.susc.axiality for mol in molecules]),
        "rho": np.array([mol.susc.rhombicity for mol in molecules]),
    }

    if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
        if (
            config.susc_vt_ab_initio_format is None
            or "orca" not in config.susc_vt_ab_initio_format
        ):
            raise ValueError("Only Orca is currently supported")

        section = config.susc_vt_ab_initio_format.split("orca_", 1)[1]

        g_tensor = rdrs.read_orca_g_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )

        suscs_ab_initio = load_susceptibilities(
            config.susc_vt_ab_initio_file,
            config.susc_vt_ab_initio_format,
            electronic=molecules[0].electronic,
            g_tensor=g_tensor,
        )
        eff_H = rdrs.read_eff_hamiltonian_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )

        # Find the ab initio susceptibility temperature closest to the fit temperatures
        ab_temps = np.array([s.temperature for s in suscs_ab_initio], dtype=float)
        idx = int(np.argmin(np.abs(ab_temps - np.max(temps_fit))))

        # Use only the reference susceptibility for the VT/TIP pipeline
        susc_ab_initio = copy.deepcopy(suscs_ab_initio[idx])

        # Rotate the effective Hamiltonian tensor into the chi eigenframe
        eff_H_rot = susc_ab_initio.eigvecs.T @ eff_H @ susc_ab_initio.eigvecs

        # Construct a diagonal g-tensor in the chi eigenframe
        g_rot_diag = np.diag(
            np.diag(susc_ab_initio.eigvecs.T @ g_tensor @ susc_ab_initio.eigvecs)
        )

        # Precompute g^2 invariants in the chi eigenframe for analytic chi(T) evaluation
        g_sq = vt.compute_g_sq_components(g_rot_diag)

        # Compute the axial and rhombic parts of the effective Hamiltonian tensor (J)
        D_J, E_J = vt.calculate_E_D_components(eff_H_rot)

        # Map VT component identifiers to Susceptibility attribute names
        comp_to_attr = {"iso": "iso", "ax": "axiality", "rho": "rhombicity"}
        analytic_chi_vt: dict[str, float] = {}

        for comp in fit_component:
            analytic_val = float(
                vt.compute_analytic_component(
                    comp, susc_ab_initio.temperature, g_sq, D_J, E_J, spin
                )
            )
            analytic_chi_vt[comp] = analytic_val

            tip_ref = vt.compute_tip_correction(
                getattr(susc_ab_initio, comp_to_attr[comp]),
                analytic_val,
                spin,
            )
            susc_vt_variables[comp]["tip"] = ["fix", float(tip_ref)]

    # Initialize fitted chi errors to zero (if not available)
    chi_errors = {comp: np.zeros(len(temps_fit)) for comp in fit_component}

    # If chi errors are available, take them from the fitted model standard deviations
    if susc_models and isinstance(susc_models[0], models.IsoAxRhoFitter):
        fix = susc_models[0].fix_vars

        if "iso" not in fix:
            chi_errors["iso"] = np.array(
                [m.fit_stdev["iso"] for m in susc_models], dtype=float
            )

        if "ax" not in fix:
            chi_errors["ax"] = np.array(
                [m.fit_stdev["ax"] for m in susc_models], dtype=float
            )

        if "rho_over_ax" not in fix:
            chi_errors["rho"] = np.array(
                [m.fit_stdev["rho_over_ax"] for m in susc_models],
                dtype=float,
            )

    # Create dictionaries to store fitted chiT values, errors, and fit parameters
    chiT_reduced = {}
    chiT_err_reduced = {}
    chiT_fit_params = {}

    # Fit the VT model parameters for each susceptibility component
    for comp in fit_component:
        if method == "vt_2nd_order":
            vals, errs, params = vt.fit_chit_linear_model(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
                susc_vt_variables=susc_vt_variables[comp],
            )

        if temps_fit.size == 1 or method == "ht_limit":
            vals, errs, params = vt.compute_chit_high_t_limit(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
            )

        # Store results
        chiT_reduced[comp] = vals
        chiT_err_reduced[comp] = errs
        chiT_fit_params[comp] = params

    # Precompute inverse temperature for plotting
    inv_temps_fit = 1.0 / temps_fit

    # Build the resolved plotting contract for this run.
    spec = apply_profile(plot_profile)

    if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
        ab_series_full = build_ab_initio_chit_series(
            suscs_ab_initio,
            g_corr_iso=True,
            spin=spin,
            orbit=molecules[0].electronic.orbit_L,
            total_momentum_J=molecules[0].electronic.total_J,
            g_tensor=g_tensor,
        )

        # Native ab initio temperature grid
        exp_t = np.asarray(temps_fit, dtype=float)
        ab_inv_full = np.asarray(ab_series_full["inv_t"], dtype=float)
        ab_t_full = 1.0 / ab_inv_full

        # Keep the ab initio series as a continuous curve, clipped to the experimental
        # temperature window. If exp_t extends beyond the ab initio range, the series
        # naturally stops at the ab initio limits.
        exp_t_min = float(np.min(exp_t))
        exp_t_max = float(np.max(exp_t))

        ab_mask = (ab_t_full >= exp_t_min) & (ab_t_full <= exp_t_max)

        ab_series = {"inv_t": ab_inv_full[ab_mask]}
        for comp in fit_component:
            ab_series[comp] = np.asarray(ab_series_full[comp], dtype=float)[ab_mask]

        # Normalise ab initio chiT series by the Curie prefactor for consistency
        curie_prefactor = vt.compute_curie_prefactor(spin)
        for comp in fit_component:
            ab_series[comp] = ab_series[comp] / curie_prefactor

        with spec.context():
            plot_exp_vs_ab_initio(
                params=chiT_fit_params,
                g_sq=g_sq,
                inv_t=inv_temps_fit,
                ab_series=ab_series,
                analytic_chi_vt=analytic_chi_vt,
                spec=spec,
                show=show_plots,
                save=True,
                save_name=os.path.join(config.project_name, "exp_vs_ab_initio_susc"),
                verbose=True,
            )

    # Plot chiT temperature dependence
    with spec.context():
        plot_isoaxrho(
            vals=chiT_reduced,
            errs=chiT_err_reduced,
            params=chiT_fit_params,
            inv_t=inv_temps_fit,
            spec=spec,
            show=show_plots,
            save=True,
            save_name=os.path.join(
                config.project_name, "susceptibility_components_chiT"
            ),
            verbose=True,
        )

    # Write iso/ax/rho fit parameters to CSV
    out_file = os.path.join(config.project_name, "isoaxrho_fit.csv")
    fits_list = [
        chiT_fit_params.get("iso"),
        chiT_fit_params.get("ax"),
        chiT_fit_params.get("rho"),
    ]
    fit.save_slope_intercept(fits_list, out_file)

    return


def _obtain_r2a(
    molecule: Molecule,
    assignment: list[str],
    model: models.SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    echo_r2: bool,
):
    """
    Fit a susceptibility model for a proposed assignment and return adjusted R^2.

    This helper is designed to be run in parallel when searching over assignment
    permutations.

    Args:
        molecule (Molecule): Molecule instance used for shift prediction.
        assignment (list[str]): Proposed assignment list (one per signal).
        model (models.SusceptibilityModel): Model instance to fit.
        experiment (Experiment): Experiment data to fit against.
        average_labels (list[list[str]]): Groups of labels to average during fitting.

    Returns:
        float: Adjusted R^2 value for this assignment.
    """

    # and swap in new, permuted, assignments
    for it, new in enumerate(assignment):
        experiment.signals[it].assignment = new

    # Fit susceptibility model to experimental chemical shifts
    model.fit_to(molecule, experiment, average_labels=average_labels)

    # Print to screen if envvar enabled
    if echo_r2:
        print(model.adj_r2)

    return model.adj_r2
