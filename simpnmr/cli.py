# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""SimpNMR command-line interface.

This module defines the CLI entry points used to run SimpNMR workflows such as
pNMR prediction, susceptibility fitting, hyperfine plotting, and data extraction.
"""

import argparse
import copy
import logging
import os
import re
import sys
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathos import multiprocessing as mp
from scipy.optimize import curve_fit

from . import inputs as inps
from . import main, models
from . import utils as ut
from . import visualise as vis
from .io import writers as out
from .io.qc import qc_readers as rdrs
from .scripts import fit_vt
from .scripts.coords_tools import transform as tfm
from .scripts.coords_tools import xyz_format as xyzf

logger = logging.getLogger(__name__)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m",  # gray
        logging.INFO: "\033[36m",  # cyan
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{msg}{self.RESET}"


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.ERROR

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColorFormatter("%(levelname)-7s |  %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)


# Change figure save dialog to use current working directory
mpl.rcParams["savefig.directory"] = ""

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Change matplotlib font size to be larger
mpl.rcParams.update({"font.size": 14})

# Print r2 to terminal for each assignment
ECHO_R2 = False
if os.getenv("pnmr_echo_r2"):
    try:
        if os.getenv("pnmr_echo_r2").lower() == "true":
            ECHO_R2 = True
    except ValueError:
        logger.error("Error in pnmr_echo_r2 environment variable")
        raise SystemExit(1)

# Set user specified font name
if os.getenv("pnmr_fontname"):
    try:
        plt.rcParams["font.family"] = os.getenv("pnmr_fontname")
    except ValueError:
        logger.error("Error in pnmr_fontname environment variable")
        raise SystemExit(1)

# Set user specified plot file format
PFF = ".png"
if os.getenv("pnmr_plot_format"):
    try:
        PFF = os.getenv("pnmr_plot_format")
        if PFF[0] != ".":
            PFF = f".{PFF}"
    except ValueError:
        logger.error("Error in pnmr_plot_format environment variable")
        raise SystemExit(1)

# Set user specified plot file format
CSV_DELIMITER = ","
if os.getenv("pnmr_csvdelimiter"):
    try:
        CSV_DELIMITER = os.getenv("pnmr_csvdelimiter")
    except ValueError:
        logger.error("Error in pnmr_csvdelimiter environment variable")
        raise SystemExit(1)

_SHOW_CONV = {"on": True, "save": False, "show": True, "off": False}

_SAVE_CONV = {"on": True, "save": True, "show": False, "off": False}

_PLOT_ACTIVE = ["on", "show", "save"]


def extract_dia_func(uargs):
    """
    Extract diamagnetic isotropic shifts and save them to a CSV file.

    If a reference output file is provided, the shifts are referenced by atom type
    (non-indexed labels).

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    logger.info("Extracting shifts from %s", uargs.output_file)

    data = rdrs.QCCS.guess_from_file(uargs.output_file)

    if len(uargs.ref_output_file):
        ref_data = rdrs.QCCS.guess_from_file(uargs.ref_output_file)
        logger.info("Extracting reference shifts from %s", uargs.ref_output_file)

        ref_labels = list(ref_data.cs_iso.keys())
        ref_labels_nn = xyzf.remove_label_indices(ref_labels)

        avg_ref_iso = dict.fromkeys(ref_labels_nn, 0)

        for lab, lab_nn in zip(ref_labels, ref_labels_nn):
            avg_ref_iso[lab_nn] += ref_data.cs_iso[lab]

        for lab_nn in np.unique(ref_labels_nn):
            avg_ref_iso[lab_nn] /= ref_labels_nn.count(lab_nn)

        # Subtract from reference shifts based on atom type
        labels = list(data.cs_iso.keys())
        labels_nn = xyzf.remove_label_indices(labels)

        for lab, lab_nn in zip(labels, labels_nn):
            if lab_nn in avg_ref_iso.keys():
                data.cs_iso[lab] = avg_ref_iso[lab_nn] - data.cs_iso[lab]

    # Save diamagnetic shifts to file
    iso_shifts = list(data.cs_iso.values())
    labels = list(data.cs_iso.keys())
    labels = xyzf.remove_label_indices(labels)
    labels = xyzf.add_label_indices(labels)

    out = [f"{label}, {value:.6f}" for label, value in zip(labels, iso_shifts)]

    np.savetxt(
        "extracted_dia.csv",
        out,
        delimiter=",",
        header="atom_label, shift",
        fmt="%s",
        comments="",
    )

    logger.info("Extracted shifts saved to extracted_dia.csv")

    return


def fit_susc_func(uargs):
    """
    Run susceptibility-tensor fitting from a YAML configuration file.

    This handler builds a Molecule from the requested hyperfine source, loads
    experimental data, fits the chosen susceptibility model, generates plots, and
    writes output CSV files into the project directory.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    # Parse input file
    config = inps.FitSuscConfig.from_file(uargs.input_file)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Either load hyperfines from DFT output file
    if config.hyperfine_method == "dft":
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        # Write raw hyperfine data to output file
        qc_hyperfine_data.save_to_csv(
            os.path.join(config.project_name, "dft_hyperfines.csv"),
            verbose=True,
            delimiter=CSV_DELIMITER,
            comment=f"# Data taken from file {config.hyperfine_file}",
        )

        # Create molecule object from quantum chemical hyperfine data
        # Retain only the atoms that are given in the labels file
        base_molecule = main.Molecule.from_QCA(
            qc_hyperfine_data, converter="MHz_to_Ang-3", elements=config.nuclei_include
        )
        logger.info("Group(s)/Atoms included: %s", config.nuclei_include)
    # generate using point dipole approximation
    elif config.hyperfine_method == "pdip":
        if os.path.splitext(config.hyperfine_file)[1] == ".xyz":
            labels, coords = xyzf.load_xyz(config.hyperfine_file)
        elif os.path.splitext(config.hyperfine_file)[1] in [".log", ".out"]:
            QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
            labels = QCS.labels
            coords = QCS.coords
        else:
            raise ValueError(
                "Specified hyperfine file format "
                f"{os.path.splitext(config.hyperfine_file)[1]} unsupported"
            )

        # Create molecule
        base_molecule = main.Molecule.from_labels_coords(
            labels, coords, elements=config.nuclei_include
        )

        # Calculate point dipole hyperfine
        base_molecule.calc_pdip(config.hyperfine_pdip_centres)

    # or load from CSV
    elif config.hyperfine_method == "csv":
        base_molecule = main.Molecule.from_csv(
            config.hyperfine_file, elements=config.nuclei_include
        )

    # Load Spin
    base_molecule.electronic.load_from_config(config)
    spin = base_molecule.electronic.spin_S

    # Add chemical labels
    if len(config.chem_labels_file):
        try:
            base_molecule.add_chem_labels_from_file(config.chem_labels_file)
        except ValueError as err:
            raise ValueError(f"{err}\nCheck chem_labels and hyperfine files.")
        except KeyError as err:
            # treat missing labels/keys as a user input error
            raise ValueError(str(err))

        # Save xyz file with chemical labels for chemcraft
        base_molecule.save_chemcraft_xyz(
            file_name=os.path.join(config.project_name, "chemcraft_structure.xyz")
        )

    # Save xyz file with chemical labels for chemcraft
    base_molecule.save_xyz(
        file_name=os.path.join(config.project_name, "structure.xyz"),
        comment=f"Structure from {config.hyperfine_file}",
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
            config.diamagnetic_ref_method,
        )

    # Rotationally average hyperfines
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Create experiments
    experiments = main.Experiment.from_file(config.experiment_files)

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

    if uargs.dry_run:
        logger.info("Dry run successful — no computations executed")
        raise SystemExit(0)

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
            permed_assignments = main.Experiment.generate_permutations(
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
            iterables = [
                (
                    molecule,
                    permed_assgn,
                    susc_model,
                    copy.deepcopy(experiment),
                    average_labels,
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
            assignment = permed_assignments[np.nanargmax(results)]
            opt_r2 = np.nanmax(results)

            # and swap in new, permuted, assignments
            for it, new in enumerate(assignment):
                experiment.signals[it].assignment = new

            # Save assigned experiment to file
            experiment.to_csv(
                os.path.join(
                    config.project_name,
                    "assigned_experiment_{:.2f}_K.csv".format(experiment.temperature),
                ),
                delimiter=CSV_DELIMITER,
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

        visible = ["show", "on"]

        if uargs.shift_plots in _PLOT_ACTIVE:
            if any(cfg in visible for cfg in [uargs.contrib_plots, uargs.spread_plots]):
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
                    config.project_name, f"shifts_{experiment.temperature:.2f}_K{PFF}"
                ),
                verbose=True,
                window_title=f"Fitted shifts at {experiment.temperature:.2f} K",
            )

            visible = ["show", "on"]

            if all(
                cfg not in visible for cfg in [uargs.contrib_plots, uargs.spread_plots]
            ):
                plt.close("all")

            if uargs.spread_plots in _PLOT_ACTIVE:
                vis.plot_shift_spread(
                    molecule,
                    experiment,
                    terms=_terms,
                    show=_SHOW_CONV[uargs.spread_plots],
                    save=_SAVE_CONV[uargs.spread_plots],
                    save_name=os.path.join(
                        config.project_name,
                        f"shift_spread_{molecule.susc.temperature:.2f}_K{PFF}",
                    ),
                    verbose=True,
                    window_title=(
                        f"Spread of predicted shift components "
                        f"at {experiment.temperature:.2f} K"
                    ),
                    order="descending",
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
                        f"mean_components_{experiment.temperature:.2f}_K{PFF}",
                    ),
                    verbose=True,
                    window_title=(
                        f"Predicted shift components at {experiment.temperature:.2f} K"
                    ),
                    order="descending",
                )

                plt.close("all")

    # Write shift data to file
    _comment = f"# Hyperfines from file {config.hyperfine_file}\n"
    if len(config.diamagnetic_file):
        _comment += f"# Diamagnetic shifts from file {config.diamagnetic_file}\n"
    if len(config.diamagnetic_ref_file):
        _comment += f"# Diamagnetic reference from file {config.diamagnetic_ref_file}\n"
    _comment += f"# T = {molecule.susc.temperature:.2f} K"

    for molecule in molecules:
        molecule.to_csv(
            os.path.join(
                config.project_name,
                f"hyperfines_and_shifts_{molecule.susc.temperature:.2f}_K.csv",
            ),
            delimiter=CSV_DELIMITER,
            comment=_comment,
        )

    # Write susceptibility tensor with model terms
    out.save_susc(
        molecules,
        os.path.join(config.project_name, "susceptibility_tensor.csv"),
        susc_models=susc_models,
        susc_units=uargs.susc_units,
    )

    if uargs.pcs_isosurface:
        for molecule in molecules:
            # Generate and save PCS isosurface
            molecule.susc.save_pcs_isosurface(
                molecule.labels,
                molecule.coords,
                molecule.labels[0],
                comment="PCS Isosurface",
                file_name=os.path.join(
                    config.project_name,
                    f"pcs_isosurf_{molecule.susc.temperature:.2f}_K.cube",
                ),
            )

    shift_range = [
        np.min([nuc.shift.avg for nuc in molecule.nuclei]),
        np.max([nuc.shift.avg for nuc in molecule.nuclei]),
    ]

    extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]

    shift_range = [
        shift_range[0] + np.negative(np.max(extras)),
        shift_range[1] + np.positive(np.max(extras)),
    ]

    if config.susc_fit_type == "isoaxrho" and spin is not None:
        fit_isoaxrho_vt(
            uargs=uargs,
            config=config,
            molecules=molecules,
            spin=spin,
            susc_models=susc_models,
        )

    vis.plot_pred_spectrum(
        molecule,
        isotope=molecule.nuclei[0].isotope,
        shift_range=shift_range,
        save=True,
        show=False,
        save_name=os.path.join(
            config.project_name,
            f"pred_spectrum_{molecule.susc.temperature:.2f}_K{PFF}",
        ),
    )

    return


def fit_isoaxrho_vt(
    uargs: argparse.Namespace,
    config: inps.FitSuscConfig,
    molecules: list[main.Molecule],
    spin: float | None,
    susc_models: list[models.SusceptibilityModel] | None,
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

        suscs_ab_initio = main.Susceptibility.from_orca(
            config.susc_vt_ab_initio_file,
            section=section,
        )
        g_tensor = rdrs.read_orca_g_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )
        eff_H = rdrs.read_eff_hamiltonian_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )

        # Find the ab initio susceptibility temperature closest to the fit temperatures
        ab_temps = np.array([s.temperature for s in suscs_ab_initio], dtype=float)
        idx = int(np.argmin(np.abs(ab_temps - np.max(temps_fit))))
        susc_ab_initio = suscs_ab_initio[idx]

        # Compute the irreducible representation of the susceptibility tensor
        susc_ab_initio.calc_irred()

        # Rotate the effective Hamiltonian tensor into the chi eigenframe
        eff_H_rot = susc_ab_initio.eigvecs.T @ eff_H @ susc_ab_initio.eigvecs

        # Construct a diagonal g-tensor in the chi eigenframe
        g_rot_diag = np.diag(
            np.diag(susc_ab_initio.eigvecs.T @ g_tensor @ susc_ab_initio.eigvecs)
        )

        # Compute the axial and rhombic parts of the effective Hamiltonian tensor (J)
        D_J, E_J = fit_vt.calculate_E_D_components(eff_H_rot)

        # Compute the corrected isotropic component of the susceptibility tensor
        susc_ab_initio.iso = ut.get_true_iso_susceptibility(
            spin=spin,
            orbit=config.orbit,
            g_tensor=g_tensor,
            chi_tensors=susc_ab_initio.tensor,
            total_momentum_J=config.total_momentum_J,
        )

        # Map VT component identifiers to Susceptibility attribute names
        comp_to_attr = {"iso": "iso", "ax": "axiality", "rho": "rhombicity"}

        # Compute analytic Iso/Ax/Rho components
        for comp in fit_component:
            analytic_chi_ref = fit_vt.compute_analytic_component(
                comp, susc_ab_initio.temperature, g_rot_diag, D_J, E_J, spin
            )
            tip_ref = fit_vt.compute_tip_correction(
                getattr(susc_ab_initio, comp_to_attr[comp]),
                analytic_chi_ref,
                spin,
            )
            # Inject ab initio TIP into VT variables for this component
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
            vals, errs, params = fit_vt.fit_chit_linear_model(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
                susc_vt_variables=susc_vt_variables[comp],
            )

        if temps_fit.size == 1 or method == "ht_limit":
            vals, errs, params = fit_vt.compute_chit_high_t_limit(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
            )

        # Store results
        chiT_reduced[comp] = vals
        chiT_err_reduced[comp] = errs
        chiT_fit_params[comp] = params

    # Plot chiT temperature dependence
    vis.plot_isoaxrho(
        vals=chiT_reduced,
        errs=chiT_err_reduced,
        params=chiT_fit_params,
        temperatures=temps_fit,
        show=_SHOW_CONV[uargs.isoaxrho_plots],
        save=_SAVE_CONV[uargs.isoaxrho_plots],
        y_label=r"$\chi T$",
        save_name=os.path.join(
            config.project_name, f"susceptibility_components_chiT{PFF}"
        ),
        window_title="ChiT Susceptibility components",
        verbose=True,
        out_file=os.path.join(config.project_name, "isoaxrho_fit.csv"),
    )

    # Plot chi temperature dependence
    vis.plot_isoaxrho(
        vals=chi_vals,
        errs=chi_errors,
        params=None,
        temperatures=temps_fit,
        show=_SHOW_CONV[uargs.isoaxrho_plots],
        save=_SAVE_CONV[uargs.isoaxrho_plots],
        y_label=r"$\chi$",
        save_name=os.path.join(
            config.project_name, f"susceptibility_components_chi{PFF}"
        ),
        window_title="Susceptibility components",
        verbose=True,
        out_file=os.path.join(config.project_name, "isoaxrho_fit.csv"),
    )

    return


def obtain_r2a(
    molecule: main.Molecule,
    assignment: list[str],
    model: models.SusceptibilityModel,
    experiment: main.Experiment,
    average_labels: list[list[str]],
):
    """
    Fit a susceptibility model for a proposed assignment and return adjusted R^2.

    This helper is designed to be run in parallel when searching over assignment
    permutations.

    Args:
        molecule (main.Molecule): Molecule instance used for shift prediction.
        assignment (list[str]): Proposed assignment list (one per signal).
        model (models.SusceptibilityModel): Model instance to fit.
        experiment (main.Experiment): Experiment data to fit against.
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
    if ECHO_R2:
        print(model.adj_r2)

    return model.adj_r2


def plot_a_func(uargs):
    """
    Plot hyperfine data from a single quantum-chemistry output file.

    Depending on the flags provided, this generates spread and/or component plots and
    optionally saves the figures to disk.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(uargs.calculation_data)

    # Create molecule object from quantum chemical hyperfine data
    # to convert units
    molecule = main.Molecule.from_QCA(
        calc_data, converter="MHz_to_Ang-3", elements=uargs.elements
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
                save_name=f"hyperfine_spread_{file_head}{PFF}",
                window_title=(
                    f"Spread of hyperfine data from {uargs.calculation_data}"
                ),
                verbose=True,
            )

        vis.plot_hyperfine(
            molecule.nuclei,
            components=uargs.components,
            save=uargs.save,
            show=False,
            save_name=f"hyperfine_{file_head}{PFF}",
            window_title=f"Hyperfine data from {uargs.calculation_data}",
            verbose=True,
        )

        if not uargs.hide_plots:
            plt.show()

    return


def plot_a_iso_ax_func(uargs):
    """
    Plot isotropic-vs-axial hyperfine ratios for multiple input files.

    This handler loads the base configuration from a YAML file, computes
    A_iso/(A_xx + A_yy) per nucleus, and generates a comparison plot.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    config = inps.PlotAConfig.from_file(uargs.input_file)

    symbols = ["x", "o"]
    fig, ax = plt.subplots(1, 1)

    for hf_file, symb in zip(config.hyperfine_file[1:], symbols):
        # Either load hyperfines from DFT output file
        if config.hyperfine_method == "dft":
            qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
            # Write raw calculation data to output file
            qc_hyperfine_data.save_to_csv(
                os.path.join(config.project_name, "dft_hyperfines.csv"),
                verbose=True,
                delimiter=CSV_DELIMITER,
                comment=f"# Data taken from file {config.hyperfine_file}",
            )

            # Create molecule object from quantum chemical hyperfine data
            # Retain only the atoms that are given in the labels file
            base_molecule = main.Molecule.from_QCA(
                qc_hyperfine_data,
                converter="MHz_to_Ang-3",
                elements=config.nuclei_include,
            )

        # generate using point dipole approximation
        elif config.hyperfine_method == "pdip":
            if os.path.splitext(config.hyperfine_file)[1] == ".xyz":
                labels, coords = xyzf.load_xyz(config.hyperfine_file)
            elif os.path.splitext(config.hyperfine_file)[1] in [".log", ".out"]:
                QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
                labels = QCS.labels
                coords = QCS.coords
            else:
                raise ValueError(
                    "Specified hyperfine file format "
                    f"{os.path.splitext(hf_file)[1]} unsupported"
                )

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
            base_molecule.add_chem_labels_from_file(config.chem_labels_file)

        file_head = os.path.splitext(hf_file)[0]

        iso_div_ax = {
            nuc.chem_math_label: nuc.A.iso / (nuc.A.dip[0, 0] + nuc.A.dip[1, 1])
            for nuc in base_molecule.nuclei
        }

        if symb == "x":
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
                save_name=f"hyperfine_iso_ax_{file_head}{PFF}",
                verbose=True,
                window_title=f"Hyperfine data from {hf_file}",
            )

    xlims = ax.get_xlim()

    ax.hlines(0, *xlims, colors="k")

    ax.set_xlim(xlims)
    plt.show()
    return


def extract_a_func(uargs):
    """
    Extract hyperfine data from a quantum-chemistry output file and write a CSV.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(uargs.calculation_data)

    # Create molecule object from quantum chemical hyperfine data
    # to convert units
    base = main.Molecule.from_QCA(calc_data, converter="MHz_to_Ang-3")

    base.to_csv(
        "hyperfine_{}.csv".format(uargs.calculation_data),
        verbose=True,
        delimiter=CSV_DELIMITER,
    )

    return


def calc_pdip_func(uargs):
    """
    Compute point-dipole hyperfine dipolar tensors for a structure and optionally plot.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    # Parse user specified centres
    centres = [centre.lower().capitalize() for centre in uargs.centres]

    if os.path.splitext(uargs.structure_file)[1] == ".xyz":
        labels, coords = xyzf.load_xyz(uargs.structure_file)
    elif os.path.splitext(uargs.structure_file)[1] in [".log", ".out"]:
        QCS = rdrs.QCStructure.guess_from_file(uargs.structure_file)
        labels = QCS.labels
        coords = QCS.coords
    else:
        raise ValueError(
            "Specified hyperfine file format "
            f"{os.path.splitext(uargs.structure_file)[1]} unsupported"
        )

    # Create molecule
    molecule = main.Molecule.from_labels_coords(labels, coords, elements=uargs.elements)

    # Calculate point dipole A_dip tensor
    molecule.calc_pdip(centres)

    if uargs.chem_labels is not None:
        molecule.add_chem_labels_from_file(uargs.chem_labels)

    # Save hyperfine data to file
    out = np.array(
        [
            "{}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                nuc.label,
                nuc.chem_label,
                *nuc.A.dip[0, :],
                *nuc.A.dip[1, 1:],
                nuc.A.dip[2, 2],
            )
            for nuc in molecule.nuclei
        ]
    )

    # Save to file
    file_head = os.path.splitext(uargs.structure_file)[0]
    file_name = f"point_dipole_A_dip_{file_head}.csv"

    header = (
        "Label, Adip_xx (ppm Å^-3), "
        "Adip_xy (ppm Å^-3), "
        "Adip_xz (ppm Å^-3), "
        "Adip_yy (ppm Å^-3), "
        "Adip_yz (ppm Å^-3), "
        "Adip_zz (ppm Å^-3)"
    )

    np.savetxt(file_name, out, delimiter=",", header=header, fmt="%s")
    logger.info("Point dipole dipolar tensors saved to %s", file_name)

    if len(uargs.plot_components):
        vis.plot_hyperfine(
            molecule.nuclei,
            uargs.plot_components,
            save=True,
            show=False,
            save_name=f"point_dipole_A_dip_{file_head}{PFF}",
            verbose=True,
            window_title="Point-Dipole Hyperfines",
        )

        if uargs.chem_labels is not None:
            vis.plot_hyperfine_spread(
                molecule.nuclei,
                uargs.plot_components,
                save=True,
                show=False,
                save_name=f"spread_point_dipole_A_dip_{file_head}{PFF}",
                verbose=True,
                window_title="Point-Dipole Hyperfines Spread",
            )

        plt.show()

    return


def calc_pcs_iso_func(uargs):
    """
    Generate PCS isosurfaces for selected temperatures using an isotropic chi tensor.

    The script loads a structure, reads susceptibility data (ORCA or CSV formats),
    and writes a cube file for each requested temperature.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    if os.path.splitext(uargs.structure_file)[1] == ".xyz":
        labels, coords = xyzf.load_xyz(uargs.structure_file)
    elif os.path.splitext(uargs.structure_file)[1] in [".log", ".out"]:
        QCS = rdrs.QCStructure.guess_from_file(uargs.structure_file)
        labels = QCS.labels
        coords = QCS.coords
    else:
        raise ValueError(
            "Specified structure file format "
            f"{os.path.splitext(uargs.structure_file)[1]} unsupported"
        )

    if uargs.central_atom not in labels:
        raise ValueError(
            "Specified central atom not present in structure file \n"
            "Perhaps try with indexing e.g. Ni1"
        )

    # Load susceptibility information
    if "orca" in uargs.susc_format:
        suscs = main.Susceptibility.from_orca(
            uargs.susc_file, section=uargs.susc_format.split("orca_")[1]
        )
    elif "csv" in uargs.susc_format:
        suscs = main.Susceptibility.from_csv(uargs.susc_file)
    elif "molcas" in uargs.susc_format:
        raise ValueError("Molcas files are not currently supported")

    for susc in suscs:
        if susc.temperature in uargs.temperatures:
            # Calculate irreducible representations of susceptibility tensor
            susc.calc_irred()

            # Generate and save PCS isosurface
            susc.save_pcs_isosurface(
                labels,
                coords,
                uargs.central_atom,
                comment=(
                    f"PCS Isosurface from {uargs.susc_file} at {susc.temperature:.2f} K"
                ),
                file_name=f"pcs_isosurface_{susc.temperature:.2f}_K.cube",
            )

    return


def predict_func(uargs):
    """
    Run PCS prediction from a YAML configuration file.

    This handler loads hyperfine and susceptibility data, applies optional frame
    alignment (ORCA chi eigenframe), optionally applies a relaxation linewidth model,
    computes predicted shifts/spectra, generates plots, and writes output CSV files.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    # Parse input file
    config = inps.PredictConfig.from_file(uargs.input_file)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Either load hyperfines from DFT output file
    if config.hyperfine_method == "dft":
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        # Write raw calculation data to output file
        qc_hyperfine_data.save_to_csv(
            os.path.join(config.project_name, "dft_hyperfines.csv"),
            verbose=True,
            delimiter=CSV_DELIMITER,
            comment=f"# Data taken from file {config.hyperfine_file}",
        )

        # Create molecule object from quantum chemical hyperfine data
        # Retain only the atoms that are given in the labels file
        base_molecule = main.Molecule.from_QCA(
            qc_hyperfine_data, converter="MHz_to_Ang-3", elements=config.nuclei_include
        )

    # generate using point dipole approximation
    elif config.hyperfine_method == "pdip":
        if os.path.splitext(config.hyperfine_file)[1] == ".xyz":
            labels, coords = xyzf.load_xyz(config.hyperfine_file)
        elif os.path.splitext(config.hyperfine_file)[1] in [".log", ".out"]:
            QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
            labels = QCS.labels
            coords = QCS.coords
        else:
            raise ValueError(
                "Specified hyperfine file format "
                f"{os.path.splitext(config.hyperfine_file)[1]} unsupported"
            )

        # Create molecule
        base_molecule = main.Molecule.from_labels_coords(
            labels, coords, elements=config.nuclei_include
        )

        # Calculate point dipole hyperfine
        base_molecule.calc_pdip(config.hyperfine_pdip_centres)

    # or load from CSV
    elif config.hyperfine_method == "csv":
        base_molecule = main.Molecule.from_csv(
            config.hyperfine_file, elements=config.nuclei_include
        )

    # Load Spin
    base_molecule.electronic.load_from_config(config)
    spin = base_molecule.electronic.spin_S

    # Add chemical labels
    if len(config.chem_labels_file):
        base_molecule.add_chem_labels_from_file(config.chem_labels_file)

        # Save xyz file with chemical labels for chemcraft
        base_molecule.save_chemcraft_xyz(
            file_name=os.path.join(config.project_name, "chemcraft_structure.xyz")
        )

    # Save xyz file with chemical labels for chemcraft
    base_molecule.save_xyz(
        file_name=os.path.join(config.project_name, "structure.xyz"),
        comment=f"Structure from {config.hyperfine_file}",
    )

    # Load diamagnetic shift file
    if len(config.diamagnetic_file):
        base_molecule.load_diamagnetic_shifts(
            config.diamagnetic_file,
            config.diamagnetic_method,
            config.diamagnetic_ref_file,
            config.diamagnetic_ref_method,
        )

    # Rotationally average hyperfines of user selected nuclei:
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Rotate hyperfine tensors from DFT frame into chi eigenframe (if provided)
    if "orca" in config.susceptibility_format:
        rot_mat, trans_mat = tfm.get_rotation_and_transformation(config)
        base_molecule.rotate_hyperfines(rot_mat)

        # Rotate HFC coords frame into chi eigenframe and save the transformed coords
        tfm.rotate_coords_to_chi_frame(config.project_name, config)

    # Load susceptibility information
    if "orca" in config.susceptibility_format:
        suscs = main.Susceptibility.from_orca(
            config.susceptibility_file,
            section=config.susceptibility_format.split("orca_")[1],
            # section = 'auto'
        )
    elif "csv" in config.susceptibility_format:
        suscs = main.Susceptibility.from_csv(config.susceptibility_file)
    elif "molcas" in config.susceptibility_format:
        raise ValueError("Molcas files are not currently supported")

    suscs = [
        susc for susc in suscs if susc.temperature in config.susceptibility_temperatures
    ]

    if not len(suscs):
        raise ValueError("No susceptibility data found for specified temperature(s)")

    # Calculate linewidths using user-specified relaxation model (optional)
    if not getattr(config, "relaxation_model", None):
        (
            logger.warning(
                "No relaxation model specified — linewidths will be fixed at 1 ppm."
            ),
        )
    elif config.relaxation_magnetic_field_tesla is None:
        logger.warning(
            "relaxation_magnetic_field_tesla "
            "not provided — relaxation effects skipped, "
            "linewidths will be fixed at 1 ppm \n"
        )
    else:
        apply_relaxation_model(config, base_molecule)

    # Load experimental data from file into list of experiment objects
    if len(config.experiment_files):
        experiments = main.Experiment.from_file(config.experiment_files)
        for susc, exp in zip(suscs, experiments):
            if susc.temperature != exp.temperature:
                logger.warning(
                    "Mismatch in Susceptibility (%.2f K) and "
                    "Experimental (%.2f K) temperatures",
                    susc.temperature,
                    exp.temperature,
                )
            if re.sub("[0-9]", "", exp.isotope) not in config.nuclei_include:
                logger.warning(
                    "Experimental isotope (%s) not requested in input file (%s)",
                    exp.isotope,
                    config.nuclei_include,
                )
    else:
        experiments = [None] * len(suscs)

    # Create a molecule object which accompanies each experiment object
    molecules = [copy.deepcopy(base_molecule) for _ in range(len(experiments))]

    if len(config.experiment_spectrum_files):
        for experiment, spectrum in zip(experiments, config.experiment_spectrum_files):
            experiment.load_spectrum_from_file(spectrum)

    _terms = ["pc", "fc", "d"]

    if config.hyperfine_method == "pdip":
        _terms.pop(_terms.index("fc"))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index("d"))

    if "orca" in config.susceptibility_format:
        section = config.susceptibility_format.split("orca_")[1]
        g_tensor = rdrs.read_orca_g_tensor(
            config.susceptibility_file,
            section=section,
        )
        chi_tensors = rdrs.read_orca_susceptibility(
            config.susceptibility_file,
            section=section,
        )

    # Determine how to compute chi_iso in the next step:
    use_orca_correction = (
        "orca" in config.susceptibility_format
        and spin is not None
        and g_tensor is not None
        and chi_tensors is not None
    )

    # Update susceptibility tensor of Molecule using model
    for molecule, susc, experiment in zip(molecules, suscs, experiments):
        molecule.susc = susc

        if use_orca_correction:
            # Compute the corrected isotropic component of the susceptibility tensor
            susc.iso = ut.get_true_iso_susceptibility(
                spin=spin,
                orbit=config.orbit,
                g_tensor=g_tensor,
                chi_tensors=susc.tensor,
                total_momentum_J=config.total_momentum_J,
            )
        elif config.susceptibility_format == "csv":
            pass
        elif spin is not None:
            # Fall back to a spin-only Curie susceptibility
            # when no ORCA susceptibility tensor is provided
            susc.iso = ut.get_spin_only_susceptibility(
                spin=spin,
                orbit=config.orbit,
                total_momentum_J=config.total_momentum_J,
                temperature=susc.temperature,
            )
        else:
            logger.info(
                "Spin not specified and could not be inferred — "
                "using chi iso from susceptibility file (no spin-only correction)"
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
            save_name=os.path.join(
                config.project_name,
                f"pred_shift_spread_{molecule.susc.temperature:.2f}_K{PFF}",
            ),
            verbose=True,
            window_title=f"Spread of predicted shifts at {susc.temperature:.2f} K",
            order="descending",
        )

        # Bar chart for means
        vis.plot_shift_contrib(
            molecule,
            experiment=experiment,
            save=True,
            show=False,
            save_name=os.path.join(
                config.project_name,
                f"pred_mean_components_{molecule.susc.temperature:.2f}_K{PFF}",
            ),
            verbose=True,
            window_title=(
                f"Predicted mean shifts and components at {susc.temperature:.2f} K"
            ),
            order="descending",
        )

        shift_range = [
            np.min([nuc.shift.avg for nuc in molecule.nuclei]),
            np.max([nuc.shift.avg for nuc in molecule.nuclei]),
        ]

        extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]

        shift_range = [
            shift_range[0] + np.negative(np.max(extras)),
            shift_range[1] + np.positive(np.max(extras)),
        ]

        if len(config.experiment_files):
            vis.plot_raw_deconv_pred(
                molecule=molecule,
                isotope=molecule.nuclei[0].isotope,
                shift_range=shift_range,
                experiment=experiment,
                save=True,
                show=False,
                save_name=os.path.join(
                    config.project_name,
                    f"pred_and_exp_spectrum_{molecule.susc.temperature:.2f}_K{PFF}",
                ),
            )
        vis.plot_pred_spectrum(
            molecule,
            isotope=molecule.nuclei[0].isotope,
            shift_range=shift_range,
            save=True,
            show=False,
            save_name=os.path.join(
                config.project_name,
                f"pred_spectrum_{molecule.susc.temperature:.2f}_K{PFF}",
            ),
        )

        plt.show()

        plt.close("all")

    # TODO If more than one temperature, then make a stacked plot of spectra

    # Save susceptibility data to file
    out.save_susc(
        molecules,
        os.path.join(config.project_name, "susceptibility_tensor.csv"),
        comment="#Data from {} ({})".format(
            config.susceptibility_file, config.susceptibility_format
        ),
        susc_units=uargs.susc_units,
    )

    # Write shift data to file
    for molecule in molecules:
        molecule.to_csv(
            os.path.join(
                config.project_name,
                f"hyperfines_and_shifts_{molecule.susc.temperature:.2f}_K.csv",
            ),
            delimiter=CSV_DELIMITER,
            comment=f"# T = {molecule.susc.temperature:.2f} K",
        )

    return


def apply_relaxation_model(config: inps.PredictConfig, base_molecule: main.Molecule):
    """
    Apply linewidths using a user-specified relaxation model.

    This function updates `base_molecule.nuclei` in-place by setting `nuc.shift.lw`
    when relaxation inputs are provided in the config.

    Args:
        config (inps.PredictConfig): Prediction configuration containing relaxation
            settings and physical parameters.
        base_molecule (main.Molecule): Molecule instance to update in-place.

    Returns:
        None
    """

    # Solomon linewidths if relaxation model is SBM
    nuclei_labels = (
        config.nuclei_include
        if isinstance(config.nuclei_include, list)
        else [config.nuclei_include]
    )

    # Use all nuclei in the molecule that match the requested element(s)
    nuclei_coords = {
        nuc.label: nuc.coord
        for nuc in base_molecule.nuclei
        if ut.lf.remove_numbers(nuc.label) in nuclei_labels
    }
    electron_coords = config.relaxation_electron_coords
    B0 = config.relaxation_magnetic_field_tesla

    # Build Aiso, gamma and omega dictionaries for selected nuclei
    # Converts nuclear gyromagnetic ratios from MHz/T to rad/s/T
    # and multiplies Aiso by 1e6 to convert from MHz to Hz

    if config.hyperfine_method == "pdip":
        # In point-dipole (pdip) model, contact hyperfine A_iso = 0 for all nuclei.
        A_iso_dict = {label: 0.0 for label in nuclei_coords}
    else:
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        A_iso_dict_MHz = qc_hyperfine_data.a_iso  # MHz
        A_iso_dict = {
            nuc.label: A_iso_dict_MHz[nuc.label] * 1e6
            for nuc in base_molecule.nuclei
            if nuc.label in nuclei_coords
        }

    gamma_I_dict = {
        label: ut.NUCLEAR_GAMMAS[ut.lf.remove_numbers(label)] * 2 * np.pi * 1e6
        for label in nuclei_coords
    }
    omega_I_dict = {label: gamma_I_dict[label] * B0 for label in nuclei_coords}
    omega_S = ut.EGAMMA * B0 * 2 * np.pi * 1e6
    tau_c1 = 1 / ((1 / config.relaxation_tR) + (1 / config.relaxation_T1e))
    tau_c2 = 1 / ((1 / config.relaxation_tR) + (1 / config.relaxation_T2e))
    tau_e1 = config.relaxation_T1e
    tau_e2 = config.relaxation_T2e
    tau_R = config.relaxation_tR

    # Load electronic states
    spin = base_molecule.electronic.spin_S
    orbit = base_molecule.electronic.orbit_L
    total_momentum_J = base_molecule.electronic.total_J

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
            spin,
            orbit,
            total_momentum_J,
        )
        # Calculate SBM contact rates (R1)
        sbm_contact_r1_rates = ut.sbm_r1_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e2,
            spin,
            total_momentum_J,
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
            spin,
            orbit,
            total_momentum_J,
        )
        # Calculate SBM contact rates (R2)
        sbm_contact_r2_rates = ut.sbm_r2_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e1,
            tau_e2,
            spin,
            total_momentum_J,
        )
        # Combine rates into a single dictionary
        rates_r1 = {
            label: sbm_dipolar_r1_rates[label] + sbm_contact_r1_rates[label]
            for label in nuclei_coords
        }
        rates_r2 = {
            label: sbm_dipolar_r2_rates[label] + sbm_contact_r2_rates[label]
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
            spin,
            orbit,
            total_momentum_J,
        )
        curie_r2_rates = ut.gueron_r2_curie(
            list(nuclei_coords.keys()),
            nuclei_coords,
            electron_coords,
            omega_I_dict,
            config.relaxation_temperature,
            tau_R,
            spin,
            orbit,
            total_momentum_J,
        )
        rates_r1 = {label: curie_r1_rates[label] for label in nuclei_coords}
        rates_r2 = {label: curie_r2_rates[label] for label in nuclei_coords}

    # Combined SBM and Curie mechanisms
    elif (
        config.relaxation_model == "sbm curie" or config.relaxation_model == "curie sbm"
    ):
        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
            list(nuclei_coords.keys()),
            nuclei_coords,
            electron_coords,
            gamma_I_dict,
            omega_I_dict,
            omega_S,
            tau_c1,
            tau_c2,
            spin,
            orbit,
            total_momentum_J,
        )
        sbm_contact_r1_rates = ut.sbm_r1_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e1,
            spin,
            total_momentum_J,
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
            spin,
            orbit,
            total_momentum_J,
        )

        # Calculate SBM contact rates
        sbm_contact_r2_rates = ut.sbm_r2_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e1,
            tau_e2,
            spin,
            total_momentum_J,
        )

        curie_r1_rates = ut.gueron_r1_curie(
            list(nuclei_coords.keys()),
            nuclei_coords,
            electron_coords,
            omega_I_dict,
            config.relaxation_temperature,
            tau_R,
            spin,
            orbit,
            total_momentum_J,
        )
        curie_r2_rates = ut.gueron_r2_curie(
            list(nuclei_coords.keys()),
            nuclei_coords,
            electron_coords,
            omega_I_dict,
            config.relaxation_temperature,
            tau_R,
            spin,
            orbit,
            total_momentum_J,
        )

        rates_r1 = {
            label: sbm_dipolar_r1_rates[label]
            + sbm_contact_r1_rates[label]
            + curie_r1_rates[label]
            for label in nuclei_coords
        }
        rates_r2 = {
            label: sbm_dipolar_r2_rates[label]
            + sbm_contact_r2_rates[label]
            + curie_r2_rates[label]
            for label in nuclei_coords
        }

    # Group rates by chemical label
    r1_by_chem_label = defaultdict(list)
    for nuc in base_molecule.nuclei:
        if nuc.label in rates_r1:
            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

    r2_by_chem_label = defaultdict(list)
    for nuc in base_molecule.nuclei:
        if nuc.label in rates_r2:
            r2_by_chem_label[nuc.chem_label].append(rates_r2[nuc.label])

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

    if "sbm" in config.relaxation_model:
        dipolar_by_chem_label = defaultdict(list)
        contact_by_chem_label = defaultdict(list)
        for nuc in base_molecule.nuclei:
            if "sbm_dipolar_r1_rates" in locals() and nuc.label in sbm_dipolar_r1_rates:
                dipolar_by_chem_label[nuc.chem_label].append(
                    sbm_dipolar_r1_rates[nuc.label]
                )
            if "sbm_contact_r1_rates" in locals() and nuc.label in sbm_contact_r1_rates:
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

    if "curie" in config.relaxation_model:
        curie_by_chem_label = defaultdict(list)
        for nuc in base_molecule.nuclei:
            if "curie_r1_rates" in locals() and nuc.label in curie_r1_rates:
                curie_by_chem_label[nuc.chem_label].append(curie_r1_rates[nuc.label])
        avg_curie_by_chem_label = {
            chem_label: np.mean(rate_list)
            for chem_label, rate_list in curie_by_chem_label.items()
        }

    # Save the relaxation data to CSV
    out.save_relaxation_decomposition(
        file_name=os.path.join(config.project_name, "relaxation_decomposition.csv"),
        avg_r1_by_chem_label=avg_r1_by_chem_label,
        avg_r2_by_chem_label=avg_r2_by_chem_label,
        avg_lw_by_chem_label=avg_lw_by_chem_label,
        avg_dipolar_by_chem_label=avg_dipolar_by_chem_label,
        avg_contact_by_chem_label=avg_contact_by_chem_label,
        avg_curie_by_chem_label=avg_curie_by_chem_label,
    )

    for nuc in base_molecule.nuclei:
        if nuc.chem_label in avg_lw_by_chem_label:
            nuc.shift.lw = (
                avg_lw_by_chem_label[nuc.chem_label]
                / (abs(omega_I_dict[nuc.label]) / (2 * np.pi))
                * 1e6
            )

    return


def fit_corr_time_func(uargs):
    """
    Fit correlation time parameters (tau_R and/or tau_E) to experimental R1 values.

    This handler reads experiments and hyperfine data, evaluates the requested
    relaxation model, and uses `scipy.optimize.curve_fit` to fit correlation times.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    config = inps.FitCorrTimeConfig.from_file(uargs.input_file)

    if config.spin_S is not None:
        spin = config.spin_S
    else:
        spin = rdrs.QCSpin.guess_from_file(config.hyperfine_file).S

    orbit = config.orbit

    total_momentum_J = config.total_momentum_J

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    tau_R_mode, tau_R_guess = (
        config.fit_corr_time_tau_R[0].lower(),
        config.fit_corr_time_tau_R[1],
    )
    tau_R_bounds = (
        config.fit_corr_time_tau_R[2] if len(config.fit_corr_time_tau_R) > 2 else None
    )

    tau_E_mode, tau_E_guess = (
        config.fit_corr_time_tau_E[0].lower(),
        config.fit_corr_time_tau_E[1],
    )
    tau_E_bounds = (
        config.fit_corr_time_tau_E[2] if len(config.fit_corr_time_tau_E) > 2 else None
    )

    if tau_R_mode == "fix" and tau_E_mode == "fit":
        fix_param = "tau_r"
    elif tau_R_mode == "fit" and tau_E_mode == "fix":
        fix_param = "tau_e"
    elif tau_R_mode == "fit" and tau_E_mode == "fit":
        fix_param = None  # Fit both
    elif tau_R_mode == "fix" and tau_E_mode == "fix":
        raise ValueError(
            "Both tau_R and tau_E cannot be fixed. At least one must be set to 'fit'."
        )
    else:
        raise ValueError(
            "Use syntax 'tau_C: [fit/fix, guess, [upper-bound, lower-bound]]', "
            "with bounds optional (tau_C refers to tau_R or tau_E)."
        )

    # Placeholders for fitted parameters and covariance
    tau_R_fit = None
    tau_E_fit = None
    pcov = None
    initial_guess = None

    if (
        getattr(config, "fit_corr_time_tau_R", None) is not None
        and getattr(config, "relaxation_model", None) is not None
    ):
        experiments = main.Experiment.from_file(config.experiment_files)

        # Filter signals to only those with valid R1 values
        # Only include signals for specified elements (e.g., 'C')

        elements = (
            config.nuclei_include
            if isinstance(config.nuclei_include, list)
            else [config.nuclei_include]
        )

        exp_blocks = []
        for experiment in experiments:
            labels_this = []
            r1_this = []
            for signal in experiment.signals:
                if (
                    signal.r1 is not None
                    and np.isfinite(signal.r1)
                    and any(signal.assignment.startswith(e) for e in elements)
                ):
                    labels_this.append(signal.assignment)
                    r1_this.append(signal.r1)
            if len(labels_this) > 0:
                exp_blocks.append(
                    (experiment, np.array(labels_this), np.array(r1_this))
                )

        if not exp_blocks:
            raise ValueError("No valid experimental R1 values found for fitting.")

        chem_labels = np.concatenate([blk[1] for blk in exp_blocks])
        exp_r1 = np.concatenate([blk[2] for blk in exp_blocks])
        xdata = np.arange(len(exp_r1))

        # Load hyperfine data and create molecule object
        if config.hyperfine_method == "dft":
            qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
            qc_hyperfine_data.save_to_csv(
                os.path.join(config.project_name, "dft_hyperfines.csv"),
                verbose=True,
                delimiter=CSV_DELIMITER,
                comment=f"# Data taken from file {config.hyperfine_file}",
            )
            base_molecule = main.Molecule.from_QCA(
                qc_hyperfine_data,
                converter="MHz_to_Ang-3",
                elements=config.nuclei_include,
            )
        elif config.hyperfine_method == "pdip":
            if os.path.splitext(config.hyperfine_file)[1] == ".xyz":
                labels, coords = xyzf.load_xyz(config.hyperfine_file)
            elif os.path.splitext(config.hyperfine_file)[1] in [".log", ".out"]:
                QCS = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
                labels = QCS.labels
                coords = QCS.coords
            else:
                raise ValueError(
                    "Specified hyperfine file format "
                    f"{os.path.splitext(config.hyperfine_file)[1]} unsupported"
                )
            base_molecule = main.Molecule.from_labels_coords(
                labels, coords, elements=config.nuclei_include
            )
            base_molecule.calc_pdip(config.hyperfine_pdip_centres)
        elif config.hyperfine_method == "csv":
            base_molecule = main.Molecule.from_csv(
                config.hyperfine_file, elements=config.nuclei_include
            )

        # Add chemical labels if provided
        if len(config.chem_labels_file):
            base_molecule.add_chem_labels_from_file(config.chem_labels_file)
            base_molecule.save_chemcraft_xyz(
                file_name=os.path.join(config.project_name, "chemcraft_structure.xyz")
            )
        base_molecule.save_xyz(
            file_name=os.path.join(config.project_name, "structure.xyz"),
            comment=f"Structure from {config.hyperfine_file}",
        )
        label_to_chem_label = {
            nuc.label: nuc.chem_label for nuc in base_molecule.nuclei
        }

        # Prepare relaxation model inputs
        nuclei_coords = {nuc.label: nuc.coord for nuc in base_molecule.nuclei}
        electron_coords = config.relaxation_electron_coords

        # Dictionaries for relaxation calculations
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        A_iso_dict_MHz = qc_hyperfine_data.a_iso
        A_iso_dict = {
            nuc.label: A_iso_dict_MHz[nuc.label] * 1e6 for nuc in base_molecule.nuclei
        }
        gamma_I_dict = {
            label: ut.NUCLEAR_GAMMAS[ut.lf.remove_numbers(label)] * 2 * np.pi * 1e6
            for label in nuclei_coords
        }

        multiplicity = rdrs.read_gaussian_log_spin(config.hyperfine_file)
        spin = (multiplicity - 1) / 2

        # --- Model function for curve_fit ---
        if fix_param == "tau_r":
            tau_R = float(tau_R_guess)
            initial_guess = [float(tau_E_guess)]

            def r1_model(_, tau_E):
                """
                Compute model R1 values for the current tau_E with tau_R fixed
                across all experiments.

                Args:
                    _ (np.ndarray): Dummy data input required by `curve_fit` (not used).
                    tau_E (float): Electron relaxation time (s).

                Returns:
                    np.ndarray: Model R1 values aligned with the flattened
                    experimental array.
                """
                tau_c1 = 1.0 / ((1.0 / tau_R) + (1.0 / tau_E))
                tau_c2 = tau_c1

                theory_all = []

                for experiment, labels_this, r1_this in exp_blocks:
                    B0 = experiment.magnetic_field
                    temp = experiment.temperature
                    omega_I_dict = {
                        label: -gamma_I_dict[label] * B0 for label in nuclei_coords
                    }
                    omega_S = -ut.EGAMMA * B0 * 2 * np.pi * 1e6

                    # Calculate relaxation rates for current tau_R, tau_E
                    if config.relaxation_model == "sbm":
                        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            gamma_I_dict,
                            omega_I_dict,
                            omega_S,
                            tau_c1,
                            tau_c2,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        sbm_contact_r1_rates = ut.sbm_r1_contact(
                            list(nuclei_coords.keys()),
                            A_iso_dict,
                            omega_I_dict,
                            omega_S,
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            label: sbm_dipolar_r1_rates[label]
                            + sbm_contact_r1_rates[label]
                            for label in nuclei_coords
                        }
                    elif config.relaxation_model == "curie":
                        curie_r1_rates = ut.gueron_r1_curie(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            omega_I_dict,
                            temp,
                            tau_R,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            label: curie_r1_rates[label] for label in nuclei_coords
                        }
                    elif config.relaxation_model in ["sbm curie", "curie sbm"]:
                        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            gamma_I_dict,
                            omega_I_dict,
                            omega_S,
                            tau_c1,
                            tau_c2,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        sbm_contact_r1_rates = ut.sbm_r1_contact(
                            list(nuclei_coords.keys()),
                            A_iso_dict,
                            omega_I_dict,
                            omega_S,
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        curie_r1_rates = ut.gueron_r1_curie(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            omega_I_dict,
                            temp,
                            tau_R,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            label: sbm_dipolar_r1_rates[label]
                            + sbm_contact_r1_rates[label]
                            + curie_r1_rates[label]
                            for label in nuclei_coords
                        }
                    else:
                        raise ValueError("Unknown relaxation model")

                    # Group rates by chemical label
                    r1_by_chem_label = defaultdict(list)
                    for nuc in base_molecule.nuclei:
                        if nuc.label in rates_r1:
                            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

                    # Calculate average R1 rates for each chemical label
                    avg_r1_by_chem_label = {
                        chem_label: np.mean(rate_list)
                        for chem_label, rate_list in r1_by_chem_label.items()
                    }

                    for label in labels_this:
                        chem_label = label_to_chem_label.get(label, label)
                        theory_all.append(avg_r1_by_chem_label.get(chem_label, np.nan))

                return np.array(theory_all)

            # --- Run the fit ---
            if tau_E_bounds:
                popt, pcov = curve_fit(
                    r1_model, xdata, exp_r1, p0=initial_guess, bounds=tau_E_bounds
                )
            elif tau_E_bounds is None:
                popt, pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

            tau_E_fit = popt[0]
            theory_r1 = r1_model(xdata, tau_E_fit)
            if tau_E_fit <= 0:
                raise ValueError(f"Fitted tau_E is negative: {tau_E_fit:.3e} s.")

        elif fix_param == "tau_e":
            tau_E = float(tau_E_guess)
            initial_guess = [float(tau_R_guess)]

            def r1_model(_, tau_R):
                """
                Compute model R1 values for the current tau_R with tau_E fixed
                across all experiments.

                Args:
                    _ (np.ndarray): Dummy data input required by `curve_fit` (not used).
                    tau_R (float): Rotational correlation time (s).

                Returns:
                    np.ndarray: Model R1 values aligned with the flattened
                    experimental array.
                """
                tau_c1 = 1 / ((1 / tau_R) + (1 / tau_E))
                tau_c2 = tau_c1

                theory_all = []

                for experiment, labels_this, r1_this in exp_blocks:
                    B0 = experiment.magnetic_field
                    temp = experiment.temperature
                    omega_I_dict = {
                        label: -gamma_I_dict[label] * B0 for label in nuclei_coords
                    }
                    omega_S = -ut.EGAMMA * B0 * 2 * np.pi * 1e6

                    # Calculate relaxation rates for current tau_R, tau_E
                    if config.relaxation_model == "sbm":
                        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            gamma_I_dict,
                            omega_I_dict,
                            omega_S,
                            tau_c1,
                            tau_c2,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        sbm_contact_r1_rates = ut.sbm_r1_contact(
                            list(nuclei_coords.keys()),
                            A_iso_dict,
                            omega_I_dict,
                            omega_S,
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            label: sbm_dipolar_r1_rates[label]
                            + sbm_contact_r1_rates[label]
                            for label in nuclei_coords
                        }
                    elif config.relaxation_model == "curie":
                        curie_r1_rates = ut.gueron_r1_curie(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            omega_I_dict,
                            temp,
                            tau_R,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            label: curie_r1_rates[label] for label in nuclei_coords
                        }
                    elif config.relaxation_model in ["sbm curie", "curie sbm"]:
                        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            gamma_I_dict,
                            omega_I_dict,
                            omega_S,
                            tau_c1,
                            tau_c2,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        sbm_contact_r1_rates = ut.sbm_r1_contact(
                            list(nuclei_coords.keys()),
                            A_iso_dict,
                            omega_I_dict,
                            omega_S,
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        curie_r1_rates = ut.gueron_r1_curie(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            omega_I_dict,
                            temp,
                            tau_R,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            label: sbm_dipolar_r1_rates[label]
                            + sbm_contact_r1_rates[label]
                            + curie_r1_rates[label]
                            for label in nuclei_coords
                        }
                    else:
                        raise ValueError("Unknown relaxation model")

                    # Group rates by chemical label
                    r1_by_chem_label = defaultdict(list)
                    for nuc in base_molecule.nuclei:
                        if nuc.label in rates_r1:
                            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

                    # Calculate average R1 rates for each chemical label
                    avg_r1_by_chem_label = {
                        chem_label: np.mean(rate_list)
                        for chem_label, rate_list in r1_by_chem_label.items()
                    }

                    for label in labels_this:
                        chem_label = label_to_chem_label.get(label, label)
                        theory_all.append(avg_r1_by_chem_label.get(chem_label, np.nan))

            # --- Run the fit ---
            if tau_R_bounds:
                popt, pcov = curve_fit(
                    r1_model, xdata, exp_r1, p0=initial_guess, bounds=tau_R_bounds
                )
            elif tau_R_bounds is None:
                popt, pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

            tau_R_fit = popt[0]
            theory_r1 = r1_model(xdata, tau_R_fit)
            if tau_R_fit <= 0:
                raise ValueError(f"Fitted tau_R is negative: {tau_R_fit:.3e} s.")
            else:
                logger.info("Fitted tau_R: %.3e s", tau_R_fit)

        elif not fix_param or fix_param in ["none", ""]:
            # Fit both tau_R and tau_E
            initial_guess = [float(tau_R_guess), float(tau_E_guess)]
            bounds = None
            if tau_R_bounds and tau_E_bounds:
                bounds = (
                    [tau_R_bounds[0], tau_E_bounds[0]],
                    [tau_R_bounds[1], tau_E_bounds[1]],
                )

            def r1_model(_, tau_R, tau_E):
                """
                Compute model R1 values for the current (tau_R, tau_E)
                across all experiments.

                Args:
                    _ (np.ndarray): Dummy data input required by `curve_fit` (not used).
                    tau_R (float): Rotational correlation time (s).
                    tau_E (float): Electron relaxation time (s).

                Returns:
                    np.ndarray: Model R1 values aligned with the flattened
                    experimental array.
                """
                tau_c1 = 1.0 / ((1.0 / tau_R) + (1.0 / tau_E))
                tau_c2 = tau_c1

                theory_all = []

                for experiment, labels_this, r1_this in exp_blocks:
                    B0 = experiment.magnetic_field
                    Temp = experiment.temperature

                    # Dictionaries for relaxation calculations
                    omega_I_dict = {
                        label: -gamma_I_dict[label] * B0 for label in nuclei_coords
                    }
                    omega_S = -ut.EGAMMA * B0 * 2 * np.pi * 1e6

                    if config.relaxation_model == "sbm":
                        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            gamma_I_dict,
                            omega_I_dict,
                            omega_S,
                            tau_c1,
                            tau_c2,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        sbm_contact_r1_rates = ut.sbm_r1_contact(
                            list(nuclei_coords.keys()),
                            A_iso_dict,
                            omega_I_dict,
                            omega_S,
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            lab: sbm_dipolar_r1_rates[lab] + sbm_contact_r1_rates[lab]
                            for lab in nuclei_coords
                        }

                    elif config.relaxation_model == "curie":
                        curie_r1_rates = ut.gueron_r1_curie(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            omega_I_dict,
                            Temp,
                            tau_R,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        rates_r1 = {lab: curie_r1_rates[lab] for lab in nuclei_coords}

                    elif config.relaxation_model in ["sbm curie", "curie sbm"]:
                        sbm_dipolar_r1_rates = ut.sbm_r1_dipolar(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            gamma_I_dict,
                            omega_I_dict,
                            omega_S,
                            tau_c1,
                            tau_c2,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        sbm_contact_r1_rates = ut.sbm_r1_contact(
                            list(nuclei_coords.keys()),
                            A_iso_dict,
                            omega_I_dict,
                            omega_S,
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        curie_r1_rates = ut.gueron_r1_curie(
                            list(nuclei_coords.keys()),
                            nuclei_coords,
                            electron_coords,
                            omega_I_dict,
                            Temp,
                            tau_R,
                            spin,
                            orbit,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            lab: sbm_dipolar_r1_rates[lab]
                            + sbm_contact_r1_rates[lab]
                            + curie_r1_rates[lab]
                            for lab in nuclei_coords
                        }
                    else:
                        raise ValueError("Unknown relaxation model")

                    # Group rates by chemical label
                    r1_by_chem_label = defaultdict(list)
                    for nuc in base_molecule.nuclei:
                        if nuc.label in rates_r1:
                            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

                    # Calculate average R1 rates for each chemical label
                    avg_r1_by_chem_label = {
                        chem_label: np.mean(rate_list)
                        for chem_label, rate_list in r1_by_chem_label.items()
                    }

                    for label in labels_this:
                        chem_label = label_to_chem_label.get(label, label)
                        theory_all.append(avg_r1_by_chem_label.get(chem_label, np.nan))

                return np.array(theory_all)

            # --- Run the fit ---
            if bounds:
                popt, pcov = curve_fit(
                    r1_model, xdata, exp_r1, p0=initial_guess, bounds=bounds
                )
            else:
                popt, pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

            tau_R_fit, tau_E_fit = popt
            theory_r1 = r1_model(xdata, tau_R_fit, tau_E_fit)
            if tau_R_fit <= 0 and tau_E_fit > 0:
                raise ValueError(f"tau_R is negative: {tau_R_fit:.3e} s.")
            elif tau_E_fit <= 0 and tau_R_fit > 0:
                raise ValueError(f"tau_E is negative: {tau_E_fit:.3e} s.")
            elif tau_R_fit <= 0 and tau_E_fit <= 0:
                raise ValueError(
                    f"Both tau_R and tau_E are negative: "
                    f"tau_R = {tau_R_fit:.3e} s, tau_E = {tau_E_fit:.3e} s."
                )
        else:
            raise ValueError("Correlation times must be 'tau_r' or 'tau_e'.")

        rsquared = 1 - (
            np.sum((exp_r1 - theory_r1) ** 2) / np.sum((exp_r1 - np.mean(exp_r1)) ** 2)
        )

        # Save fit diagnostics
        out.save_corr_time_fit_data(
            xdata=xdata,
            exp_r1=exp_r1,
            chem_labels=chem_labels,
            file_name=os.path.join(
                config.project_name, "corr_time_fit_diagnostics.csv"
            ),
            initial_guess=initial_guess,
            fitted_tau_r=tau_R_fit,
            fitted_tau_e=tau_E_fit,
            covariance=pcov,
            delimiter=CSV_DELIMITER,
            comment=f"r2: {rsquared:.6f}",
            verbose=True,
        )

        # Plot experimental vs theoretical R2
        plt.figure(figsize=(6, 6))
        plt.scatter(theory_r1, exp_r1, marker="x", color="blue")

        for x, y, label in zip(theory_r1, exp_r1, chem_labels):
            plt.text(x, y, label, fontsize=12)

        # Add x = y reference line
        min_val = min(np.min(theory_r1), np.min(exp_r1))
        max_val = max(np.max(theory_r1), np.max(exp_r1))
        plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, label="x = y")

        plt.xlabel("Fitted $R_1$ (s$^{-1}$)", fontsize=14)
        plt.ylabel("Experimental $R_1$ (s$^{-1}$)", fontsize=14)
        plt.title("Experimental vs Fitted $R_1$", fontsize=16)

        # Print R2 above the plot
        plt.text(
            0.01,
            0.96,
            f"$r^2$ = {rsquared:.3f}",
            fontsize=12,
            ha="left",
            va="top",
            transform=plt.gca().transAxes,
        )

        # Print fitted value just below R^2, automated by fix_param
        if fix_param.lower() == "tau_r":
            plt.text(
                0.01,
                0.91,
                f"Fitted $\\tau_{{\\mathrm{{E}}}}$: {tau_E_fit:.3e} s",
                fontsize=12,
                ha="left",
                va="top",
                transform=plt.gca().transAxes,
            )
        elif fix_param.lower() == "tau_e":
            plt.text(
                0.01,
                0.91,
                f"Fitted $\\tau_{{\\mathrm{{R}}}}$: {tau_R_fit:.3e} s",
                fontsize=12,
                ha="left",
                va="top",
                transform=plt.gca().transAxes,
            )
        elif not fix_param or fix_param in ["none", ""]:
            plt.text(
                0.01,
                0.91,
                (
                    f"Fitted $\\tau_{{\\mathrm{{R}}}}$: {tau_R_fit:.3e} s\n"
                    f"Fitted $\\tau_{{\\mathrm{{E}}}}$: {tau_E_fit:.3e} s"
                ),
                fontsize=12,
                ha="left",
                va="top",
                transform=plt.gca().transAxes,
            )

        plt.tight_layout()
        plt.savefig(os.path.join(config.project_name, "experimental_vs_fitted_R1.png"))
        plt.show()

        plt.figure(figsize=(8, 5))
        # circles for experiment
        plt.plot(chem_labels, exp_r1, "o", label="Experimental R1")
        # squares for theory
        plt.plot(chem_labels, theory_r1, "s", label="Fitted Theory R1")
        plt.plot(
            chem_labels, theory_r1, "x", color="red", label="Theory X"
        )  # X marker for theory

        plt.xlabel("Chemical Label")
        plt.ylabel("R1 (s$^{-1}$)")
        plt.title("Experimental vs Fitted R1")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config.project_name, "r1_fit_comparison.png"))
        plt.show()

    else:
        raise ValueError(
            "fit_corr_time and relaxation_model must be specified in the input file."
        )


def plot_shift_tdep_func(uargs):
    """
    Plot temperature dependence of chemical shifts from experimental datasets.

    This CLI handler loads one or more experimental CSV files and generates
    shift-versus-temperature and shift-versus-inverse-temperature plots.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    experiments = main.Experiment.from_file(uargs.experiment_files)

    vis.plot_shift_tdep(
        experiments,
        "ShiftT_vs_T",
        show=True,
        save=True,
        save_name=f"shift_x_T_vs_T{PFF}",
    )

    vis.plot_shift_tdep(
        experiments,
        "Shift_vs_1/T",
        show=True,
        save=True,
        save_name=f"shift_vs_T-1{PFF}",
    )

    return


def read_args(arg_list=None):
    """
    Parse CLI arguments and dispatch to the selected subcommand handler.

    Args:
        arg_list (list[str] | None): Optional argument list to parse (used for testing).
            If None, arguments are read from `sys.argv`.

    Returns:
        argparse.Namespace: Parsed arguments object
        (after `args.func(args)` is executed).
    """

    description = """
    A package for fitting susceptibility tensors from paramagnetic NMR data
    """

    epilog = (
        "To display options for a specific program, use\n\n"
        f"      {ut.cstr('simpnmr SUBPROGRAM -h', 'green')}"
    )

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show only errors",
    )

    parser._positionals.title = "Subprograms"

    subparsers = parser.add_subparsers(dest="prog")

    extract_dia = subparsers.add_parser(
        "extract_dia",
        description="Extract diamagnetic shifts from quantum chemistry output",
    )
    extract_dia.set_defaults(func=extract_dia_func)

    extract_dia.add_argument(
        "output_file",
        type=str,
        help=("Quantum Chemistry output file containing chemical shift information"),
    )

    extract_dia.add_argument(
        "--ref_output_file",
        metavar="ref_output_file",
        default="",
        type=str,
        help=(
            "Quantum Chemistry output file containing reference "
            "chemical shift information"
        ),
    )

    fit_susc = subparsers.add_parser(
        "fit_susc",
        description=(
            "Fit susceptibility tensor using DFT hyperfines and experimental peaks"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    fit_susc.set_defaults(func=fit_susc_func)

    fit_susc.add_argument(
        "input_file",
        type=str,
        help=("Input file for fit_susc -- see documentation for format"),
    )

    fit_susc.add_argument(
        "--dry_run",
        action="store_true",
        help=("Checks input file, loads data, but quits before simulation"),
    )

    fit_susc.add_argument(
        "--pcs_isosurface",
        action="store_true",
        help=("Saves PCS isosurface for each temperature to separate cube files"),
    )

    fit_susc.add_argument(
        "--susc_units",
        "-su",
        choices=["cm3 mol-1", "A3"],
        metavar="<str>",
        type=str,
        default="A3",
        help=('Controls susceptibility units of plots \n(wrap with "")\nDefault: A3'),
    )

    fit_susc.add_argument(
        "--shift_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot Experimental and Calculated Chemical shifts "
            "against each other \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: save"
        ),
    )

    fit_susc.add_argument(
        "--spread_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot spread of contributions to calculated shifts \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: save"
        ),
    )

    fit_susc.add_argument(
        "--contrib_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot mean of contributions to mean calculated shifts \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: on"
        ),
    )

    fit_susc.add_argument(
        "--isoaxrho_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot Isotropic, Axial, and Rhombic susceptibility "
            "as a function of temperature \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: save"
        ),
    )

    extract_a = subparsers.add_parser(
        "extract_a",
        description=("Extract A tensor information from quantum chemistry output"),
    )
    extract_a.set_defaults(func=extract_a_func)

    extract_a.add_argument(
        "calculation_data",
        type=str,
        help=("Gaussian log file, or Orca output or property file"),
    )

    plot_a = subparsers.add_parser(
        "plot_a",
        description=("Plot A tensor information from quantum chemistry output"),
    )
    plot_a.set_defaults(func=plot_a_func)

    plot_a.add_argument(
        "calculation_data",
        type=str,
        help=("Gaussian log file, or Orca output or property file"),
    )

    plot_a.add_argument(
        "components",
        choices=[
            "iso",
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
            "dxx",
            "dxy",
            "dxz",
            "dyx",
            "dyy",
            "dyz",
            "dzx",
            "dzy",
            "dzz",
            "ax",
            "rho",
        ],
        nargs="+",
        help=("Component() to plot"),
    )

    plot_a.add_argument("--chem_labels", type=str, help=("chemical label file (.csv)"))

    plot_a.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default="all",
        help=("Elements to include in plot"),
    )

    plot_a.add_argument("--save", action="store_false", help=("Save plot to file"))

    plot_a.add_argument(
        "--hide_plots", action="store_true", help=("Display plot on screen")
    )

    plot_a_iso = subparsers.add_parser(
        "plot_a_iso_ax",
        description=("Plot A tensor information from quantum chemistry output"),
    )

    plot_a_iso.set_defaults(func=plot_a_iso_ax_func)

    plot_a_iso.add_argument("input_file", type=str, help=("simpnmr Input file"))

    plot_a_iso.add_argument("--save", action="store_true", help=("Save plot to file"))

    plot_a_iso.add_argument(
        "--hide_plots", action="store_true", help=("Display plot on screen")
    )

    calc_pdip = subparsers.add_parser(
        "calc_pdip",
        description=(
            "Calculate dipolar Hyperfine tensor using point dipole approximation"
        ),
    )
    calc_pdip.set_defaults(func=calc_pdip_func)

    calc_pdip.add_argument(
        "structure_file",
        type=str,
        help=("File containing molecular structure: .xyz, .log, ORCA .out"),
    )

    calc_pdip.add_argument(
        "centres",
        type=str,
        nargs="+",
        help=("Atomic label (with index number) of paramagnetic centre(s)"),
    )

    calc_pdip.add_argument(
        "--chem_labels", type=str, help=("chemical label file (.csv)")
    )

    calc_pdip.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default="all",
        help=("Elements to include in plot"),
    )

    calc_pdip.add_argument(
        "--plot_components",
        default=[],
        choices=[
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
            "x",
            "y",
            "z",
            "ax",
            "rho",
        ],
        nargs="+",
        help=("Component(s) to plot"),
    )

    calc_pcs_iso = subparsers.add_parser(
        "calc_pcs_iso",
        description=("Calculates PCS isosurface and saves to .cube file"),
    )
    calc_pcs_iso.set_defaults(func=calc_pcs_iso_func)

    calc_pcs_iso.add_argument("susc_file", help="File containing susceptibility data")

    calc_pcs_iso.add_argument(
        "susc_format",
        help="Susceptibility file format",
        choices=["csv", "orca_nev", "orca_cas", "molcas"],
    )

    calc_pcs_iso.add_argument(
        "temperatures",
        nargs="+",
        help="Temperatures for which to produce pcs isosurface plot",
        type=float,
    )

    calc_pcs_iso.add_argument(
        "structure_file",
        help=(
            "File containing molecular structure"
            "Either Gaussian .log, ORCA .out, or plain old .xyz"
        ),
    )

    calc_pcs_iso.add_argument(
        "central_atom",
        help="Atom on which isosurface is centered. Must include indexing",
    )

    predict = subparsers.add_parser(
        "predict", description="Calculate shifts using Hyperfine and Susceptibility"
    )
    predict.set_defaults(func=predict_func)

    predict.add_argument(
        "input_file",
        type=str,
        help=("Input file for predict - see documentation for format"),
    )

    predict.add_argument(
        "--susc_units",
        "-su",
        choices=["cm3 mol-1", "A3"],
        metavar="<str>",
        type=str,
        default="A3",
        help=(
            "Controls susceptibility units of plots and output files \n"
            '(wrap with "")\n'
            "Default: A3"
        ),
    )

    plot_shift_tdep = subparsers.add_parser(
        "plot_shift_tdep",
        description="Calculate shifts using Hyperfine and Susceptibility",
    )
    plot_shift_tdep.set_defaults(func=plot_shift_tdep_func)

    plot_shift_tdep.add_argument(
        "experiment_files", type=str, nargs="+", help=("simpnmr experiment.csv files")
    )

    fit_corr_time = subparsers.add_parser(
        "fit_corr_time", description="Fit correlation times using experimental R1 data"
    )
    fit_corr_time.set_defaults(func=fit_corr_time_func)

    fit_corr_time.add_argument(
        "input_file",
        type=str,
        help="Input file for fit_corr_time -- see documentation for format",
    )

    # Read sub-parser and parse arguments
    parser.set_defaults(func=lambda args: parser.print_help())
    args = parser.parse_args(arg_list)

    return args


def interface(argv=None):
    args = read_args(argv)

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        return args.func(args)
    except ValueError as err:
        logger.error("%s", err)
        raise SystemExit(1) from None
    except FileNotFoundError as err:
        logger.error("File not found: %s", err.filename)
        raise SystemExit(1) from None
