# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Predict paramagnetic shifts and spectra from input data.

Loads molecular, susceptibility, and experimental inputs, computes shifts, and
writes tables and plots for selected temperatures.
"""

import copy
import logging
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.application.loaders.chem_labels import load_chem_labels_from_csv
from simpnmr.application.loaders.diamagnetic import load_diamagnetic_shifts
from simpnmr.application.loaders.electronic_state import load_electronic_state
from simpnmr.application.loaders.experiment import load_experiments
from simpnmr.application.loaders.molecule import load_molecule_from_csv
from simpnmr.application.loaders.susceptibility import load_susceptibilities
from simpnmr.application.setup.options import PredictRunOptions
from simpnmr.config import config as cfg
from simpnmr.core.constants.gammas import NUCLEAR_GAMMAS
from simpnmr.core.constants.physics import EGAMMA
from simpnmr.core.domain.molecule import Molecule
from simpnmr.core.factories.molecule import build_molecule_from_qca
from simpnmr.core.factories.susc import get_g_corr_iso_susc, get_spin_only_susc
from simpnmr.core.relaxation import gueron, sbm
from simpnmr.core.utils.strings import remove_numbers
from simpnmr.io.csv import relaxation, susceptibility
from simpnmr.io.csv.molecule import save_molecule_to_csv
from simpnmr.io.csv.spectrum import read_spectrum
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.io.xyz import xyz
from simpnmr.tools.coords_tools import transform as tfm
from simpnmr.tools.coords_tools import xyz_format as xyzf
from simpnmr.viz.plots.shifts import plot_shift_contrib, plot_shift_spread
from simpnmr.viz.plots.spectrum_1d import plot_pred_spectrum, plot_raw_deconv_pred

logger = logging.getLogger(__name__)


def run_predict(
    config: cfg.PredictConfig,
    options: PredictRunOptions | None = None,
) -> int:
    """Run PCS prediction from a YAML configuration file.

    Args:
        config: Prediction configuration loaded from YAML.

    Returns:
        Exit code: 0 on success.
    """

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    if options is None:
        raise ValueError("PredictRunOptions is required")

    delimiter = options.runtime.csv_delimiter

    # Either load hyperfines from DFT output file
    if config.hyperfine_method == "dft":
        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)
        # Write raw calculation data to output file
        qc_hyperfine_data.save_to_csv(
            os.path.join(config.project_name, "dft_hyperfines.csv"),
            verbose=True,
            delimiter=delimiter,
            comment=f"# Data taken from file {config.hyperfine_file}",
        )

        # Create molecule object from quantum chemical hyperfine data
        # Retain only the atoms that are given in the labels file
        base_molecule = build_molecule_from_qca(
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
                f"{os.path.splitext(config.hyperfine_file)[1]} unsupported"
            )

        # Create molecule
        base_molecule = Molecule.from_labels_coords(
            labels, coords, elements=config.nuclei_include
        )

        # Calculate point dipole hyperfine
        base_molecule.calc_pdip(config.hyperfine_pdip_centres)

    # or load from CSV
    elif config.hyperfine_method == "csv":
        base_molecule = load_molecule_from_csv(
            config.hyperfine_file,
            elements=config.nuclei_include,
        )

    # Load electronic state
    base_molecule.electronic = load_electronic_state(
        spin_S=config.spin_S,
        orbit_L=config.orbit,
        total_J=config.total_momentum_J,
        hyperfine_file=config.hyperfine_file if config.spin_S is None else None,
        hyperfine_method=config.hyperfine_method if config.spin_S is None else None,
    )
    spin = base_molecule.electronic.spin_S

    # Add chemical labels
    if len(config.chem_labels_file):
        al_to_cl, al_to_cml = load_chem_labels_from_csv(config.chem_labels_file)
        base_molecule.apply_chem_labels(al_to_cl, al_to_cml)

        # Save xyz file with chemical labels for chemcraft
        xyz.save_chemcraft_xyz(
            file_name=os.path.join(config.project_name, "chemcraft_structure.xyz"),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
            chem_labels={nuc.label: nuc.chem_label for nuc in base_molecule.nuclei},
        )

    # Save xyz file with chemical labels for chemcraft
    xyz.save_xyz(
        file_name=os.path.join(config.project_name, "structure.xyz"),
        labels=base_molecule.labels,
        coords=base_molecule.coords,
        comment=f"Structure from {config.hyperfine_file}",
    )

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

    # Rotationally average hyperfines of user selected nuclei:
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Rotate hyperfine tensors from DFT frame into chi eigenframe (if provided)
    if "orca" in config.susceptibility_format:
        if (
            "dft" in config.hyperfine_method
        ):  # TODO: remove condition and remove config from tf.
            rot_mat, trans_mat = tfm.get_rotation_and_transformation(config)
            base_molecule.rotate_hyperfines(rot_mat)

            # Rotate HFC coords frame into chi eigenframe and save the transformed coords
            tfm.rotate_coords_to_chi_frame(config.project_name, config)

    # Load susceptibility information
    suscs = load_susceptibilities(
        config.susceptibility_file,
        config.susceptibility_format,
    )

    suscs = [
        susc for susc in suscs if susc.temperature in config.susceptibility_temperatures
    ]

    if not suscs:
        raise ValueError("No susceptibility data found for specified temperature(s)")

    # Calculate linewidths using user-specified relaxation model (optional)
    if not getattr(config, "relaxation_model", None):
        (
            logger.warning(
                "No relaxation model specified — linewidths will be fixed at 1 ppm."
            )
        )
    elif config.relaxation_magnetic_field_tesla is None:
        logger.warning(
            "relaxation_magnetic_field_tesla "
            "not provided — relaxation effects skipped, "
            "linewidths will be fixed at 1 ppm \n"
        )
    else:
        _apply_relaxation_linewidths(config, base_molecule)

    # Load experimental data from file into list of experiment objects
    if len(config.experiment_files):
        experiments = load_experiments(config.experiment_files)
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
            spectrum_array = read_spectrum(spectrum)
            experiment.spectrum = spectrum_array

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
            susc.iso = get_g_corr_iso_susc(
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
            susc.iso = get_spin_only_susc(
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
        plot_shift_spread(
            molecule,
            experiment=experiment,
            save=True,
            show=False,
            terms=_terms,
            save_name=os.path.join(
                config.project_name,
                f"pred_shift_spread_{molecule.susc.temperature:.2f}_K",
            ),
            verbose=True,
            window_title=f"Spread of predicted shifts at {susc.temperature:.2f} K",
            order="descending",
        )

        # Bar chart for means
        plot_shift_contrib(
            molecule,
            experiment=experiment,
            save=True,
            show=False,
            save_name=os.path.join(
                config.project_name,
                f"pred_mean_components_{molecule.susc.temperature:.2f}_K",
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
            plot_raw_deconv_pred(
                molecule=molecule,
                isotope=molecule.nuclei[0].isotope,
                shift_range=shift_range,
                experiment=experiment,
                save=True,
                show=False,
                save_name=os.path.join(
                    config.project_name,
                    f"pred_and_exp_spectrum_{molecule.susc.temperature:.2f}_K",
                ),
            )
        plot_pred_spectrum(
            molecule,
            isotope=molecule.nuclei[0].isotope,
            shift_range=shift_range,
            save=True,
            show=False,
            save_name=os.path.join(
                config.project_name,
                f"pred_spectrum_{molecule.susc.temperature:.2f}_K",
            ),
        )

        plt.show()

        plt.close("all")

    # TODO If more than one temperature, then make a stacked plot of spectra

    # Save susceptibility data to file
    susceptibility.save_susc(
        molecules,
        os.path.join(config.project_name, "susceptibility_tensor.csv"),
        comment="#Data from {} ({})".format(
            config.susceptibility_file, config.susceptibility_format
        ),
        susc_units=getattr(config, "susc_units", "A3"),
    )

    # Write shift data to file
    for molecule in molecules:
        save_molecule_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                f"hyperfines_and_shifts_{molecule.susc.temperature:.2f}_K.csv",
            ),
            delimiter=delimiter,
            comment=f"# T = {molecule.susc.temperature:.2f} K",
            verbose=True,
        )

    return 0


def _apply_relaxation_linewidths(config: cfg.PredictConfig, base_molecule: Molecule):
    """
    Apply linewidths using a user-specified relaxation model.

    This function updates `base_molecule.nuclei` in-place by setting `nuc.shift.lw`
    when relaxation inputs are provided in the config.

    Args:
        config (PredictConfig): Prediction configuration containing relaxation
            settings and physical parameters.
        base_molecule (Molecule): Molecule instance to update in-place.

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
        if remove_numbers(nuc.label) in nuclei_labels
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
        label: NUCLEAR_GAMMAS[remove_numbers(label)] * 2 * np.pi * 1e6
        for label in nuclei_coords
    }
    omega_I_dict = {label: gamma_I_dict[label] * B0 for label in nuclei_coords}
    omega_S = EGAMMA * B0 * 2 * np.pi * 1e6
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
        sbm_dipolar_r1_rates = sbm.calc_r1_dipolar(
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
        sbm_contact_r1_rates = sbm.calc_r1_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e2,
            spin,
            total_momentum_J,
        )
        # Calculate SBM dipolar rates (R2)
        sbm_dipolar_r2_rates = sbm.calc_r2_dipolar(
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
        sbm_contact_r2_rates = sbm.calc_r2_contact(
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
        curie_r1_rates = gueron.calc_r1_curie(
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
        curie_r2_rates = gueron.calc_r2_curie(
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
        sbm_dipolar_r1_rates = sbm.calc_r1_dipolar(
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
        sbm_contact_r1_rates = sbm.calc_r1_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e1,
            spin,
            total_momentum_J,
        )
        sbm_dipolar_r2_rates = sbm.calc_r2_dipolar(
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
        sbm_contact_r2_rates = sbm.calc_r2_contact(
            list(nuclei_coords.keys()),
            A_iso_dict,
            omega_I_dict,
            omega_S,
            tau_e1,
            tau_e2,
            spin,
            total_momentum_J,
        )

        curie_r1_rates = gueron.calc_r1_curie(
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
        curie_r2_rates = gueron.calc_r2_curie(
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
    relaxation.save_relaxation_decomposition(
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
