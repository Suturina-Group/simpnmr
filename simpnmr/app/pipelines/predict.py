# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Predict paramagnetic shifts and spectra from input data.

Loads molecular, susceptibility, and experimental inputs, computes shifts, and
writes tables and plots for selected temperatures.
"""

import copy
import logging
import os
from pathlib import Path

import numpy as np

# Application layer
from simpnmr.app.loaders.dia_load import load_diamagnetic_shifts
from simpnmr.app.loaders.elstate_load import load_electronic_state
from simpnmr.app.loaders.exp_load import load_experiments
from simpnmr.app.loaders.hfc_load import load_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_base_molecule
from simpnmr.app.loaders.paramag_centre_load import load_paramagnetic_centre
from simpnmr.app.loaders.sh_load import (
    load_g_tensor_ab_initio,
    load_g_tensor_dft,
)
from simpnmr.app.loaders.susc_load import load_susceptibilities
from simpnmr.app.params.options import PredictRunOptions
from simpnmr.app.policies.hfc import has_missing_selected_chem_labels
from simpnmr.app.policies.linewidth import resolve_output_linewidths
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.core.phys.susc import get_spin_only_susc
from simpnmr.app.policies.relax import resolve_relaxation_conditions
from simpnmr.app.policies.susc import resolve_susceptibility_source

# Core / domain
from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.conv.ang_to_freq import angstrom_to_mhz
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.relaxation.eval import evaluate_relaxation_rates

# Tools
from simpnmr.core.util import transform as tfm
from simpnmr.core.util.strings import remove_numbers

# IO layer
from simpnmr.io.csv.mol import save_molecule_to_csv
from simpnmr.io.csv.peaks import save_peak_data_to_csv
from simpnmr.io.csv.spec import read_spectrum
from simpnmr.io.csv.susc import save_susc
from simpnmr.io.qc.backends.orca.geom import read_orca5_output_xyz  # TODO: remove
from simpnmr.io.xyz import xyz_write

# Visualisation
from simpnmr.viz.plots.corr_time import plot_corr_time_contrib
from simpnmr.viz.plots.orb_dep import plot_orbital_shift_distance_dependence
from simpnmr.viz.plots.shifts import plot_shift_contrib, plot_shift_spread
from simpnmr.viz.plots.spect import plot_pred_spectrum, plot_raw_deconv_pred
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_predict(config, options: PredictRunOptions | None = None) -> int:
    """Run pNMR prediction from a YAML configuration file.

    Args:
        config: Prediction configuration loaded from YAML.

    Returns:
        Exit code: 0 on success.
    """

    # TODO(policy): introduce a unified InputSpec policy (file backend/format/section)
    #               to resolve HFC/SH/Susceptibility readers in one place and avoid
    #               duplicated format detection across pipelines and loaders.

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    if options is None:
        raise ValueError("PredictRunOptions is required")

    # Build the resolved plotting contract for this run.
    spec = apply_profile(options.runtime.plot_profile)

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

    # Load ab-initio g-tensor and derived scalar components (if available)
    base_molecule = load_g_tensor_ab_initio(
        molecule=base_molecule,
        susceptibility_file=config.susceptibility_file,
        susceptibility_format=config.susceptibility_format,
    )

    # Load Hyperfines
    base_molecule = load_hyperfines(
        molecule=base_molecule,
        config=config,
    )

    # Load Electronic State
    base_molecule.electronic = load_electronic_state(
        spin_S=config.spin_S,
        orbit_L=config.orbit,
        total_J=config.total_momentum_J,
        hyperfine_file=config.hyperfine_file if config.spin_S is None else None,
        hyperfine_method=config.hyperfine_method if config.spin_S is None else None,
    )

    # Resolve susceptibility source for downstream operations when an explicit
    # susceptibility file is available.
    backend, section = None, None
    if config.susceptibility_file is not None:
        backend, section = resolve_susceptibility_source(
            config.susceptibility_file,
            config.susceptibility_format,
        )

    # TODO(app): Temporary ORCA-only chi-source geometry load for prediction.
    # Move this into the appropriate loader/builder layer once the chi-source
    # geometry flow is formalized outside the pipeline.
    if backend == "orca":
        try:
            base_molecule.chi_source_labels, base_molecule.chi_source_coords = (
                read_orca5_output_xyz(config.susceptibility_file)
            )
        except Exception as exc:
            logger.warning(
                "Failed to read chi-source geometry from ORCA susceptibility file: %s",
                exc,
            )

    # Build magnetic susceptibility objects.
    if getattr(config, "susceptibility_method", None) == "spin_only":
        # Spin-only path: no file required. Build isotropic-only Susceptibility
        # objects directly from quantum numbers (S, L, J) using the Curie law.
        # This gives Fermi contact shifts only (isotropic A × isotropic χ).
        spin = base_molecule.electronic.spin_S
        orbit = base_molecule.electronic.orbit_L
        total_J = base_molecule.electronic.total_J
        suscs = []
        for T in config.susceptibility_temperatures:
            chi_iso = get_spin_only_susc(spin, orbit, total_J, T)
            tensor = np.eye(3) * chi_iso
            suscs.append(Susceptibility(tensor=tensor, temperature=T))
        logger.info(
            "Spin-only susceptibility built for %d temperature(s) "
            "(S=%.1f, L=%.1f, J=%.1f)",
            len(suscs),
            spin,
            orbit,
            total_J if total_J is not None else float("nan"),
        )
        if config.hyperfine_method == "pdip":
            logger.warning(
                "susceptibility:method spin_only produces an isotropic susceptibility "
                "tensor, so only the Fermi contact shift (A_iso × χ_iso) contributes. "
                "The point-dipole hyperfine model (pdip) gives A_iso = 0, so all "
                "predicted paramagnetic shifts will be zero. "
                "Provide a contact hyperfine file (e.g. from DFT) and set "
                "hyperfine:method to 'qc' or 'pdip+fc'."
            )
    else:
        suscs = load_susceptibilities(
            config.susceptibility_file,
            config.susceptibility_format,
            electronic=base_molecule.electronic,
            g_tensor=base_molecule.sh.g_tensor_ab_initio,
        )
        suscs = [
            susc for susc in suscs if susc.temperature in config.susceptibility_temperatures
        ]

    if not suscs:
        raise ValueError("No susceptibility data found for specified temperature(s)")

    # Load chemical labels
    if len(config.chem_labels_file):
        al_to_cl, al_to_cml, _ = load_chem_labels_from_csv(
            config.chem_labels_file
        )
        if has_missing_selected_chem_labels(base_molecule, al_to_cl):
            logger.warning(
                "Chemical labels file does not define labels for all selected nuclei; "
                "missing labels will use atom labels."
            )
        base_molecule.apply_chem_labels(al_to_cl, al_to_cml)

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

    # Load experimental data from file into list of experiment objects
    if len(config.experiment_files):
        experiments = load_experiments(config.experiment_files)
        for susc, exp in zip(suscs, experiments):
            if susc.temperature != exp.temperature:
                logger.warning(
                    "Mismatch in Susceptibility (%.2f K) and "
                    "Experimental (%.2f K) temperatures. "
                    "Proceeding with susceptibility temperature (%.2f K) "
                    "as the active calculation temperature.",
                    susc.temperature,
                    exp.temperature,
                    susc.temperature,
                )
    else:
        experiments = [None] * len(suscs)

    if len(config.experiment_spectrum_files):
        for experiment, spectrum in zip(experiments, config.experiment_spectrum_files):
            if experiment is None:
                continue
            spectrum_array = read_spectrum(spectrum)
            experiment.exp_reference = config.experiment_exp_reference
            experiment.spectrum = spectrum_array

    # Rotationally average hyperfines of user selected nuclei:
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # TODO(app): Temporary ORCA-specific chi-frame wiring.
    # Move chi-source geometry loading, alignment susceptibility selection, and
    # frame preparation out of predict.py into the appropriate loader/builder
    # layer once chi-frame domain flow is formalized.

    # Use the first loaded susceptibility for chi-frame rotation
    rotation_susc = suscs[0]

    # Rotate hyperfine tensors from DFT frame into chi eigenframe (if provided)
    if backend == "orca":
        if "dft" in config.hyperfine_method:
            if base_molecule.chi_source_coords is None:
                logger.warning(
                    "Chi-source geometry is unavailable; skipping chi-frame rotation"
                )
            else:
                rot_mat, _ = tfm.get_rotation_and_transformation(
                    chi_tensor=rotation_susc.tensor,
                    temperature=rotation_susc.temperature,
                    chi_source_coords=base_molecule.chi_source_coords,
                    dft_coords=base_molecule.coords,
                )
                base_molecule.apply_frame_rotation(rot_mat)

                # Rotate DFT g-tensor into chi frame
                if base_molecule.sh.g_tensor_dft is not None:
                    base_molecule.sh.g_tensor_dft = (
                        rot_mat @ base_molecule.sh.g_tensor_dft @ rot_mat.T
                    )

                # Rotate chi-source coords into chi eigenframe and save the
                # transformed susceptibility-source structure.
                if base_molecule.chi_source_labels is not None:
                    tfm.rotate_coords_to_chi_frame(
                        config.project_name,
                        chi_tensor=rotation_susc.tensor,
                        chi_source_labels=base_molecule.chi_source_labels,
                        chi_source_coords=base_molecule.chi_source_coords,
                    )

    _terms = ["pc", "fc", "d"]

    if config.hyperfine_method == "pdip":
        _terms.pop(_terms.index("fc"))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index("d"))

    # Create a molecule object which accompanies each experiment object
    molecules = [copy.deepcopy(base_molecule) for _ in range(len(experiments))]
    linewidth_outputs = []

    # Update susceptibility tensor of Molecule using model
    for molecule, susc, experiment in zip(molecules, suscs, experiments):
        molecule.susc = susc

        # Apply relaxation linewidth (relaxation-aware when inputs are available).
        _apply_relaxation_linewidths(config, molecule, experiment)

        # Calculate shifts using new susceptibility tensor and rotated hyperfines
        molecule.calculate_shifts()

        # Calculate average shifts
        molecule.average_shifts()

        unique_isotopes = sorted({nuc.isotope for nuc in molecule.nuclei})
        _iso_suffix = len(unique_isotopes) > 1

        for iso in unique_isotopes:
            iso_nuclei = [nuc for nuc in molecule.nuclei if nuc.isotope == iso]
            iso_mol = copy.copy(molecule)
            iso_mol.nuclei = iso_nuclei
            _suffix = f"_{iso}" if _iso_suffix else ""

            # Build chem_label -> math_label mapping for axis labels
            _cl_to_ml = {
                nuc.chem_label: nuc.chem_math_label
                for nuc in iso_nuclei
                if nuc.chem_label and nuc.chem_math_label
            }

            # Plot R1 decomposition if relaxation was computed
            if molecule.relaxation is not None and molecule.relaxation.r1 is not None:
                r1_channels = molecule.relaxation.r1
                # Average per-atom R1 channels into per-chem-label values
                _cl_total: dict[str, list] = {}
                _cl_dipolar: dict[str, list] = {}
                _cl_contact: dict[str, list] = {}
                _cl_curie: dict[str, list] = {}
                for nuc in iso_nuclei:
                    cl = nuc.chem_label
                    if r1_channels.total and nuc.label in r1_channels.total:
                        _cl_total.setdefault(cl, []).append(r1_channels.total[nuc.label])
                    if r1_channels.dipolar and nuc.label in r1_channels.dipolar:
                        _cl_dipolar.setdefault(cl, []).append(r1_channels.dipolar[nuc.label])
                    if r1_channels.contact and nuc.label in r1_channels.contact:
                        _cl_contact.setdefault(cl, []).append(r1_channels.contact[nuc.label])
                    if r1_channels.curie and nuc.label in r1_channels.curie:
                        _cl_curie.setdefault(cl, []).append(r1_channels.curie[nuc.label])

                _chem_labels = sorted(_cl_total.keys())
                _theory_r1 = np.array([np.mean(_cl_total[cl]) for cl in _chem_labels])
                _theory_dipolar = (
                    np.array([np.mean(_cl_dipolar.get(cl, [0.0])) for cl in _chem_labels])
                    if _cl_dipolar else None
                )
                _theory_contact = (
                    np.array([np.mean(_cl_contact.get(cl, [0.0])) for cl in _chem_labels])
                    if _cl_contact else None
                )
                _theory_curie = (
                    np.array([np.mean(_cl_curie.get(cl, [0.0])) for cl in _chem_labels])
                    if _cl_curie else None
                )
                # Use experimental R1 if available
                _exp_r1_dict = {
                    sig.assignment: float(sig.r1)
                    for sig in experiment.signals
                    if sig.r1 is not None and not np.isnan(float(sig.r1))
                } if experiment is not None else {}
                _exp_r1 = (
                    np.array([_exp_r1_dict[cl] for cl in _chem_labels if cl in _exp_r1_dict])
                    if _exp_r1_dict and all(cl in _exp_r1_dict for cl in _chem_labels)
                    else None
                )

                with spec.context():
                    plot_corr_time_contrib(
                        theory_r1=_theory_r1,
                        theory_r1_dipolar=_theory_dipolar,
                        theory_r1_contact=_theory_contact,
                        theory_r1_curie=_theory_curie,
                        exp_r1=_exp_r1,
                        chem_labels=[_cl_to_ml.get(cl, cl) for cl in _chem_labels],
                        spec=spec,
                        save=True,
                        show=options.runtime.show_plots,
                        save_name=os.path.join(
                            config.project_name,
                            f"pred_r1_decomposition_{susc.temperature:.2f}_K{_suffix}",
                        ),
                        verbose=True,
                    )

            # Plot linewidth (R2/π) decomposition if relaxation was computed
            if molecule.relaxation is not None and molecule.relaxation.r2 is not None:
                r2_channels = molecule.relaxation.r2
                _cl_total_r2: dict[str, list] = {}
                _cl_dipolar_r2: dict[str, list] = {}
                _cl_contact_r2: dict[str, list] = {}
                _cl_curie_r2: dict[str, list] = {}
                for nuc in iso_nuclei:
                    cl = nuc.chem_label
                    if r2_channels.total and nuc.label in r2_channels.total:
                        _cl_total_r2.setdefault(cl, []).append(r2_channels.total[nuc.label] / np.pi)
                    if r2_channels.dipolar and nuc.label in r2_channels.dipolar:
                        _cl_dipolar_r2.setdefault(cl, []).append(r2_channels.dipolar[nuc.label] / np.pi)
                    if r2_channels.contact and nuc.label in r2_channels.contact:
                        _cl_contact_r2.setdefault(cl, []).append(r2_channels.contact[nuc.label] / np.pi)
                    if r2_channels.curie and nuc.label in r2_channels.curie:
                        _cl_curie_r2.setdefault(cl, []).append(r2_channels.curie[nuc.label] / np.pi)

                _chem_labels_r2 = sorted(_cl_total_r2.keys())
                _theory_lw = np.array([np.mean(_cl_total_r2[cl]) for cl in _chem_labels_r2])
                _theory_lw_dipolar = (
                    np.array([np.mean(_cl_dipolar_r2.get(cl, [0.0])) for cl in _chem_labels_r2])
                    if _cl_dipolar_r2 else None
                )
                _theory_lw_contact = (
                    np.array([np.mean(_cl_contact_r2.get(cl, [0.0])) for cl in _chem_labels_r2])
                    if _cl_contact_r2 else None
                )
                _theory_lw_curie = (
                    np.array([np.mean(_cl_curie_r2.get(cl, [0.0])) for cl in _chem_labels_r2])
                    if _cl_curie_r2 else None
                )
                _exp_lw_dict = {
                    sig.assignment: float(sig.width)
                    for sig in experiment.signals
                    if sig.width is not None
                } if experiment is not None else {}
                _exp_lw = (
                    np.array([_exp_lw_dict[cl] for cl in _chem_labels_r2])
                    if _exp_lw_dict and all(cl in _exp_lw_dict for cl in _chem_labels_r2)
                    else None
                )

                with spec.context():
                    plot_corr_time_contrib(
                        theory_r1=_theory_lw,
                        theory_r1_dipolar=_theory_lw_dipolar,
                        theory_r1_contact=_theory_lw_contact,
                        theory_r1_curie=_theory_lw_curie,
                        exp_r1=_exp_lw,
                        chem_labels=[_cl_to_ml.get(cl, cl) for cl in _chem_labels_r2],
                        spec=spec,
                        ylabel=r"Linewidth (Hz)",
                        save=True,
                        show=options.runtime.show_plots,
                        save_name=os.path.join(
                            config.project_name,
                            f"pred_linewidth_decomposition_{susc.temperature:.2f}_K{_suffix}",
                        ),
                        verbose=True,
                    )

            # Plot theoretical shifts
            with spec.context():
                # Spread
                plot_shift_spread(
                    iso_mol,
                    experiment=experiment,
                    spec=spec,
                    save=True,
                    show=options.runtime.show_plots,
                    terms=_terms,
                    save_name=os.path.join(
                        config.project_name,
                        f"pred_shift_spread_{molecule.susc.temperature:.2f}_K{_suffix}",
                    ),
                    verbose=True,
                    window_title=f"Spread of predicted shifts at {susc.temperature:.2f} K",
                    order="descending",
                )

                # Bar chart for means
                plot_shift_contrib(
                    iso_mol,
                    experiment=experiment,
                    spec=spec,
                    save=True,
                    show=options.runtime.show_plots,
                    save_name=os.path.join(
                        config.project_name,
                        f"pred_mean_components_{molecule.susc.temperature:.2f}_K{_suffix}",
                    ),
                    verbose=True,
                    window_title=(
                        f"Predicted mean shifts and components at {susc.temperature:.2f} K"
                    ),
                    order="descending",
                )

                if molecule.metadata.get("hyperfine", {}).get("orbital_contribution") == (
                    "available"
                ):
                    plot_orbital_shift_distance_dependence(
                        iso_mol,
                        spec=spec,
                        save=True,
                        show=options.runtime.show_plots,
                        save_name=os.path.join(
                            config.project_name,
                            f"pred_orbital_distance_dependence_{molecule.susc.temperature:.2f}_K{_suffix}",
                        ),
                        verbose=True,
                        window_title=(
                            f"Orbital shift distance dependence at {susc.temperature:.2f} K"
                        ),
                        order="ascending",
                    )

            shift_range = [
                np.min([nuc.shift.avg for nuc in iso_nuclei]),
                np.max([nuc.shift.avg for nuc in iso_nuclei]),
            ]

            with spec.context():
                if len(config.experiment_files):
                    plot_raw_deconv_pred(
                        molecule=iso_mol,
                        isotope=iso,
                        shift_range=shift_range,
                        experiment=experiment,
                        spec=spec,
                        save=True,
                        show=options.runtime.show_plots,
                        save_name=os.path.join(
                            config.project_name,
                            f"pred_and_exp_spectrum_{molecule.susc.temperature:.2f}_K{_suffix}",
                        ),
                    )

                plot_pred_spectrum(
                    iso_mol,
                    isotope=iso,
                    shift_range=shift_range,
                    spec=spec,
                    save=True,
                    show=options.runtime.show_plots,
                    save_name=os.path.join(
                        config.project_name,
                        f"pred_spectrum_{molecule.susc.temperature:.2f}_K{_suffix}",
                    ),
                )

        _all_avgs = [nuc.shift.avg for nuc in molecule.nuclei if nuc.shift.avg is not None]
        _overall_range = [np.min(_all_avgs), np.max(_all_avgs)] if _all_avgs else [0.0, 1.0]
        linewidth_output = resolve_output_linewidths(molecule, _overall_range)
        linewidth_outputs.append(linewidth_output)

    # TODO If more than one temperature, then make a stacked plot of spectra

    # Save susceptibility data to file
    save_susc(
        molecules,
        os.path.join(config.project_name, "susceptibility_tensor.csv"),
        comment=(
            "Data from spin-only fallback (no susceptibility file)"
            if config.susceptibility_file is None
            else "Data from {} ({})".format(
                Path(config.susceptibility_file).name,
                config.susceptibility_format
                if config.susceptibility_format is not None
                else (f"orca_{section}" if backend == "orca" else backend),
            )
        ),
        susc_units=getattr(config, "susc_units", "A3"),
    )

    # Write shift and peak data to file
    for molecule, linewidth_output in zip(molecules, linewidth_outputs):
        save_molecule_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                f"hyperfines_and_shifts_{molecule.susc.temperature:.2f}_K.csv",
            ),
            delimiter=options.runtime.csv_delimiter,
            comment=f"T = {molecule.susc.temperature:.2f} K",
            verbose=True,
        )

        save_peak_data_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                f"peak_data_{molecule.susc.temperature:.2f}_K.csv",
            ),
            linewidth_by_label=linewidth_output.values_by_label,
            linewidth_column_name=linewidth_output.column_name,
            comment=f"T = {molecule.susc.temperature:.2f} K",
            verbose=True,
        )

    return 0


def _apply_relaxation_linewidths(
    config,
    base_molecule: Molecule,
    experiment,
):
    """
    Apply linewidths using a user-specified relaxation model.

    This function updates `base_molecule` in-place by storing the computed
    relaxation evaluation in the domain object and by setting `nuc.shift.lw`
    when relaxation inputs are provided in the config.

    Args:
        config (PredictConfig): Prediction configuration containing relaxation
            settings and physical parameters.
        base_molecule (Molecule): Molecule instance to update in-place.
        experiment: Optional experiment object used as fallback when relaxation
            temperature and magnetic field are not explicitly overridden in the
            config.

    Returns:
        None
    """

    if not getattr(config, "relaxation_model", None):
        logger.warning(
            "No relaxation model specified; linewidths will be scaled "
            "automatically for plotting and CSV output"
        )
        base_molecule.relaxation = None
        return

    temperature, magnetic_field_tesla = resolve_relaxation_conditions(
        config,
        experiment,
    )

    if magnetic_field_tesla is None or temperature is None:
        base_molecule.relaxation = None
        return

    # Solomon linewidths if relaxation model is SBM
    nuclei_labels = (
        config.nuclei_include
        if isinstance(config.nuclei_include, list)
        else [config.nuclei_include]
    )
    nuclei_labels = [lbl for lbl in nuclei_labels if lbl]

    # Use all nuclei in the molecule that match the requested element(s)
    nuclei_coords = {
        nuc.label: nuc.coord
        for nuc in base_molecule.nuclei
        if remove_numbers(nuc.label) in nuclei_labels
    }
    B0 = magnetic_field_tesla

    # Build Aiso, gamma and omega dictionaries for selected nuclei
    # Converts nuclear gyromagnetic ratios from MHz/T to rad/s/T
    # and multiplies Aiso by 1e6 to convert from MHz to Hz

    if config.hyperfine_method == "pdip":
        # In point-dipole (pdip) model, contact hyperfine A_iso = 0 for all nuclei.
        A_iso_dict = {label: 0.0 for label in nuclei_coords}
    else:
        # Convert domain A_iso (ppm Å^-3) back to MHz for relaxation formulas.
        # Note: conversion depends on the nuclear gyromagnetic ratio for each nucleus.
        A_iso_dict_MHz = {
            nuc.label: float(
                angstrom_to_mhz(
                    1.0 / 3.0 * np.trace(nuc.A.fc),
                    nuclear_gamma=get_nuclear_gamma(nuc.isotope),
                )
            )
            for nuc in base_molecule.nuclei
            if nuc.label in nuclei_coords
        }

        # Convert MHz -> Hz for relaxation routines.
        A_iso_dict = {label: val_mhz * 1e6 for label, val_mhz in A_iso_dict_MHz.items()}

    gamma_I_dict = {
        label: get_nuclear_gamma(label) * 2 * np.pi * 1e6
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

    relaxation_eval = evaluate_relaxation_rates(
        relaxation_model=config.relaxation_model,
        nuclei_coords=nuclei_coords,
        electron_coords=base_molecule.paramagnetic_centre,
        gamma_I_dict=gamma_I_dict,
        omega_I_dict=omega_I_dict,
        omega_S=omega_S,
        spin=spin,
        orbit=orbit,
        total_momentum_J=total_momentum_J,
        A_iso_dict=A_iso_dict,
        temperature=temperature,
        tau_R=tau_R,
        tau_c1=tau_c1,
        tau_c2=tau_c2,
        tau_e1=tau_e1,
        tau_e2=tau_e2,
        compute_r1=True,
        compute_r2=True,
    )

    # Persist the computed relaxation evaluation on the molecule domain object.
    base_molecule.relaxation = relaxation_eval

    rates_r1 = base_molecule.relaxation.r1.total
    rates_r2 = base_molecule.relaxation.r2.total

    if rates_r1 is None or rates_r2 is None:
        raise ValueError("Shared relaxation evaluator returned incomplete R1/R2 rates")

    # Group R2 rates by chemical label for linewidth averaging in Hz.
    r2_by_chem_label = {}
    for nuc in base_molecule.nuclei:
        if nuc.label not in rates_r2:
            continue
        if nuc.chem_label not in r2_by_chem_label:
            r2_by_chem_label[nuc.chem_label] = []
        r2_by_chem_label[nuc.chem_label].append(rates_r2[nuc.label])

    # Calculate average linewidths for each chemical label (Hz)
    avg_lw_by_chem_label = {
        chem_label: np.mean([rate / np.pi for rate in rate_list])
        for chem_label, rate_list in r2_by_chem_label.items()
    }

    min_lw_hz = getattr(config, "relaxation_min_linewidth_hz", 0.0) or 0.0

    for nuc in base_molecule.nuclei:
        if nuc.chem_label in avg_lw_by_chem_label:
            larmor_hz = abs(omega_I_dict[nuc.label]) / (2 * np.pi)
            lw_hz = avg_lw_by_chem_label[nuc.chem_label] + min_lw_hz
            nuc.shift.lw = lw_hz / larmor_hz * 1e6

    return
