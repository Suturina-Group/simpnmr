# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit correlation-time parameters to experimental R1 data.

Loads experiments and hyperfine data, evaluates relaxation models, and fits
correlation times using non-linear least squares.
"""

from __future__ import annotations

import logging
import os
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

# Application layer
from simpnmr.app.loaders.elstate_load import load_electronic_state
from simpnmr.app.loaders.exp_load import load_experiments
from simpnmr.app.loaders.hfc_load import load_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_base_molecule
from simpnmr.app.loaders.paramag_centre_load import load_paramagnetic_centre
from simpnmr.app.params.options import FitCorrTimeRunOptions
from simpnmr.app.policies.hfc import has_missing_selected_chem_labels
from simpnmr.app.policies.relax import average_relaxation_rates_by_chem_label

# Core / domain
from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.conv.ang_to_freq import angstrom_to_mhz, mhz_to_rad_s
from simpnmr.core.relaxation.eval import evaluate_relaxation_rates

# IO layer
from simpnmr.io.csv.corr_time import save_corr_time_fit_data
from simpnmr.io.xyz import xyz_write

# Visualisation
from simpnmr.viz.plots.corr_time import plot_corr_time_contrib, plot_corr_time_scatter
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def _assemble_relaxation_rows(
    *,
    observable: str,
    exp_blocks,
    relaxation_model: str,
    nuclei_coords: dict[str, np.ndarray],
    electron_coords: np.ndarray,
    gamma_I_dict: dict[str, float],
    spin: float,
    orbit: float,
    total_momentum_J: float,
    A_fc_dict: dict[str, float],
    base_molecule,
    tau_R: float,
    tau_E: float,
) -> dict[str, np.ndarray | None]:
    """Evaluate row-aligned relaxation outputs for correlation-time workflows.

    This helper evaluates the canonical per-nucleus relaxation decomposition for
    the supplied correlation times, projects the selected observable onto
    chemical labels, and returns row-aligned arrays matching the diagnostic and
    fitting order used by ``exp_blocks``.

    Args:
        observable: Relaxation observable to evaluate. Supported values are
            ``"r1"`` and ``"r2"``.
        exp_blocks: Experimental blocks used by the fitting workflow.
        relaxation_model: Relaxation model selector passed to the shared
            evaluator.
        nuclei_coords: Mapping from atom label to Cartesian coordinates.
        electron_coords: Paramagnetic-centre coordinates.
        gamma_I_dict: Mapping from atom label to nuclear gyromagnetic ratio.
        spin: Electronic spin quantum number.
        orbit: Electronic orbital angular momentum quantum number.
        total_momentum_J: Total angular momentum quantum number.
        A_fc_dict: Mapping from atom label to isotropic hyperfine coupling.
        base_molecule: Molecule providing chemical-label mappings.
        tau_R: Rotational correlation time.
        tau_E: Electronic correlation time.

    Returns:
        Dictionary with row-aligned arrays for ``total``, ``dipolar``,
        ``contact``, and ``curie``. Optional channels are returned as ``None``
        when absent for the selected model.

    Raises:
        ValueError: If `observable` is not ``"r1"`` or ``"r2"``.
    """
    if observable not in {"r1", "r2"}:
        raise ValueError(
            f"Unsupported relaxation observable: {observable!r}. Expected 'r1' or 'r2'."
        )
    tau_c1 = 1.0 / ((1.0 / tau_R) + (1.0 / tau_E))
    tau_c2 = tau_c1

    total_all: list[float] = []
    dipolar_all: list[float] = []
    contact_all: list[float] = []
    curie_all: list[float] = []

    has_dipolar = False
    has_contact = False
    has_curie = False

    for experiment, chem_labels_block, _exp_r1_block in exp_blocks:
        b0 = experiment.magnetic_field
        temperature = experiment.temperature
        omega_I_dict = {label: -gamma_I_dict[label] * b0 for label in nuclei_coords}
        omega_S = -EGAMMA * b0 * 2 * np.pi * 1e6

        relaxation_eval = evaluate_relaxation_rates(
            relaxation_model=relaxation_model,
            nuclei_coords=nuclei_coords,
            electron_coords=electron_coords,
            gamma_I_dict=gamma_I_dict,
            omega_I_dict=omega_I_dict,
            omega_S=omega_S,
            spin=spin,
            orbit=orbit,
            total_momentum_J=total_momentum_J,
            A_iso_dict=A_fc_dict,
            temperature=temperature,
            tau_R=tau_R,
            tau_c1=tau_c1,
            tau_c2=tau_c2,
            tau_e2=tau_E,
            compute_r1=observable == "r1",
            compute_r2=observable == "r2",
        )
        channels = relaxation_eval.r1 if observable == "r1" else relaxation_eval.r2
        rates_total = channels.total

        if rates_total is None:
            raise ValueError(
                "Shared relaxation evaluator returned no total rates for "
                f"observable={observable!r}"
            )

        avg_total_by_chem_label = average_relaxation_rates_by_chem_label(
            base_molecule, rates_total
        )
        avg_dipolar_by_chem_label = average_relaxation_rates_by_chem_label(
            base_molecule, channels.dipolar
        )
        avg_contact_by_chem_label = average_relaxation_rates_by_chem_label(
            base_molecule, channels.contact
        )
        avg_curie_by_chem_label = average_relaxation_rates_by_chem_label(
            base_molecule, channels.curie
        )

        if avg_dipolar_by_chem_label is not None:
            has_dipolar = True
        if avg_contact_by_chem_label is not None:
            has_contact = True
        if avg_curie_by_chem_label is not None:
            has_curie = True

        for chem_label in chem_labels_block:
            total_all.append(avg_total_by_chem_label.get(chem_label, np.nan))
            dipolar_all.append(
                np.nan
                if avg_dipolar_by_chem_label is None
                else avg_dipolar_by_chem_label.get(chem_label, np.nan)
            )
            contact_all.append(
                np.nan
                if avg_contact_by_chem_label is None
                else avg_contact_by_chem_label.get(chem_label, np.nan)
            )
            curie_all.append(
                np.nan
                if avg_curie_by_chem_label is None
                else avg_curie_by_chem_label.get(chem_label, np.nan)
            )

    return {
        "total": np.array(total_all),
        "dipolar": np.array(dipolar_all) if has_dipolar else None,
        "contact": np.array(contact_all) if has_contact else None,
        "curie": np.array(curie_all) if has_curie else None,
    }


def _fit_corr_time_from_r1(
    *,
    fix_param: str | None,
    xdata: np.ndarray,
    exp_r1: np.ndarray,
    assemble_r1_rows,
    tau_R_guess: float,
    tau_E_guess: float,
    tau_R_bounds,
    tau_E_bounds,
) -> tuple[float | None, float | None, np.ndarray, dict[str, np.ndarray | None]]:
    """Fit correlation-time parameters against experimental ``R1`` values.

    Args:
        fix_param: Fit-mode selector. Supported values are ``"tau_r"``,
            ``"tau_e"``, ``None``, ``"none"``, and ``""``.
        xdata: Dummy x-axis required by ``curve_fit``.
        exp_r1: Experimental ``R1`` values.
        assemble_r1_rows: Callable returning row-aligned fitted ``R1`` outputs
            for supplied ``tau_R`` and ``tau_E``.
        tau_R_guess: Initial guess for ``tau_R``.
        tau_E_guess: Initial guess for ``tau_E``.
        tau_R_bounds: Optional bounds for ``tau_R``.
        tau_E_bounds: Optional bounds for ``tau_E``.

    Returns:
        Tuple ``(tau_R_fit, tau_E_fit, theory_r1, fitted_rows)``.

    Raises:
        ValueError: If fitted correlation times are invalid or the fit mode is
            unsupported.
    """
    tau_R_fit = None
    tau_E_fit = None

    if fix_param == "tau_r":
        tau_R = float(tau_R_guess)
        initial_guess = [float(tau_E_guess)]

        def r1_model(_, tau_E):
            return assemble_r1_rows(tau_R=tau_R, tau_E=tau_E)["total"]

        if tau_E_bounds:
            popt, _pcov = curve_fit(
                r1_model, xdata, exp_r1, p0=initial_guess, bounds=tau_E_bounds
            )
        else:
            popt, _pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

        tau_E_fit = popt[0]
        fitted_rows = assemble_r1_rows(tau_R=tau_R, tau_E=tau_E_fit)
        theory_r1 = fitted_rows["total"]
        if tau_E_fit <= 0:
            raise ValueError(f"Fitted tau_E is negative: {tau_E_fit:.3e} s.")
        return tau_R_fit, tau_E_fit, theory_r1, fitted_rows

    if fix_param == "tau_e":
        tau_E = float(tau_E_guess)
        initial_guess = [float(tau_R_guess)]

        def r1_model(_, tau_R):
            return assemble_r1_rows(tau_R=tau_R, tau_E=tau_E)["total"]

        if tau_R_bounds:
            popt, _pcov = curve_fit(
                r1_model, xdata, exp_r1, p0=initial_guess, bounds=tau_R_bounds
            )
        else:
            popt, _pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

        tau_R_fit = popt[0]
        fitted_rows = assemble_r1_rows(tau_R=tau_R_fit, tau_E=tau_E)
        theory_r1 = fitted_rows["total"]
        if tau_R_fit <= 0:
            raise ValueError(f"Fitted tau_R is negative: {tau_R_fit:.3e} s.")
        logger.info("Fitted tau_R: %.3e s", tau_R_fit)
        return tau_R_fit, tau_E_fit, theory_r1, fitted_rows

    if not fix_param or fix_param in ["none", ""]:
        initial_guess = [float(tau_R_guess), float(tau_E_guess)]
        bounds = None
        if tau_R_bounds and tau_E_bounds:
            bounds = (
                [tau_R_bounds[0], tau_E_bounds[0]],
                [tau_R_bounds[1], tau_E_bounds[1]],
            )

        def r1_model(_, tau_R, tau_E):
            return assemble_r1_rows(tau_R=tau_R, tau_E=tau_E)["total"]

        if bounds:
            popt, _pcov = curve_fit(
                r1_model, xdata, exp_r1, p0=initial_guess, bounds=bounds
            )
        else:
            popt, _pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

        tau_R_fit, tau_E_fit = popt
        fitted_rows = assemble_r1_rows(tau_R=tau_R_fit, tau_E=tau_E_fit)
        theory_r1 = fitted_rows["total"]
        if tau_R_fit <= 0 and tau_E_fit > 0:
            raise ValueError(f"tau_R is negative: {tau_R_fit:.3e} s.")
        if tau_E_fit <= 0 and tau_R_fit > 0:
            raise ValueError(f"tau_E is negative: {tau_E_fit:.3e} s.")
        if tau_R_fit <= 0 and tau_E_fit <= 0:
            raise ValueError(
                f"Both tau_R and tau_E are negative: "
                f"tau_R = {tau_R_fit:.3e} s, tau_E = {tau_E_fit:.3e} s."
            )
        return tau_R_fit, tau_E_fit, theory_r1, fitted_rows

    raise ValueError("Correlation times must be 'tau_r' or 'tau_e'.")


def _resolve_corr_time_fit_mode(
    config,
) -> tuple[str | None, float, float, object, object]:
    """Resolve correlation-time fit modes, guesses, and optional bounds.

    Args:
        config: Fit configuration object containing ``fit_corr_time_tau_R`` and
            ``fit_corr_time_tau_E`` entries.

    Returns:
        Tuple ``(fix_param, tau_R_guess, tau_E_guess, tau_R_bounds,
        tau_E_bounds)``.

    Raises:
        ValueError: If both correlation times are fixed or if the fit-mode
            syntax is invalid.
    """
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
        fix_param = None
    elif tau_R_mode == "fix" and tau_E_mode == "fix":
        raise ValueError(
            "Both tau_R and tau_E cannot be fixed. At least one must be set to 'fit'."
        )
    else:
        raise ValueError(
            "Use syntax 'tau_C: [fit/fix, guess, [upper-bound, lower-bound]]', "
            "with bounds optional (tau_C refers to tau_R or tau_E)."
        )

    return fix_param, tau_R_guess, tau_E_guess, tau_R_bounds, tau_E_bounds


def run_fit_corr_time(config, options: FitCorrTimeRunOptions | None = None) -> int:
    """Fit correlation-time parameters to experimental R1 values.

    This pipeline reads experiments and hyperfine data, evaluates the requested
    relaxation model, and uses `scipy.optimize.curve_fit` to fit correlation times.

    Args:
        config: FitCorrTimeConfig loaded from YAML.

    Returns:
        Exit code: 0 on success.
    """

    if options is None:
        raise ValueError("FitCorrTimeRunOptions is required")

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Build the resolved plotting contract for this run.
    spec = apply_profile(options.runtime.plot_profile)
    show_plots = options.runtime.show_plots
    project_dir = config.project_name

    fix_param, tau_R_guess, tau_E_guess, tau_R_bounds, tau_E_bounds = (
        _resolve_corr_time_fit_mode(config)
    )

    if (
        getattr(config, "fit_corr_time_tau_R", None) is not None
        and getattr(config, "relaxation_model", None) is not None
    ):
        experiments = load_experiments(config.experiment_files)

        exp_blocks = []
        for experiment in experiments:
            chem_labels_block = []
            exp_r1_block = []
            for signal in experiment.signals:
                if signal.r1 is not None and np.isfinite(signal.r1):
                    chem_labels_block.append(signal.assignment)
                    exp_r1_block.append(signal.r1)
            if len(chem_labels_block) > 0:
                exp_blocks.append(
                    (
                        experiment,
                        np.array(chem_labels_block),
                        np.array(exp_r1_block),
                    )
                )

        if not exp_blocks:
            raise ValueError("No valid experimental R1 values found for fitting.")

        chem_labels = np.concatenate([blk[1] for blk in exp_blocks])
        exp_r1 = np.concatenate([blk[2] for blk in exp_blocks])
        xdata = np.arange(len(exp_r1))
        plot_chem_labels = None

        # Load Molecule
        base_molecule = load_base_molecule(config)

        # Load canonical paramagnetic centre into the molecule domain container
        base_molecule = load_paramagnetic_centre(
            molecule=base_molecule,
            paramagnetic_centre=config.hyperfine_paramagnetic_centre,
        )

        # Load Hyperfines
        base_molecule = load_hyperfines(
            molecule=base_molecule,
            config=config,
        )

        # Add chemical labels if provided
        if len(config.chem_labels_file):
            al_to_cl, al_to_cml, _ = load_chem_labels_from_csv(
                config.chem_labels_file
            )
            if has_missing_selected_chem_labels(base_molecule, al_to_cl):
                logger.warning(
                    "Chemical labels file does not define labels for all selected "
                    "nuclei; missing labels will use atom labels."
                )
            base_molecule.apply_chem_labels(al_to_cl, al_to_cml)

        chem_label_to_math_label = {}
        for nuc in base_molecule.nuclei:
            if nuc.chem_label and nuc.chem_math_label:
                chem_label_to_math_label[nuc.chem_label] = nuc.chem_math_label

        plot_chem_labels = [
            chem_label_to_math_label.get(label, label) for label in chem_labels
        ]

        # Prepare relaxation model inputs
        nuclei_coords = {nuc.label: nuc.coord for nuc in base_molecule.nuclei}

        # Dictionaries for relaxation calculations
        # A_iso in angular frequency (rad/s): ORCA prints A/h in MHz, and the
        # SBM/Abragam contact rates use A/hbar = 2*pi*nu.
        A_fc_dict = {
            nuc.label: float(
                mhz_to_rad_s(
                    angstrom_to_mhz(
                        np.trace(nuc.A.fc) / 3.0,
                        get_nuclear_gamma(nuc.isotope),
                    )
                )
            )
            for nuc in base_molecule.nuclei
            if nuc.A is not None
        }
        gamma_I_dict = {
            label: get_nuclear_gamma(label) * 2 * np.pi * 1e6
            for label in nuclei_coords
        }

        # Load electronic state
        base_molecule.electronic = load_electronic_state(
            spin_S=config.spin_S,
            orbit_L=config.orbit,
            total_J=config.total_momentum_J,
            hyperfine_file=config.hyperfine_file,
            hyperfine_method=config.hyperfine_method,
        )
        spin = base_molecule.electronic.spin_S
        orbit = base_molecule.electronic.orbit_L
        total_momentum_J = base_molecule.electronic.total_J

        assemble_r1_rows = partial(
            _assemble_relaxation_rows,
            observable="r1",
            exp_blocks=exp_blocks,
            relaxation_model=config.relaxation_model,
            nuclei_coords=nuclei_coords,
            electron_coords=base_molecule.paramagnetic_centre,
            gamma_I_dict=gamma_I_dict,
            spin=spin,
            orbit=orbit,
            total_momentum_J=total_momentum_J,
            A_fc_dict=A_fc_dict,
            base_molecule=base_molecule,
        )

        tau_R_fit, tau_E_fit, theory_r1, fitted_rows = _fit_corr_time_from_r1(
            fix_param=fix_param,
            xdata=xdata,
            exp_r1=exp_r1,
            assemble_r1_rows=assemble_r1_rows,
            tau_R_guess=float(tau_R_guess),
            tau_E_guess=float(tau_E_guess),
            tau_R_bounds=tau_R_bounds,
            tau_E_bounds=tau_E_bounds,
        )

        rsquared = 1 - (
            np.sum((exp_r1 - theory_r1) ** 2) / np.sum((exp_r1 - np.mean(exp_r1)) ** 2)
        )

        # Write fit diagnostics data
        save_corr_time_fit_data(
            exp_r1=exp_r1,
            theory_r1=theory_r1,
            chem_labels=chem_labels,
            file_name=os.path.join(project_dir, "corr_time_fit_diagnostics.csv"),
            fitted_tau_r=tau_R_fit,
            fitted_tau_e=tau_E_fit,
            rsquared=rsquared,
            theory_r1_dipolar=fitted_rows["dipolar"],
            theory_r1_contact=fitted_rows["contact"],
            theory_r1_curie=fitted_rows["curie"],
            comment="",
            verbose=True,
        )

        # Plot fit diagnostics data
        with spec.context():
            plot_corr_time_scatter(
                theory_r1=theory_r1,
                exp_r1=exp_r1,
                chem_labels=plot_chem_labels,
                rsquared=rsquared,
                fix_param=fix_param,
                tau_R_fit=tau_R_fit,
                tau_E_fit=tau_E_fit,
                spec=spec,
                save=True,
                show=show_plots,
                save_name=os.path.join(project_dir, "experimental_vs_fitted_R1.pdf"),
                verbose=True,
            )
            plot_corr_time_contrib(
                theory_r1=theory_r1,
                exp_r1=exp_r1,
                theory_r1_dipolar=fitted_rows["dipolar"],
                theory_r1_contact=fitted_rows["contact"],
                theory_r1_curie=fitted_rows["curie"],
                chem_labels=plot_chem_labels,
                spec=spec,
                save=True,
                show=show_plots,
                save_name=os.path.join(project_dir, "r1_fit_contributions.pdf"),
                verbose=True,
            )

        if len(config.chem_labels_file):
            xyz_write.save_chemcraft_xyz(
                file_name=os.path.join(project_dir, "chemcraft_structure.xyz"),
                labels=base_molecule.labels,
                coords=base_molecule.coords,
                chem_labels={nuc.label: nuc.chem_label for nuc in base_molecule.nuclei},
            )

        xyz_write.save_xyz(
            file_name=os.path.join(project_dir, "structure.xyz"),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
            comment=f"Structure from {config.hyperfine_file}",
        )

    else:
        raise ValueError(
            "fit_corr_time and relaxation_model must be specified in the input file."
        )

    return 0
