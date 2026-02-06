# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit correlation-time parameters to experimental R1 data.

Loads experiments and hyperfine data, evaluates relaxation models, and fits
correlation times using non-linear least squares.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from simpnmr.app.loaders.elstate_load import load_electronic_state
from simpnmr.app.loaders.exp_load import load_experiments
from simpnmr.app.loaders.hfc_load import load_base_molecule_from_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.params.options import FitCorrTimeRunOptions
from simpnmr.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.relaxation import gueron, sbm
from simpnmr.core.util.strings import remove_numbers
from simpnmr.io.csv import relax
from simpnmr.io.qc import gateway as rdrs
from simpnmr.io.xyz import xyz_write

logger = logging.getLogger(__name__)

CSV_DELIMITER = ","


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
        experiments = load_experiments(config.experiment_files)

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

        # Load hyperfines / construct base molecule
        base_molecule = load_base_molecule_from_hyperfines(
            config=config,
            delimiter=CSV_DELIMITER,
        )

        # Add chemical labels if provided
        if len(config.chem_labels_file):
            al_to_cl, al_to_cml = load_chem_labels_from_csv(config.chem_labels_file)
            base_molecule.apply_chem_labels(al_to_cl, al_to_cml)

            xyz_write.save_chemcraft_xyz(
                file_name=os.path.join(config.project_name, "chemcraft_structure.xyz"),
                labels=base_molecule.labels,
                coords=base_molecule.coords,
                chem_labels={nuc.label: nuc.chem_label for nuc in base_molecule.nuclei},
            )
        xyz_write.save_xyz(
            file_name=os.path.join(config.project_name, "structure.xyz"),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
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
            label: NUCLEAR_GAMMAS[remove_numbers(label)] * 2 * np.pi * 1e6
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
                    omega_S = -EGAMMA * B0 * 2 * np.pi * 1e6

                    # Calculate relaxation rates for current tau_R, tau_E
                    if config.relaxation_model == "sbm":
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
                        curie_r1_rates = gueron.calc_r1_curie(
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
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        curie_r1_rates = gueron.calc_r1_curie(
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
                    omega_S = -EGAMMA * B0 * 2 * np.pi * 1e6

                    # Calculate relaxation rates for current tau_R, tau_E
                    if config.relaxation_model == "sbm":
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
                        curie_r1_rates = gueron.calc_r1_curie(
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
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        curie_r1_rates = gueron.calc_r1_curie(
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
                    omega_S = -EGAMMA * B0 * 2 * np.pi * 1e6

                    if config.relaxation_model == "sbm":
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
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        rates_r1 = {
                            lab: sbm_dipolar_r1_rates[lab] + sbm_contact_r1_rates[lab]
                            for lab in nuclei_coords
                        }

                    elif config.relaxation_model == "curie":
                        curie_r1_rates = gueron.calc_r1_curie(
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
                            tau_E,
                            spin,
                            total_momentum_J,
                        )
                        curie_r1_rates = gueron.calc_r1_curie(
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
        relax.save_corr_time_fit_data(
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

        # TODO(viz): Move all inline Matplotlib plotting from this pipeline into
        # viz/plots and render via the unified PlotSpec style contract.

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
        if fix_param == "tau_r":
            plt.text(
                0.01,
                0.91,
                f"Fitted $\\tau_{{\\mathrm{{E}}}}$: {tau_E_fit:.3e} s",
                fontsize=12,
                ha="left",
                va="top",
                transform=plt.gca().transAxes,
            )
        elif fix_param == "tau_e":
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

    return 0
