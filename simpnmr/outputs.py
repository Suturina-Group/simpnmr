# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Output helpers for SimpNMR.

This module contains functions for writing susceptibility tensors, relaxation
rate decompositions, and fit diagnostics to CSV/text files.
"""

import datetime
import logging

import numpy as np
import pandas as pd
import scipy.constants as constants

from . import main, models
from .__version__ import __version__

logger = logging.getLogger(__name__)


def save_susc(
    molecules: list[main.Molecule],
    file_name: str = "susceptibility.csv",
    verbose: bool = True,
    susc_models: list[models.SusceptibilityModel] = [],
    susc_units: str = "A3",
    delimiter: str = ",",
    comment: str = "",
):
    """Writes susceptibility tensors for molecules to a CSV file.

    Optionally includes fit-quality metrics and parameter uncertainties from a
    list of fitted susceptibility models.

    Args:
        molecules: Molecules with an associated `molecule.susc` object.
        file_name: Output CSV file path.
        verbose: If ``True``, prints the output file path.
        susc_models: Optional fitted susceptibility models used to add parameter
            standard deviations and fit metrics.
        susc_units: Units for susceptibility values. Supported values are
            ``"A3"``, ``"A3 mol-1"``, ``"cm3 mol-1"``, and ``"cm3 "``.
        delimiter: CSV delimiter.
        comment: Optional comment line appended to the file header. If provided,
            it must begin with ``#`` (or will be prefixed automatically).

    Returns:
        None.
    """

    if susc_units == "A3":
        conv = 1.0
        unit_label = r"Å^3"
    elif susc_units == "A3 mol-1":
        conv = constants.Avogadro
        unit_label = r"Å^3 mol^-1"
    elif susc_units == "cm3 ":
        conv = 1e-24
        unit_label = r"cm^3"
    elif susc_units == "cm3 mol-1":
        conv = 1e-24 * constants.Avogadro / (4 * np.pi)
        unit_label = r"cm^3 mol^-1"

    # Write susceptibility tensor to CSV
    out = {
        "Temperature (K)": [molecule.susc.temperature for molecule in molecules],
        f"chi_iso ({unit_label})": [molecule.susc.iso * conv for molecule in molecules],
        f"chi_iso-s-dev ({unit_label})": None,
        f"chi_ax ({unit_label})": [
            molecule.susc.axiality * conv for molecule in molecules
        ],
        f"chi_ax-s-dev ({unit_label})": None,
        f"chi_rho ({unit_label})": [
            molecule.susc.rhombicity * conv for molecule in molecules
        ],
        f"chi_rho-s-dev ({unit_label})": None,
        f"chi_xx ({unit_label})": [
            molecule.susc.tensor[0, 0] * conv for molecule in molecules
        ],
        f"chi_xx-s-dev ({unit_label})": None,
        f"chi_xy ({unit_label})": [
            molecule.susc.tensor[0, 1] * conv for molecule in molecules
        ],
        f"chi_xy-s-dev ({unit_label})": None,
        f"chi_xz ({unit_label})": [
            molecule.susc.tensor[0, 2] * conv for molecule in molecules
        ],
        f"chi_xz-s-dev ({unit_label})": None,
        f"chi_yy ({unit_label})": [
            molecule.susc.tensor[1, 1] * conv for molecule in molecules
        ],
        f"chi_yy-s-dev ({unit_label})": None,
        f"chi_yz ({unit_label})": [
            molecule.susc.tensor[1, 2] * conv for molecule in molecules
        ],
        f"chi_yz-s-dev ({unit_label})": None,
        f"chi_zz ({unit_label})": [
            molecule.susc.tensor[2, 2] * conv for molecule in molecules
        ],
        f"chi_zz-s-dev ({unit_label})": None,
        f"dchi_xx ({unit_label})": [
            molecule.susc.dtensor[0, 0] * conv for molecule in molecules
        ],
        f"dchi_xx-s-dev ({unit_label})": None,
        f"dchi_xy ({unit_label})": [
            molecule.susc.dtensor[0, 1] * conv for molecule in molecules
        ],
        f"dchi_xy-s-dev ({unit_label})": None,
        f"dchi_xz ({unit_label})": [
            molecule.susc.dtensor[0, 2] * conv for molecule in molecules
        ],
        f"dchi_xz-s-dev ({unit_label})": None,
        f"dchi_yy ({unit_label})": [
            molecule.susc.dtensor[1, 1] * conv for molecule in molecules
        ],
        f"dchi_yy-s-dev ({unit_label})": None,
        f"dchi_yz ({unit_label})": [
            molecule.susc.dtensor[1, 2] * conv for molecule in molecules
        ],
        f"dchi_yz-s-dev ({unit_label})": None,
        f"dchi_zz ({unit_label})": [
            molecule.susc.dtensor[2, 2] * conv for molecule in molecules
        ],
        f"dchi_zz-s-dev ({unit_label})": None,
        f"chi_x ({unit_label})": [
            molecule.susc.eigvals[0] * conv for molecule in molecules
        ],
        f"chi_x-s-dev ({unit_label})": None,
        f"chi_y ({unit_label})": [
            molecule.susc.eigvals[1] * conv for molecule in molecules
        ],
        f"chi_y-s-dev ({unit_label})": None,
        f"chi_z ({unit_label})": [
            molecule.susc.eigvals[2] * conv for molecule in molecules
        ],
        f"chi_z-s-dev ({unit_label})": None,
        "alpha (degrees)": [molecule.susc.alpha for molecule in molecules],
        "alpha-s-dev (degrees)": None,
        "beta (degrees)": [molecule.susc.beta for molecule in molecules],
        "beta-s-dev (degrees)": None,
        "gamma (degrees)": [molecule.susc.gamma for molecule in molecules],
        "gamma-s-dev (degrees)": None,
        "r2 ()": None,
        "r2_adjusted ()": None,
        "MAE (ppm)": None,
        "RMSE (ppm)": None,
    }

    if len(susc_models):
        out["r2 ()"] = [model.r2 for model in susc_models]
        out["r2_adjusted ()"] = [model.adj_r2 for model in susc_models]
        out["MAE (ppm)"] = [model.mae for model in susc_models]
        out["RMSE (ppm)"] = [model.rmse for model in susc_models]
        for key in susc_models[0].fit_stdev:
            if key == "rho_over_ax":
                continue
            out[f"chi_{key}-s-dev ({unit_label})"] = [
                model.fit_stdev[key] * conv for model in susc_models
            ]

    to_pop = [key for key, value in out.items() if value is None]
    for pop in to_pop:
        out.pop(pop)

    df = pd.DataFrame(data=out)

    # TODO: the current pipeline works fine for fitting only,
    # For prediction, we need to check whether chi iso has been treated
    # as spin only value, or calculated with g contribution, or just as Tr(chi)/3

    # Update outpul labels to reflect the physically meaningful definition
    # chi_iso_g_corr = g_e / 3 * Tr(chi @ g.T)
    df = df.rename(
        columns={
            f"chi_iso ({unit_label})": f"chi_iso_g_corr ({unit_label})",
            f"chi_iso-s-dev ({unit_label})": f"chi_iso_g_corr-s-dev ({unit_label})",
        }
    )

    _comment = f"#This file was generated with SimpNMR v{__version__} at {{}}\n".format(
        datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
    )

    if len(comment):
        if comment[0] != "#":
            comment = f"#{comment}"
        _comment += f"{comment}\n"

    with open(file_name, "w") as _f:
        _f.write(_comment)

        df.to_csv(_f, sep=delimiter, header=True, float_format="%.5f", index=None)

    if verbose:
        logger.info("Susceptibility data written to %s", file_name)

    return


def save_relaxation_decomposition(
    avg_r1_by_chem_label: dict[str, float],
    avg_r2_by_chem_label: dict[str, float],
    avg_lw_by_chem_label: dict[str, float],
    file_name: str,
    avg_dipolar_by_chem_label: dict[str, float] | None = None,
    avg_contact_by_chem_label: dict[str, float] | None = None,
    avg_curie_by_chem_label: dict[str, float] | None = None,
    delimiter: str = ",",
    comment: str = "",
    verbose: bool = True,
) -> None:
    """Writes relaxation-rate decompositions and linewidths to a CSV file.

    The function writes per-chemical-label averages for total R1/R2 rates and
    linewidths, and optionally includes decomposed contributions.

    Args:
        avg_r1_by_chem_label: Average total R1 rates by chemical label (s^-1).
        avg_r2_by_chem_label: Average total R2 rates by chemical label (s^-1).
        avg_lw_by_chem_label: Average linewidths by chemical label (Hz).
        file_name: Output CSV file path.
        avg_dipolar_by_chem_label: Optional average SBM dipolar R1 contribution (s^-1).
        avg_contact_by_chem_label: Optional average SBM contact R1 contribution (s^-1).
        avg_curie_by_chem_label: Optional average Curie R1 contribution (s^-1).
        delimiter: CSV delimiter.
        comment: Optional comment line appended to the file header. If provided,
            it must begin with ``#`` (or will be prefixed automatically).
        verbose: If ``True``, prints the output file path.

    Returns:
        None.
    """

    # Collect the union of all chemical labels that appear in any dict
    chem_labels: set[str] = set(avg_r1_by_chem_label.keys())
    chem_labels |= set(avg_r2_by_chem_label.keys())
    chem_labels |= set(avg_lw_by_chem_label.keys())

    if avg_dipolar_by_chem_label is not None:
        chem_labels |= set(avg_dipolar_by_chem_label.keys())
    if avg_contact_by_chem_label is not None:
        chem_labels |= set(avg_contact_by_chem_label.keys())
    if avg_curie_by_chem_label is not None:
        chem_labels |= set(avg_curie_by_chem_label.keys())

    chem_labels = sorted(chem_labels)

    # Base columns
    out: dict[str, list] = {
        "chem_label": chem_labels,
        "R1_total (s^-1)": [
            avg_r1_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
        "R2_total (s^-1)": [
            avg_r2_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
        "linewidth (Hz)": [
            avg_lw_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
    }

    # Optional decompositions
    if avg_dipolar_by_chem_label is not None:
        out["R1_sbm_dipolar (s^-1)"] = [
            avg_dipolar_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_contact_by_chem_label is not None:
        out["R1_sbm_contact (s^-1)"] = [
            avg_contact_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_curie_by_chem_label is not None:
        out["R1_curie (s^-1)"] = [
            avg_curie_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]

    df = pd.DataFrame(data=out)

    _comment = f"#This file was generated with SimpNMR v{__version__} at {{}}\n".format(
        datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
    )

    if len(comment):
        if comment[0] != "#":
            comment = f"#{comment}"
        _comment += f"{comment}\n"

    with open(file_name, "w") as _f:
        _f.write(_comment)
        df.to_csv(_f, sep=delimiter, header=True, float_format="%.5e", index=None)

    if verbose:
        logger.info("Relaxation decomposition written to %s", file_name)

    return


def save_corr_time_fit_data(
    xdata: np.ndarray,
    exp_r1: np.ndarray,
    chem_labels: np.ndarray,
    file_name: str,
    initial_guess: np.ndarray | list[float] | None = None,
    fitted_tau_r: float | None = None,
    fitted_tau_e: float | None = None,
    covariance: np.ndarray | None = None,
    delimiter: str = ",",
    comment: str = "",
    verbose: bool = True,
) -> None:
    """Writes correlation-time fit data (SBM model) to a CSV file.

    The file contains per-point data (`xdata`, `chem_label`, and experimental R1)
    and a header section with optional fit diagnostics (initial guesses, fitted
    parameters, and covariance).

    Args:
        xdata: Independent variable used in the fit (e.g. index or distance).
        exp_r1: Experimental R1 values (s^-1) corresponding to `xdata`.
        chem_labels: Chemical labels corresponding to each data point.
        file_name: Output CSV file path.
        initial_guess: Optional initial guess parameters used in the fit.
        fitted_tau_r: Optional fitted tau_R value (s).
        fitted_tau_e: Optional fitted tau_E value (s).
        covariance: Optional covariance matrix returned by the fit.
        delimiter: CSV delimiter.
        comment: Optional comment line appended to the file header. If provided,
            it must begin with ``#`` (or will be prefixed automatically).
        verbose: If ``True``, prints the output file path.

    Returns:
        None.
    """

    # Tabular data per data point
    out: dict[str, list] = {
        "xdata": list(xdata),
        "chem_label": list(chem_labels),
        "R1_exp (s^-1)": list(exp_r1),
    }

    df = pd.DataFrame(data=out)

    # Header comment with version, timestamp and fit diagnostics
    _comment = f"#This file was generated with SimpNMR v{__version__} at {{}}\n".format(
        datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
    )

    if len(comment):
        if comment[0] != "#":
            comment = f"#{comment}"
        _comment += f"{comment}\n"

    # Append diagnostic information that is currently printed to the terminal
    if initial_guess is not None:
        _comment += f"#initial_guess: {np.array(initial_guess).tolist()}\n"

    if fitted_tau_r is not None:
        _comment += f"#fitted_tau_R (s): {fitted_tau_r:.5e}\n"

    if fitted_tau_e is not None:
        _comment += f"#fitted_tau_E (s): {fitted_tau_e:.5e}\n"

    if covariance is not None:
        cov_array = np.array(covariance)
        _comment += f"#covariance_shape: {cov_array.shape}\n"
        _comment += f"#covariance_flat: {cov_array.flatten().tolist()}\n"

    with open(file_name, "w") as _f:
        _f.write(_comment)
        df.to_csv(_f, sep=delimiter, header=True, float_format="%.5e", index=None)

    if verbose:
        logger.info("Correlation time fit data written to %s", file_name)

    return


def save_slope_intercept(
    fits,
    file_name: str = "isoaxrho_fit.csv",
    delimiter: str = ",",
    verbose: bool = True,
) -> None:
    """Writes slope/intercept results for chiT fits to a CSV file.

    Args:
        fits: Fit results for each component. Each entry is expected to be a mapping
            containing values such as ``slope``, ``intercept``, and associated errors.
        file_name: Output CSV file path.
        delimiter: CSV delimiter.
        verbose: If ``True``, prints the output file path.

    Returns:
        None.

    Notes:
        The output includes fitted parameters and adjusted R² values for each
        component type.
    """

    labels = ["iso", "ax", "rho"]

    types = []
    intercepts = []
    slopes = []
    intercept_errs = []
    slope_errs = []
    adj_r2s = []

    for i in range(len(fits)):
        types.append(labels[i] if i < len(labels) else "c{}".format(i))

        fit = fits[i] if fits[i] is not None else {}

        intercepts.append(fit.get("intercept", np.nan))
        slopes.append(fit.get("slope", np.nan))
        intercept_errs.append(fit.get("intercept_err", np.nan))
        slope_errs.append(fit.get("slope_err", np.nan))
        adj_r2s.append(fit.get("adj_r2", np.nan))

    out = {
        "type": types,
        "intercept": intercepts,
        "slope": slopes,
        "intercept_err": intercept_errs,
        "slope_err": slope_errs,
        "adj_r2": adj_r2s,
    }

    df = pd.DataFrame(data=out)

    _comment = (
        f"#This file was generated with SimpNMR v{__version__} at {{}}\n".format(
            datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        )
        + "#Data reported in Curie-normalised chiT (dimensionless) Units\n"
    )

    with open(file_name, "w") as _f:
        _f.write(_comment)
        df.to_csv(_f, sep=delimiter, header=True, index=None)

    if verbose:
        logger.info("Temperature dependence data is written to %s", file_name)

    return
