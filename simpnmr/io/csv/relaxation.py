# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write relaxation analysis outputs as CSV.

Provides helpers to export relaxation decompositions and correlation-time fit data.
"""

import datetime
import logging

import numpy as np
import pandas as pd

from simpnmr.__version__ import __version__

logger = logging.getLogger(__name__)


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
