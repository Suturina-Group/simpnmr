# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write chiT regression fit results as CSV.

Provides helpers to serialize fitted slope/intercept parameters and to parse
flattened regression results from CSV files.
"""

import logging

import numpy as np
import pandas as pd

from simpnmr.io.csv.csv_util import read_csv_safe, write_csv_safe

logger = logging.getLogger(__name__)


def save_slope_intercept(
    fits,
    spin: float | None = None,
    file_name: str = "isoaxrho_fit.csv",
    verbose: bool = True,
) -> None:
    """Writes slope/intercept results for chiT fits to a CSV file.

    Args:
        fits: Fit results for each component. Each entry is expected to be a
            mapping containing values such as ``slope``, ``intercept``, and
            associated errors.
        spin: Spin value from the domain model to serialize in the CSV
            comments.
        file_name: Output CSV file path.
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
    comment = [
        "Data reported in Curie-normalised chiT (dimensionless) Units",
    ]
    if spin is not None:
        comment.append(f"spin {spin}")

    df = pd.DataFrame(data=out)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Temperature dependence data is written to %s", file_name)

    return


def save_r6_fit(
    result: dict,
    observable: str,
    temperature: float,
    magnetic_field: float,
    isotope: str,
    file_name: str = "r6_fit.csv",
    verbose: bool = True,
) -> None:
    """Write r^-6 fit parameters (p1, p2) and per-label data to CSV.

    Args:
        result: Dict returned by
            :func:`~simpnmr.core.fitting.r6_fit.fit_r6`.
        observable: ``"r1"`` or ``"width"``.
        temperature: Experiment temperature in Kelvin.
        magnetic_field: Spectrometer magnetic field in Tesla.
        isotope: Isotope label (e.g. ``"1H"``).
        file_name: Output CSV path.
        verbose: Log path when ``True``.
    """
    obs_unit = "s^-1" if observable == "r1" else "ppm"
    p1_unit = f"{obs_unit}.Ang^6"
    comment = [
        f"r^-6 fit: observable={observable}",
        f"isotope={isotope}",
        f"temperature={temperature:.2f} K",
        f"magnetic_field={magnetic_field:.4f} T",
        f"p1={result['p1']:.6g}  p1_err={result['p1_err']:.6g}"
        f"  units={p1_unit}",
        f"p2={result['p2']:.6g}  p2_err={result['p2_err']:.6g}"
        f"  units={obs_unit}",
        f"rmse={result['rmse']:.6g}  units={obs_unit}",
    ]

    r6_inv = result["r_eff"] ** (-6)
    df = pd.DataFrame(
        {
            "label": result["labels"],
            "r_eff_ang": result["r_eff"],
            "mean_r6_inv": r6_inv,
            "observed": result["obs"],
            "predicted": result["pred"],
        }
    )

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("r^-6 fit results written to %s", file_name)


def read_chiT_regression_csv(filename: str) -> dict[str, float]:
    """Read a Curie-normalised chiT regression CSV file.

    Returns fit parameters as a flat dict.

    The CSV is expected to contain columns: ``type``, ``intercept``, and
    ``slope``. Each row is flattened into keys of the form
    ``{type}_intercept`` and ``{type}_slope``.

    Args:
        filename: Path to the regression CSV file.

    Returns:
        Flattened fit parameters keyed by ``{type}_{intercept|slope}``.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """

    # Read CSV, skipping comment lines
    df = read_csv_safe(filename)

    # Sanity check
    required_cols = {
        "type", "intercept", "slope", "intercept_err", "slope_err"
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain {required_cols}")

    params = {}

    for _, row in df.iterrows():
        label = row["type"]
        params[f"{label}_intercept"] = float(row["intercept"])
        params[f"{label}_slope"] = float(row["slope"])
        params[f"{label}_intercept_err"] = float(row["intercept_err"])
        params[f"{label}_slope_err"] = float(row["slope_err"])

    return params
