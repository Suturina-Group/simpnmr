# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write chiT regression fit results as CSV.

Provides helpers to serialize fitted slope/intercept parameters and to parse
flattened regression results from CSV files.
"""

import datetime
import logging

import numpy as np
import pandas as pd

from simpnmr.__version__ import __version__

logger = logging.getLogger(__name__)


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


def read_chiT_regression_csv(filename: str) -> dict[str, float]:
    """
    Read a Curie-normalised chiT regression CSV file and return fit parameters.

    The CSV is expected to contain columns: `type`, `intercept`, and `slope`. Each row
    is flattened into keys of the form `{type}_intercept` and `{type}_slope`.

    Args:
        filename (str): Path to the regression CSV file.

    Returns:
        dict[str, float]: Flattened fit parameters keyed by `{type}_{intercept|slope}`.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """

    # Read CSV, skipping comment lines
    df = pd.read_csv(filename, comment="#")

    # Sanity check
    required_cols = {"type", "intercept", "slope", "intercept_err", "slope_err"}
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
