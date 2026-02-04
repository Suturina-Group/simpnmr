# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write susceptibility data as CSV.

Provides CSV parsing and serialization helpers for susceptibility tensors and
optional fit diagnostics.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from simpnmr.__version__ import __version__
from simpnmr.core.const.physics import NA
from simpnmr.io.csv.csv_util import read_csv_safe

logger = logging.getLogger(__name__)


def read_susceptibilities_csv(file_name: str) -> List[Tuple[np.ndarray, float]]:
    data = read_csv_safe(file_name)

    # Forward conversion, A^3 --> Key (same mapping as old domain code)
    convs = {
        "(A^3)": 1.0,
        "(Å^3 mol^-1)": NA,
        "(A^3 mol^-1)": NA,
        "(cm^3)": 1e-24,
        "(cm^3 mol^-1)": 1e-24 * NA / (4 * np.pi),
    }

    renamer = {}
    for name in data.keys():
        for unit, factor in convs.items():
            if unit in name:
                data[name] /= factor
                renamer[name] = name.replace(unit, "(Å^3)")

    if renamer:
        data.rename(renamer, inplace=True, axis=1)

    out: List[Tuple[np.ndarray, float]] = []
    for _, row in data.iterrows():
        tensor = np.array(
            [
                [row["chi_xx (Å^3)"], row["chi_xy (Å^3)"], row["chi_xz (Å^3)"]],
                [row["chi_xy (Å^3)"], row["chi_yy (Å^3)"], row["chi_yz (Å^3)"]],
                [row["chi_xz (Å^3)"], row["chi_yz (Å^3)"], row["chi_zz (Å^3)"]],
            ],
            dtype=float,
        )
        out.append((tensor, float(row["Temperature (K)"])))

    return out


def save_susc(
    molecules: Sequence[Any],
    file_name: str = "susceptibility.csv",
    verbose: bool = True,
    susc_models: Sequence[Any] | None = None,
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
        conv = NA
        unit_label = r"Å^3 mol^-1"
    elif susc_units == "cm3 ":
        conv = 1e-24
        unit_label = r"cm^3"
    elif susc_units == "cm3 mol-1":
        conv = 1e-24 * NA / (4 * np.pi)
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

    if susc_models:
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
            f"chi_iso ({unit_label})": f"chi_iso ({unit_label})",
            f"chi_iso-s-dev ({unit_label})": f"chi_iso-s-dev ({unit_label})",
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
