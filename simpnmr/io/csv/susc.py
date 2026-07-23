# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write susceptibility data as CSV.

Provides CSV parsing and serialization helpers for susceptibility tensors and
optional fit diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from simpnmr.core.const.physics import NA
from simpnmr.io.csv.csv_util import read_csv_safe, write_csv_safe

logger = logging.getLogger(__name__)


def read_susceptibilities_csv(
    file_name: str,
) -> List[
    Tuple[np.ndarray, float, float | None, float | None, float | None]
]:
    """Read susceptibility tensors from a CSV file.

    The reader loads full susceptibility tensors and temperature values from a
    CSV table, normalizes supported unit variants to ``Å^3``, and optionally
    reads an isotropic susceptibility value when a ``chi_iso ...`` column is
    present. If no isotropic column is available, ``chi_iso`` is returned as
    ``None`` for that row.

    The returned tensor is reconstructed as a symmetric ``3x3`` matrix from the
    ``chi_xx``, ``chi_xy``, ``chi_xz``, ``chi_yy``, ``chi_yz``, and ``chi_zz``
    columns.

    Args:
        file_name: Path to the susceptibility CSV file.

    Returns:
        A list of tuples
        ``(tensor, temperature, chi_iso, chi_iso_spin_only, chi_iso_g_corr)``,
        where ``tensor`` is
        a ``3x3`` susceptibility tensor in ``Å^3``, ``temperature`` is in
        kelvin, and the three ``chi_iso*`` entries are the optional isotropic
        susceptibility channels read from the CSV when present.

    Raises:
        KeyError: If required tensor or temperature columns are missing.
        ValueError: If a value required for tensor or temperature construction
            cannot be converted to ``float``.
    """
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

    out: List[
        Tuple[np.ndarray, float, float | None, float | None, float | None]
    ] = []
    for _, row in data.iterrows():
        tensor = np.array(
            [
                [row["chi_xx (Å^3)"], row["chi_xy (Å^3)"], row["chi_xz (Å^3)"]],
                [row["chi_xy (Å^3)"], row["chi_yy (Å^3)"], row["chi_yz (Å^3)"]],
                [row["chi_xz (Å^3)"], row["chi_yz (Å^3)"], row["chi_zz (Å^3)"]],
            ],
            dtype=float,
        )
        # Match iso columns by name prefix to avoid unit-string encoding
        # issues. "chi_iso (" excludes "chi_iso_g_corr (" via the space.
        def _iso_col(prefix):
            for col in row.index:
                if col.startswith(prefix + " ("):
                    value = row[col]
                    return None if pd.isna(value) else float(value)
            return None

        chi_iso = _iso_col("chi_iso")
        chi_iso_spin_only = _iso_col("chi_iso_spin_only")
        chi_iso_g_corr = _iso_col("chi_iso_g_corr")
        out.append(
            (
                tensor,
                float(row["Temperature (K)"]),
                chi_iso,
                chi_iso_spin_only,
                chi_iso_g_corr,
            )
        )

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

    # Write the three isotropic susceptibility channels explicitly, each when
    # known: the true Tr(chi)/3 (chi_iso), the spin-only reference
    # (chi_iso_spin_only), and the g-corrected contact value (chi_iso_g_corr).
    # Prediction reads the spin-only and g-corrected channels back directly; the
    # true iso is also recoverable from the tensor.
    def _iso_true(m):
        v = m.susc.iso
        return None if v is None else v * conv

    def _iso_spin_only(m):
        v = m.susc.iso_spin_only
        return None if v is None else v * conv

    def _iso_g_corr(m):
        v = m.susc.iso_g_corr
        return None if v is None else v * conv

    # Write susceptibility tensor to CSV
    out = {
        "Temperature (K)": [molecule.susc.temperature for molecule in molecules],
        f"chi_iso ({unit_label})": [_iso_true(m) for m in molecules],
        f"chi_iso-s-dev ({unit_label})": None,
        f"chi_iso_spin_only ({unit_label})": [
            _iso_spin_only(m) for m in molecules
        ],
        f"chi_iso_spin_only-s-dev ({unit_label})": None,
        f"chi_iso_g_corr ({unit_label})": [_iso_g_corr(m) for m in molecules],
        f"chi_iso_g_corr-s-dev ({unit_label})": None,
        f"chi_ax ({unit_label})": [
            molecule.susc.axiality * conv for molecule in molecules
        ],
        f"chi_ax-s-dev ({unit_label})": None,
        f"chi_rh ({unit_label})": [
            molecule.susc.rhombicity * conv for molecule in molecules
        ],
        f"chi_rh-s-dev ({unit_label})": None,
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
            if key == "rh_over_ax":
                continue
            # The fitted isotropic value is written as chi_iso_g_corr, so its
            # uncertainty accompanies that column.
            col_key = "iso_g_corr" if key == "iso" else key
            out[f"chi_{col_key}-s-dev ({unit_label})"] = [
                model.fit_stdev[key] * conv for model in susc_models
            ]

    to_pop = [key for key, value in out.items() if value is None]
    for pop in to_pop:
        out.pop(pop)

    df = pd.DataFrame(data=out)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Susceptibility data written to %s", file_name)

    return
