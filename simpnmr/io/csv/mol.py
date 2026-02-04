# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write molecular structure and hyperfine data as CSV.

Provides CSV parsing and serialization helpers for atom labels, coordinates,
optional chemical labels, and hyperfine tensors.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from simpnmr.__version__ import __version__
from simpnmr.io.csv.csv_util import read_csv_safe

logger = logging.getLogger(__name__)


def read_molecule_csv(file_name: str) -> dict:
    """Read a molecule CSV file and return raw structure + (optional) hyperfine + labels

    This function preserves the legacy parsing behaviour previously implemented
    in `Molecule.from_csv` (domain), but keeps IO in the IO layer.

    Returns dict with keys:
      - labels: list[str]
      - coords: np.ndarray shape (n, 3)
      - tensors: list[np.ndarray] | None
      - chem_labels: list[str] | None
      - chem_math_labels: list[str] | None
    """
    data = read_csv_safe(file_name)

    required_cols = ["atom_label ()", "x (Å)", "y (Å)", "z (Å)"]
    split_hyperfine_cols = [
        "Aiso (ppm Å^-3)",
        "Adip_xx (ppm Å^-3)",
        "Adip_xy (ppm Å^-3)",
        "Adip_xz (ppm Å^-3)",
        "Adip_yy (ppm Å^-3)",
        "Adip_yz (ppm Å^-3)",
        "Adip_zz (ppm Å^-3)",
    ]
    full_hyperfine_cols = [
        "A_xx (ppm Å^-3)",
        "A_xy (ppm Å^-3)",
        "A_xz (ppm Å^-3)",
        "A_yy (ppm Å^-3)",
        "A_yz (ppm Å^-3)",
        "A_zz (ppm Å^-3)",
    ]

    # Standardise column names (kept identical to old domain implementation)
    name_convertor = {
        "atom_labels": "atom_label ()",
        "atom_labels ()": "atom_label ()",
        "chem_label": "chem_label ()",
        "chem_labels ()": "chem_label ()",
        "chem_math_label": "chem_math_label ()",
        "chem_math_labels ()": "chem_math_label ()",
        "x": "x (Å)",
        "x (A)": "x (Å)",
        "y": "y (Å)",
        "y (A)": "y (Å)",
        "z": "z (Å)",
        "z (A)": "z (Å)",
    }
    others = {}
    for key, val in name_convertor.items():
        others[key.capitalize()] = val
        others[val.capitalize()] = val
    name_convertor.update(others)
    data.rename(columns=name_convertor, inplace=True)

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing header(s) {missing} in {file_name}")

    # Detect hyperfine encoding (split vs full) exactly like before
    has_split = all(col in data.columns for col in split_hyperfine_cols)
    has_full = all(col in data.columns for col in full_hyperfine_cols)

    if has_split:
        split = True
    elif has_full:
        split = False
    else:
        # Preserve old behaviour: error if hyperfine headers incomplete.
        # If you later want "structure-only CSV", this is where you'd relax it.
        raise ValueError(f"Incomplete hyperfine headers in {file_name}")

    labels = data["atom_label ()"].tolist()

    coords = np.array([data["x (Å)"], data["y (Å)"], data["z (Å)"]]).T

    if split:
        tensors = [
            np.array(
                [
                    [
                        row["Adip_xx (ppm Å^-3)"],
                        row["Adip_xy (ppm Å^-3)"],
                        row["Adip_xz (ppm Å^-3)"],
                    ],
                    [
                        row["Adip_xy (ppm Å^-3)"],
                        row["Adip_yy (ppm Å^-3)"],
                        row["Adip_yz (ppm Å^-3)"],
                    ],
                    [
                        row["Adip_xz (ppm Å^-3)"],
                        row["Adip_yz (ppm Å^-3)"],
                        row["Adip_zz (ppm Å^-3)"],
                    ],
                ],
                dtype=float,
            )
            + np.eye(3) * float(row["Aiso (ppm Å^-3)"])
            for _, row in data.iterrows()
        ]
    else:
        tensors = [
            np.array(
                [
                    [
                        row["A_xx (ppm Å^-3)"],
                        row["A_xy (ppm Å^-3)"],
                        row["A_xz (ppm Å^-3)"],
                    ],
                    [
                        row["A_xy (ppm Å^-3)"],
                        row["A_yy (ppm Å^-3)"],
                        row["A_yz (ppm Å^-3)"],
                    ],
                    [
                        row["A_xz (ppm Å^-3)"],
                        row["A_yz (ppm Å^-3)"],
                        row["A_zz (ppm Å^-3)"],
                    ],
                ],
                dtype=float,
            )
            for _, row in data.iterrows()
        ]

    chem_labels = None
    chem_math_labels = None

    if "chem_label ()" in data.columns:
        chem_labels = data["chem_label ()"].tolist()

    if "chem_math_label ()" in data.columns:
        chem_math_labels = data["chem_math_label ()"].tolist()

    return {
        "labels": labels,
        "coords": coords,
        "tensors": tensors,
        "chem_labels": chem_labels,
        "chem_math_labels": chem_math_labels,
    }


def save_molecule_to_csv(
    molecule,
    file_name: str = "molecule.csv",
    verbose: bool = True,
    comment: str = "",
    delimiter: str = ",",
) -> None:
    """Save molecule structure, hyperfine data, and shifts to a CSV file.

    Args:
        file_name: Output CSV file name.
        verbose: If True, prints the output file path.
        comment: Optional additional comment line (including comment marker).
        delimiter: CSV delimiter.
    """

    df = _build_molecule_df(molecule)

    _comment = (
        f"# This file was generated with SimpNMR v{__version__} at {{}}\n".format(
            datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        )
    )

    _comment += comment + "\n"

    with open(file_name, "w") as _f:
        _f.write(_comment)

        df.to_csv(_f, sep=delimiter, header=True, float_format="%.5f", index=None)

    if verbose:
        logger.info("Molecule data written to %s", file_name)

    return


def _build_molecule_df(molecule):
    """Build a full molecule table for CSV export."""

    columns = [
        "atom_label ()",
        "chem_label ()",
        "x (Å)",
        "y (Å)",
        "z (Å)",
        "Aiso (ppm Å^-3)",
        "Adip_xx (ppm Å^-3)",
        "Adip_xy (ppm Å^-3)",
        "Adip_xz (ppm Å^-3)",
        "Adip_yy (ppm Å^-3)",
        "Adip_yz (ppm Å^-3)",
        "Adip_zz (ppm Å^-3)",
        "δ_total_avg (ppm)",
        "δ_total (ppm)",
        "δ_dia (ppm)",
        "δ_fc (ppm)",
        "δ_pc (ppm)",
        "linewidth (Hz)",
    ]

    nuclei = molecule.nuclei

    data = {
        "atom_label ()": [nuc.label for nuc in nuclei],
        "chem_label ()": [nuc.chem_label for nuc in nuclei],
        "x (Å)": [nuc.coord[0] for nuc in nuclei],
        "y (Å)": [nuc.coord[1] for nuc in nuclei],
        "z (Å)": [nuc.coord[2] for nuc in nuclei],
        "Aiso (ppm Å^-3)": [nuc.A.iso for nuc in nuclei],
        "Adip_xx (ppm Å^-3)": [nuc.A.dip[0, 0] for nuc in nuclei],
        "Adip_xy (ppm Å^-3)": [nuc.A.dip[0, 1] for nuc in nuclei],
        "Adip_xz (ppm Å^-3)": [nuc.A.dip[0, 2] for nuc in nuclei],
        "Adip_yy (ppm Å^-3)": [nuc.A.dip[1, 1] for nuc in nuclei],
        "Adip_yz (ppm Å^-3)": [nuc.A.dip[1, 2] for nuc in nuclei],
        "Adip_zz (ppm Å^-3)": [nuc.A.dip[2, 2] for nuc in nuclei],
        "δ_total_avg (ppm)": [nuc.shift.avg for nuc in nuclei],
        "δ_total (ppm)": [nuc.shift.total for nuc in nuclei],
        "δ_dia (ppm)": [nuc.shift.dia for nuc in nuclei],
        "δ_fc (ppm)": [nuc.shift.fc for nuc in nuclei],
        "δ_pc (ppm)": [nuc.shift.pc for nuc in nuclei],
        "linewidth (Hz)": [1 for _ in nuclei],
    }

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.reset_index(drop=True)
