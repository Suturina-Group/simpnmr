# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load chemical label mappings from CSV.

Reads external data and returns plain mappings for downstream use.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from simpnmr.io.csv.csv_util import read_csv_safe


def load_chem_labels_from_csv(
    file_name: str,
) -> Tuple[
    Dict[str, str],
    Optional[Dict[str, str]],
]:
    """Load chemical labels from a CSV file.

    The CSV must include columns ``atom_label`` and ``chem_label``.
    Optionally, it may include ``chem_math_label``.

    Args:
        file_name: Path to the CSV file.

    Returns:
        al_to_cl: Mapping atom_label -> chem_label.
        al_to_cml: Mapping atom_label -> chem_math_label, or None if not provided.

    Raises:
        KeyError: If duplicate atom labels exist or required entries are missing.
    """

    table = read_csv_safe(file_name)

    # Required columns
    for col in ("atom_label", "chem_label"):
        if col not in table.columns:
            raise KeyError(f"Missing required column '{col}' in chem labels file")

    # Check for duplicate atom labels
    counts = table["atom_label"].value_counts()
    dupes = counts[counts > 1]
    if not dupes.empty:
        raise KeyError(
            f"Duplicate atom_label(s) in chem labels file: {list(dupes.index)}"
        )

    # Check for missing chem_label entries
    if table["chem_label"].isnull().any():
        missing = table.loc[table["chem_label"].isnull(), "atom_label"].iloc[0]
        raise KeyError(f"Missing chem_label for atom_label '{missing}'")

    al_to_cl = dict(zip(table["atom_label"], table["chem_label"]))

    # Optional chem_math_label
    al_to_cml: Optional[Dict[str, str]] = None
    if "chem_math_label" in table.columns:
        if table["chem_math_label"].isnull().any():
            missing = table.loc[table["chem_math_label"].isnull(), "atom_label"].iloc[0]
            raise KeyError(f"Missing chem_math_label for atom_label '{missing}'")
        al_to_cml = {
            al: str(cml).strip()
            for al, cml in zip(table["atom_label"], table["chem_math_label"])
        }

    return al_to_cl, al_to_cml
