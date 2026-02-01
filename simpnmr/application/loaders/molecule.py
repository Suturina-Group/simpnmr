# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load Molecule objects from CSV or QC data.

Reads external inputs and returns populated Molecule instances.
"""

from __future__ import annotations

from simpnmr.core.domain.molecule import Molecule
from simpnmr.core.factories import molecule
from simpnmr.io.csv.molecule import read_molecule_csv
from simpnmr.io.qc import qc_readers as rdrs


def load_molecule_from_qca(
    file_name: str,
    *,
    elements: list[str] | str = "all",
    converter: str | None = "MHz_to_Ang-3",
) -> Molecule:
    """Load a `Molecule` from a QC hyperfine file.

    Args:
        file_name: Path to a QC file containing hyperfine data.
        elements: Elements/labels to include. Use "all" to include all atoms,
            "all_H" to include all H, or explicit labels like "H7".
        converter: Optional converter string matching the behaviour of
            `Molecule.from_QCA(..., converter=...)`. Defaults to "MHz_to_Ang-3".

    Returns:
        A populated `Molecule` instance.
    """
    qca = rdrs.QCA.guess_from_file(file_name)
    return molecule.build_molecule_from_qca(qca, elements=elements, converter=converter)


def load_molecule_from_csv(
    file_name: str,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Load a Molecule from a CSV file (IO -> domain).

    Reads CSV via IO layer and builds a Molecule via pure domain constructors.
    """
    payload = read_molecule_csv(file_name)
    return molecule.build_molecule_from_csv(payload, elements=elements)
