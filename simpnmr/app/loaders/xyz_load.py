# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load Molecule objects from XYZ files.

Reads atomic labels and coordinates and constructs Molecule instances.
"""

from simpnmr.core.domain.mol import Molecule
from simpnmr.io.xyz import xyz_write


def load_molecule_from_xyz(
    file_name: str,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    labels, coords = xyz_write.load_xyz(file_name)
    return Molecule.from_labels_coords(labels, coords, elements=elements)
