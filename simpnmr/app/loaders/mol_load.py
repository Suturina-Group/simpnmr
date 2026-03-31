# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Application-layer loader for constructing a base Molecule from structural inputs.

This loader encapsulates the selection of structural source (DFT/QC, point-dipole
geometry source, or CSV) and returns a base domain `Molecule` without attaching
hyperfine information.

Domain rules are implemented in builders and domain objects; this module orchestrates
IO + builders according to the provided configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from simpnmr.core.build import mol
from simpnmr.core.domain.mol import Molecule
from simpnmr.io.csv.mol import read_molecule_csv
from simpnmr.io.qc import gateway as rdrs
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)


def load_base_molecule(config: Any) -> Molecule:
    """Load or construct the base Molecule without hyperfine information.

    Args:
        config: Application configuration object. Expected attributes:
            - hyperfine_method: One of {"dft", "pdip", "csv"}.
            - hyperfine_file: Path to the input file.
            - nuclei_include: Elements to retain.

    Returns:
        A base Molecule populated only with structural information
        (labels, coordinates, elements), without hyperfine data.

    Raises:
        ValueError: If the hyperfine method or file format is unsupported.
    """
    method = config.hyperfine_method

    # DFT/QC-derived structure.
    if method == "dft":
        qcs = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
        base_molecule = Molecule.from_labels_coords(
            labels=qcs.labels,
            coords=qcs.coords,
            elements=config.nuclei_include,
        )

    # Point-dipole workflow uses the same structural sources as before,
    # but does not construct PDIP hyperfine data here.
    elif method == "pdip":
        ext = os.path.splitext(config.hyperfine_file)[1]

        if ext == ".xyz":
            labels, coords = xyzf.load_xyz(config.hyperfine_file)
            stripped = xyzf.remove_label_indices(labels)
            if labels == stripped:
                labels, coords = xyzf.load_xyz(
                    config.hyperfine_file, add_indices=True
                )
        elif ext in [".log", ".out"]:
            qcs = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
            labels = qcs.labels
            coords = qcs.coords
        else:
            raise ValueError(
                "Specified hyperfine file format "
                f"{os.path.splitext(config.hyperfine_file)[1]} unsupported"
            )

        base_molecule = Molecule.from_labels_coords(
            labels=labels,
            coords=coords,
            elements=config.nuclei_include,
        )

    # CSV-provided molecule.
    elif method == "csv":
        base_molecule = load_molecule_from_csv(
            config.hyperfine_file,
            elements=config.nuclei_include,
        )

    else:
        raise ValueError(
            "Unsupported hyperfine_method. Expected one of {'dft', 'pdip', 'csv'}, "
            f"got: {method!r}"
        )

    return base_molecule


def load_molecule_from_qca(
    file_name: str,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Load a base `Molecule` from structural data available in a QC payload.

    Args:
        file_name: Path to a QC file containing structural data.
        elements: Elements/labels to include. Use "all" to include all atoms,
            "all_H" to include all H, or explicit labels like "H7".

    Returns:
        A base `Molecule` instance.
    """
    qca = rdrs.QCA.guess_from_file(file_name)
    return mol.build_molecule_from_qca(qca, elements=elements)


def load_molecule_from_csv(
    file_name: str,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Load a Molecule from a CSV file (IO -> domain).

    Reads CSV via IO layer and builds a Molecule via pure domain constructors.
    """
    payload = read_molecule_csv(file_name)
    return mol.build_molecule_from_csv(payload, elements=elements)
