"""Application loaders for building `Molecule` aggregates.

This module adapts external data sources (QC outputs, files) into *pure* inputs
for domain constructors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simpnmr.core.convertors.hyperfine import a_iso_mhz_to_angst, a_tensor_mhz_to_angst
from simpnmr.core.domain.molecule import Molecule
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.tools.coords_tools import xyz_format as xyzf


def load_molecule_from_hfc_file(
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
    return build_molecule_from_qca(qca, elements=elements, converter=converter)


def build_molecule_from_qca(
    qca: Any,
    *,
    elements: list[str] | str = "all",
    converter: str | None = "MHz_to_Ang-3",
) -> Molecule:
    """Build a `Molecule` from a parsed QC hyperfine object.

    Expects `qca` (typically `rdrs.QCA`) to provide:
      - coords: array-like (n_atoms, 3) in Å
      - a_iso: mapping label_without_index -> float
      - a_dip: mapping label_without_index -> (3, 3) array-like

    Args:
        qca: Parsed QC hyperfine object.
        elements: Elements/labels to include.
        converter: Optional converter string matching the behaviour of
            `Molecule.from_QCA(..., converter=...)`. Defaults to "MHz_to_Ang-3".

    Returns:
        A `Molecule` instance.

    Raises:
        ValueError: If required fields are missing or unknown converter specified.
    """
    if not hasattr(qca, "coords"):
        raise ValueError("QCA object is missing required attribute: coords")
    if not hasattr(qca, "a_iso"):
        raise ValueError("QCA object is missing required attribute: a_iso")
    if not hasattr(qca, "a_dip"):
        raise ValueError("QCA object is missing required attribute: a_dip")

    coords = np.asarray(qca.coords)

    # QC readers usually key hyperfine tensors by labels without indices (e.g. H, C).
    # SimpNMR uses indexed labels (e.g. H1, C2), so assign indices deterministically
    # in the QC order and re-key tensors accordingly.
    labels_nn = list(qca.a_iso.keys())
    labels = xyzf.add_label_indices(labels_nn)

    a_iso: dict[str, float] = {}
    a_dip: dict[str, np.ndarray] = {}
    for old_lab, new_lab in zip(labels_nn, labels):
        a_iso[new_lab] = float(qca.a_iso[old_lab])
        a_dip[new_lab] = np.asarray(qca.a_dip[old_lab], dtype=float)

    # Conversion logic copied from Molecule.from_QCA to preserve previous behaviour
    if converter is None:
        pass
    elif converter == "MHz_to_Ang-3":
        # Convert isotropic hyperfine values
        a_iso = a_iso_mhz_to_angst(a_iso)
        # Convert dipolar hyperfine tensors
        a_dip = a_tensor_mhz_to_angst(a_dip)

    else:
        raise ValueError(f"Unknown converter: {converter}")

    return Molecule.from_hyperfine_data(
        labels=labels,
        coords=coords,
        a_iso=a_iso,
        a_dip=a_dip,
        elements=elements,
    )
