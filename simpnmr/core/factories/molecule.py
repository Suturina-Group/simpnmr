# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct Molecule domain objects from external payloads.

Provides helpers to build Molecule instances from QC-derived data or CSV payloads.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simpnmr.core.converters.mhz_to_ang import a_iso_mhz_to_ang, a_tensor_mhz_to_ang
from simpnmr.core.domain.molecule import Molecule
from simpnmr.tools.coords_tools import xyz_format as xyzf


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
        a_iso = a_iso_mhz_to_ang(a_iso)
        # Convert dipolar hyperfine tensors
        a_dip = a_tensor_mhz_to_ang(a_dip)

    else:
        raise ValueError(f"Unknown converter: {converter}")

    return Molecule.from_hyperfine_data(
        labels=labels,
        coords=coords,
        a_iso=a_iso,
        a_dip=a_dip,
        elements=elements,
    )


def build_molecule_from_csv(
    payload: dict,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Build a `Molecule` from an IO CSV payload."""

    labels: list[str] = payload["labels"]
    coords = payload["coords"]

    tensors = payload.get("tensors")  # dict[label, (3,3)] | None
    chem_labels = payload.get("chem_labels")  # list[str] | None
    chem_math_labels = payload.get("chem_math_labels")  # list[str] | None

    # --- Hyperfine: tensors -> (a_iso, a_dip) ---
    a_iso = None
    a_dip = None
    if tensors is not None:
        if isinstance(tensors, dict):
            tensor_by_label = {k: np.asarray(v, float) for k, v in tensors.items()}
        else:
            tensor_by_label = {
                lab: np.asarray(t, float) for lab, t in zip(labels, tensors)
            }

        a_iso = {}
        a_dip = {}
        for lab in labels:
            if lab not in tensor_by_label:
                raise KeyError(f"Missing hyperfine tensor for label: {lab}")

            A = np.asarray(tensor_by_label[lab], float)
            if A.shape != (3, 3):
                raise ValueError(
                    f"Hyperfine tensor for {lab} must be (3,3), got {A.shape}"
                )

            iso = float(np.trace(A) / 3.0)
            a_iso[lab] = iso
            a_dip[lab] = A - np.eye(3) * iso

    # --- Chem labels ---
    al_to_cl = None
    al_to_cml = None
    if chem_labels is not None:
        if len(chem_labels) != len(labels):
            raise ValueError(
                f"chem_labels length mismatch: {len(chem_labels)} vs {len(labels)}"
            )
        al_to_cl = {lab: str(cl) for lab, cl in zip(labels, chem_labels)}

        if chem_math_labels is not None:
            if len(chem_math_labels) != len(labels):
                raise ValueError(
                    f"chem_math_labels length mismatch: "
                    f"{len(chem_math_labels)} vs {len(labels)}"
                )
            al_to_cml = {lab: str(cml) for lab, cml in zip(labels, chem_math_labels)}

    # --- Build molecule ---
    if a_iso is not None and a_dip is not None:
        molecule = Molecule.from_hyperfine_data(
            labels=labels,
            coords=coords,
            a_iso=a_iso,
            a_dip=a_dip,
            elements=elements,
        )
    else:
        molecule = Molecule.from_labels_coords(
            labels=labels,
            coords=coords,
            elements=elements,
        )

    if al_to_cl is not None:
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

    return molecule
