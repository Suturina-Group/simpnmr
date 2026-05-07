# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct base Molecule domain objects from structural payloads.

Provides helpers to build Molecule instances from QC-derived or CSV-derived
structural data without attaching hyperfine information.
"""

from __future__ import annotations

import numpy as np

from simpnmr.core.domain.mol import Molecule


def build_molecule_from_qca(
    qca: object,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Build a base `Molecule` from structural data available on a QCA payload.

    This builder is intentionally limited to molecule initialisation. It reads
    only labels and coordinates from the parsed QC payload and returns a base
    domain `Molecule` without attaching any hyperfine information.

    Args:
        qca: Parsed QC payload expected to expose `labels` and `coords`.
        elements: Elements/labels to include.

    Returns:
        A base `Molecule` instance.

    Raises:
        ValueError: If required structural fields are missing or inconsistent.
    """
    if not hasattr(qca, "coords"):
        raise ValueError("QCA object is missing required attribute: coords")
    if not hasattr(qca, "labels"):
        raise ValueError("QCA object is missing required attribute: labels")

    coords = np.asarray(qca.coords, dtype=float)
    labels = [str(lab) for lab in qca.labels]

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"QCA coords must have shape (n_atoms, 3), got {coords.shape}")

    if len(labels) != coords.shape[0]:
        raise ValueError(
            "QCA labels/coords length mismatch: "
            f"{len(labels)} labels vs {coords.shape[0]} coordinate rows"
        )

    return Molecule.from_labels_coords(
        labels=labels,
        coords=coords,
        elements=elements,
    )


def build_molecule_from_labels_coords(
    labels: list[str],
    coords: np.ndarray,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Build a `Molecule` from explicit labels and coordinates.

    Args:
        labels: Atom labels.
        coords: Cartesian coordinates in Å.
        elements: Elements/labels to include.

    Returns:
        A `Molecule` instance.
    """
    return Molecule.from_labels_coords(
        labels=labels,
        coords=coords,
        elements=elements,
    )


def build_molecule_from_csv(
    payload: dict,
    *,
    elements: list[str] | str = "all",
    exclude: list[str] | None = None,
) -> Molecule:
    """Build a base `Molecule` from a CSV-derived structural payload.

    The CSV payload may contain additional fields such as hyperfine tensors or
    chemical labels. This builder intentionally uses only the structural fields
    required to initialise a base `Molecule`. Any hyperfine enrichment should be
    handled later by a dedicated HFC builder/loader path.

    Args:
        payload: CSV payload expected to contain at least `labels` and `coords`.
        elements: Elements/labels to include.

    Returns:
        A base `Molecule` instance.
    """
    labels: list[str] = payload["labels"]
    coords = payload["coords"]

    return Molecule.from_labels_coords(
        labels=labels,
        coords=coords,
        elements=elements,
        exclude=exclude,
    )
