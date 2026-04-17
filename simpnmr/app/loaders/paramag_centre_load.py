# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Application-layer loader for canonical paramagnetic-centre coordinates.

This module validates and transfers paramagnetic-centre coordinates from parsed
application input into the `Molecule` domain container. The loader requires the
provided centre to resolve unambiguously against the canonical molecule
geometry; otherwise it fails without mutating the domain object. It performs no
user-input parsing and triggers no downstream calculations.
"""

import logging

import numpy as np

from simpnmr.core.domain.mol import Molecule

logger = logging.getLogger(__name__)


def load_paramagnetic_centre(
    molecule: Molecule,
    paramagnetic_centre: list[float] | str | None,
) -> Molecule:
    """Load the canonical paramagnetic centre into a Molecule.

    Accepts either an atom label (e.g. ``"Ni1"``) or explicit XYZ coordinates
    ``[x, y, z]`` in Å.  When a label is given it is resolved against
    ``molecule.labels``; when coordinates are given they are matched against
    ``molecule.coords`` (tolerance 1e-8 Å).  Either way the loader validates
    that exactly one atom matches before mutating the domain object.

    Args:
        molecule: Molecule domain object to enrich.
        paramagnetic_centre: Atom label, XYZ coordinate list, or ``None``.

    Returns:
        The same molecule with ``paramagnetic_centre`` attached when provided.

    Raises:
        ValueError: If the label or coordinates do not resolve to exactly one
            atom in the molecule geometry.
    """
    if paramagnetic_centre is None:
        logger.info("No paramagnetic centre provided; skipping load.")
        return molecule

    if isinstance(paramagnetic_centre, str):
        # Resolve by atom label
        indices = [
            i for i, lbl in enumerate(molecule.labels)
            if lbl == paramagnetic_centre
        ]
        if len(indices) == 0:
            raise ValueError(
                f"Paramagnetic centre label '{paramagnetic_centre}' not found "
                f"in molecule geometry. Available labels: {list(molecule.labels)}"
            )
        if len(indices) > 1:
            raise ValueError(
                f"Paramagnetic centre label '{paramagnetic_centre}' matches "
                "multiple atoms in the molecule geometry"
            )
        molecule.paramagnetic_centre = np.asarray(
            molecule.coords[indices[0]], dtype=float
        )
        return molecule

    # Resolve by XYZ coordinates
    centre = np.asarray(paramagnetic_centre, dtype=float)
    matches = [
        coord
        for coord in molecule.coords
        if np.allclose(np.asarray(coord, dtype=float), centre, atol=1e-8)
    ]

    if len(matches) == 0:
        raise ValueError(
            "Paramagnetic centre coordinates do not match any "
            "coordinate in the molecule geometry"
        )
    if len(matches) > 1:
        raise ValueError(
            "Paramagnetic centre coordinates match multiple "
            "coordinates in the molecule geometry"
        )

    molecule.paramagnetic_centre = centre
    return molecule
