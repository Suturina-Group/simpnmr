# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute PCS values on 3D grids.

Provides numerical helpers to evaluate PCS isosurfaces from susceptibility tensors.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def compute_pcs_isosurface(
    chi_dtensor: np.ndarray,
    labels: ArrayLike,
    center_atom: str,
    pdip_fn: Callable[[np.ndarray], np.ndarray],
    lower: float = -15.0,
    upper: float = 15.0,
    step: float = 0.5,
) -> tuple[np.ndarray, tuple[float, float, float], float, tuple[int, int, int]]:
    """
    Compute PCS values on a 3D grid for a given deviatoric susceptibility tensor.

    Args:
        chi_dtensor: Deviatoric susceptibility tensor (3x3), in Å³.
        labels: Atomic labels (including indices).
        coords: Atomic coordinates in Å.
        center_atom: Label of the paramagnetic center atom.
        lower: Lower grid bound (bohr).
        upper: Upper grid bound (bohr).
        step: Grid spacing (bohr).

    Returns:
        Tuple of (values, origin_bohr, step_bohr, grid_shape), where:
            - values: PCS grid values in ppb, shape (nx, ny, nz).
            - origin_bohr: Grid origin (x0, y0, z0) in bohr.
            - step_bohr: Grid step in bohr.
            - grid_shape: Grid dimensions (nx, ny, nz).
    """
    labels = np.asarray(labels)

    center_idx = np.where(labels == center_atom)[0]
    if center_idx.size == 0:
        raise ValueError(f"Center atom {center_atom} not found in labels")

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        indexing="ij",
    )

    isosurf = np.zeros_like(x, dtype=float)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                r = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])

                if np.allclose(r, 0.0):
                    continue

                pdip = pdip_fn(r)

                isosurf[i, j, k] = (1.0 / 3.0) * np.trace(chi_dtensor @ pdip)

    # Convert to ppb
    values = isosurf * 1e7
    origin_bohr = (lower, lower, lower)

    return values, origin_bohr, step, values.shape
