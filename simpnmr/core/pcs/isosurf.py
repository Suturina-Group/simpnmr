# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute PCS values on 3D grids.

Provides numerical helpers to evaluate PCS isosurfaces from
susceptibility tensors.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def compute_pcs_isosurface(
    chi_dtensor: np.ndarray,
    pdip_fn: Callable[[np.ndarray], np.ndarray],
    lower: float = -15.0,
    upper: float = 15.0,
    step: float = 0.5,
) -> tuple[
    np.ndarray, tuple[float, float, float], float, tuple[int, int, int]
]:
    """Compute PCS values on a 3D grid centred at the paramagnetic centre.

    The grid is defined in bohr relative to the paramagnetic centre (origin).
    The caller is responsible for translating the returned origin to the
    molecule's absolute coordinate frame.

    Args:
        chi_dtensor: Deviatoric susceptibility tensor (3x3), in Å³.
        pdip_fn: Point-dipole function; accepts a displacement vector in Å
            and returns a 3×3 matrix in 1/Å³.
        lower: Lower grid bound relative to centre (bohr).
        upper: Upper grid bound relative to centre (bohr).
        step: Grid spacing (bohr).

    Returns:
        Tuple of (values, origin_rel_bohr, step_bohr, grid_shape), where:
            - values: PCS grid values in ppb, shape (nx, ny, nz).
            - origin_rel_bohr: Grid origin relative to the paramagnetic
              centre (x0, y0, z0) in bohr — add the centre's absolute
              bohr position to get the cube-file origin.
            - step_bohr: Grid step in bohr.
            - grid_shape: Grid dimensions (nx, ny, nz).
    """
    _BOHR_TO_ANG = 0.529177   # 1 bohr = 0.529177 Å

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
                r_bohr = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])

                if np.allclose(r_bohr, 0.0):
                    continue

                # Grid is relative to the paramagnetic centre;
                # pdip_fn expects the displacement vector in Å.
                r_ang = r_bohr * _BOHR_TO_ANG
                pdip = pdip_fn(r_ang)   # 1/Å³

                # (1/3) Tr(chi_Å³ × pdip_1/Å³) → dimensionless PCS
                isosurf[i, j, k] = (1.0 / 3.0) * np.trace(chi_dtensor @ pdip)

    # Convert dimensionless PCS → ppb (×1e9)
    values = isosurf * 1e9
    origin_rel_bohr = (lower, lower, lower)

    return values, origin_rel_bohr, step, values.shape
