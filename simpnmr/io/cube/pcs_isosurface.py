# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write PCS volumetric grids to Gaussian CUBE files.

Provides helpers to export regular 3D PCS grids in CUBE format.
"""

from __future__ import annotations

import numpy as np

from simpnmr.tools.coords_tools import xyz_format as xyzf


def write_pcs_cube(
    file_name: str,
    comment: str,
    labels,
    coords_bohr: np.ndarray,
    origin_bohr: tuple[float, float, float],
    step_bohr: float,
    grid_shape: tuple[int, int, int],
    values: np.ndarray,
) -> None:
    """Write PCS values on a regular grid to a Gaussian CUBE file.

    Args:
        file_name: Output cube file path.
        comment: First comment line written to the cube file.
        labels: Atomic labels (including indices).
        coords_bohr: Atomic coordinates in Bohr, already shifted so the paramagnetic
            center is at the origin.
        origin_bohr: Grid origin (x0, y0, z0) in Bohr.
        step_bohr: Grid step in Bohr.
        grid_shape: Grid dimensions (nx, ny, nz).
        values: PCS values array with shape (nx, ny, nz).
    """
    labels = np.asarray(labels)
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    values = np.asarray(values, dtype=float)

    nx, ny, nz = grid_shape
    x0, y0, z0 = origin_bohr

    if values.shape != (nx, ny, nz):
        raise ValueError(f"`values` has shape {values.shape}, expected {(nx, ny, nz)}")

    with open(file_name, "w") as f:
        f.write(f"{comment}\n")
        f.write("Comment line\n")
        f.write(f"{len(labels):d}   {x0:.6f} {y0:.6f} {z0:.6f}\n")
        f.write(f"{nx:d}   {step_bohr:.6f}    0.000000    0.000000\n")
        f.write(f"{ny:d}   0.000000    {step_bohr:.6f}    0.000000\n")
        f.write(f"{nz:d}   0.000000    0.000000    {step_bohr:.6f}\n")

        # Atomic labels and coordinates (Bohr)
        for lbl, c in zip(labels, coords_bohr):
            f.write(
                "{:d}   0.000000  {:.6f} {:.6f} {:.6f}\n".format(
                    xyzf.lab_to_num(lbl), *c
                )
            )

        # Grid values: 6 per line, with blank line after each z-slab
        a = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    a += 1
                    f.write("{:.5e} ".format(values[i, j, k]))
                    if a == 6:
                        f.write("\n")
                        a = 0
                f.write("\n")
                a = 0
