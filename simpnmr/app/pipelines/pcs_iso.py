# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute PCS isosurfaces using isotropic susceptibility tensors.

Loads structural and susceptibility data, computes PCS grids, and writes cube
files for selected temperatures.
"""

import logging
import os

import numpy as np

from simpnmr.app.loaders.susc_load import load_susceptibilities
from simpnmr.app.params.options import CalcPcsIsoRunOptions
from simpnmr.core.domain.tensor import Hyperfine
from simpnmr.core.pcs.isosurf import compute_pcs_isosurface
from simpnmr.io.cube.pcs_iso_write import write_pcs_cube
from simpnmr.io.qc import gateway as rdrs
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)


def run_calc_pcs_iso(
    *,
    susc_file: str,
    susc_format: str,
    temperatures: list[float],
    structure_file: str,
    central_atom: str,
    options: CalcPcsIsoRunOptions,
) -> int:
    """Run PCS isosurface calculation using an isotropic susceptibility tensor.

    This pipeline is intentionally config-free (no YAML). It is designed to be
    callable both from the CLI and from Python code.

    Args:
        susc_file: Path to the susceptibility tensor file.
        susc_format: Susceptibility file format identifier (e.g. "orca_*", "csv_*").
        temperatures: Temperatures (K) to compute PCS isosurfaces for.
        structure_file: Path to the structure file (.xyz, .log, .out).
        central_atom: Indexed label of the paramagnetic centre (e.g. "Ni1").
        options: Runtime-related options.

    Returns:
        Exit code.
    """

    # Load structure
    ext = os.path.splitext(structure_file)[1]
    if ext == ".xyz":
        labels, coords = xyzf.load_xyz(structure_file)
    elif ext in {".log", ".out"}:
        qcs = rdrs.QCStructure.guess_from_file(structure_file)
        labels = qcs.labels
        coords = qcs.coords
    else:
        raise ValueError(f"Unsupported structure file format: {ext}")

    if central_atom not in labels:
        raise ValueError(
            "Specified central atom not present in structure file. "
            "Try indexed labels, e.g. Ni1."
        )

    # Load susceptibility tensors
    suscs = load_susceptibilities(susc_file, susc_format)

    matched = [
        s
        for s in suscs
        if round(s.temperature, 2) in {round(t, 2) for t in temperatures}
    ]

    if not matched:
        raise ValueError(
            f"No susceptibility tensors matched temperatures {temperatures}. "
            f"Available: {[s.temperature for s in suscs]}"
        )

    labels_arr = np.asarray(labels)
    coords_arr = np.asarray(coords, dtype=float)

    center_idx = np.where(labels_arr == central_atom)[0]
    if center_idx.size == 0:
        raise ValueError(f"Center atom {central_atom} not found in labels")

    coords_bohr = coords_arr * 1.88973
    coords_bohr = coords_bohr - coords_bohr[center_idx[0]]

    for s in matched:
        s.calc_irred()

        values, origin_bohr, step_bohr, grid_shape = compute_pcs_isosurface(
            chi_dtensor=s.dtensor,
            labels=labels_arr,
            center_atom=central_atom,
            pdip_fn=Hyperfine.calc_pdip,
        )

        file_name = f"pcs_isosurface_{s.temperature:.2f}_K.cube"
        comment = f"PCS Isosurface from {susc_file} at {s.temperature:.2f} K"

        write_pcs_cube(
            file_name=file_name,
            comment=comment,
            labels=labels_arr,
            coords_bohr=coords_bohr,
            origin_bohr=origin_bohr,
            step_bohr=step_bohr,
            grid_shape=grid_shape,
            values=values,
        )

        logger.info("PCS Isosurface written to %s", file_name)

    return 0
