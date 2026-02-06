# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse hyperfine coupling data from Gaussian outputs.

Provides helpers to extract isotropic and anisotropic hyperfine tensors from
Gaussian quantum-chemistry calculation files.
"""

import numpy as np
import numpy.linalg as la
import numpy.typing as npt

from simpnmr.io.qc.errors import MissingSectionError


def read_gaussian_log_a_tensors(file_name: str) -> tuple[npt.NDArray, npt.NDArray]:
    """Extract isotropic and dipolar hyperfine (A) tensors from a Gaussian log.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` is an array of shape `(n_atoms,)` with isotropic values in MHz.
            * `a_dip` is an array of shape `(n_atoms, 3, 3)` with dipolar
            tensors in MHz.
    """

    # Read number of atoms
    with open(file_name, "r") as f:
        for line in f:
            if "NAtoms=" in line:
                spl_line = line.split()
                n_atoms = int(spl_line[spl_line.index("NAtoms=") + 1])
                break

    a_iso = np.zeros(n_atoms)

    # Read isotropic part
    with open(file_name, "r") as f:
        for line in f:
            if "Isotropic Fermi Contact Couplings" in line:
                line = next(f)
                for it in range(n_atoms):
                    line = next(f)
                    a_iso[it] = float(line.split()[3])  # MHz

    a_dip = np.zeros([n_atoms, 3, 3])
    # Read traceless tensor as eigenvalues and eigenvectors
    track = 0
    with open(file_name, "r") as f:
        for line in f:
            if "Anisotropic Spin Dipole Couplings" in line:
                track += 1
                # Make sure in spin density part!
            if "Anisotropic Spin Dipole Couplings" in line and track == 2:
                line = next(f)
                line = next(f)
                line = next(f)
                line = next(f)
                for it in range(n_atoms):
                    line = next(f)
                    val_1 = float(line.split()[2])  # MHz
                    vecs_1 = [float(val) for val in line.split()[-3:]]
                    line = next(f)
                    val_2 = float(line.split()[4])  # MHz
                    vecs_2 = [float(val) for val in line.split()[-3:]]
                    line = next(f)
                    val_3 = float(line.split()[2])  # MHz
                    vecs_3 = [float(val) for val in line.split()[-3:]]
                    vals = np.array([val_1, val_2, val_3])
                    vecs = np.array([vecs_1, vecs_2, vecs_3]).T

                    # Transform back to coordinate frame in MHz
                    a_dip[it, :, :] = vecs @ np.diag(vals) @ la.inv(vecs)
                    line = next(f)

    if track != 2:
        raise MissingSectionError(
            message=(
                "Dipolar hyperfine tensor block not found in Gaussian log "
                "(expected the second 'Anisotropic Spin Dipole Couplings' section)"
            ),
            path=file_name,
            backend="gaussian",
            kind="hfc",
            section="Anisotropic Spin Dipole Couplings",
        )

    return a_iso, a_dip
