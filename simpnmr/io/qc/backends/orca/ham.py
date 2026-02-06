# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse spin Hamiltonian parameters from ORCA outputs.

Provides helpers to extract spin Hamiltonian terms from ORCA
quantum-chemistry calculation files.
"""

import numpy as np


def read_eff_hamiltonian_tensor(file_name: str, section: str) -> np.ndarray | None:
    """Extract the raw effective Hamiltonian tensor from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        A 3x3 effective Hamiltonian tensor in cm-1 as a NumPy array if found,
        otherwise None.
    """

    eff_H_raw = None

    with open(file_name, "r") as f:
        for line in f:
            # Find the correct QDPT section
            if f"QDPT WITH {section.upper()}" in line:
                # Go down to the G-matrix header
                for line in f:
                    if (
                        "Effective Hamiltonian from projected relativistic states "
                        "and relativistic energies:" in line
                    ):
                        break
                # Find "Raw matrix"
                for line in f:
                    if "Raw matrix (cm-1):" in line:
                        # Next three lines are the rows of the tensor
                        row_1 = [float(val) for val in next(f).split()]
                        row_2 = [float(val) for val in next(f).split()]
                        row_3 = [float(val) for val in next(f).split()]
                        eff_H_raw = np.array([row_1, row_2, row_3])
                        break
                break

    return eff_H_raw
