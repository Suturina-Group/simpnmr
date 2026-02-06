# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse g-tensor data from ORCA outputs.

Provides helpers to extract g-tensor components from ORCA quantum-chemistry
calculation files.
"""

import numpy as np

from simpnmr.io.qc.errors import ParseError


def read_orca_g_tensor(file_name: str, section: str) -> np.ndarray | None:
    """Extract the electronic g-tensor from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        A 3x3 g-tensor as a NumPy array if found, otherwise None.
    """

    g_tensor = None

    try:
        with open(file_name, "r") as f:
            for line in f:
                # Find the correct QDPT section
                if f"QDPT WITH {section.upper()}" in line:
                    # Go down to the G-matrix header
                    for line in f:
                        if "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN" in line:
                            break
                    # Find "g-matrix:"
                    for line in f:
                        if "g-matrix:" in line:
                            # Next three lines are the rows of the tensor
                            row_1 = [float(val) for val in next(f).split()]
                            row_2 = [float(val) for val in next(f).split()]
                            row_3 = [float(val) for val in next(f).split()]
                            g_tensor = np.array([row_1, row_2, row_3])
                            break
                    break
    except Exception as e:
        raise ParseError(
            message=(
                f"g-tensor could not be parsed from ORCA output "
                f"inside the QDPT {section.upper()} block"
            ),
            path=file_name,
            backend="orca",
            kind="gtensor",
            section=f"QDPT WITH {section.upper()}",
        ) from e

    return g_tensor
