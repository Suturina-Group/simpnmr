# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse magnetic susceptibility data from ORCA outputs.

Provides helpers to extract susceptibility tensors from ORCA
quantum-chemistry calculation files.
"""

import numpy as np


def read_orca_susceptibility(file_name: str, section: str) -> dict[float, np.ndarray]:
    """Extract temperature-dependent molar magnetic susceptibility tensors.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        Dictionary mapping temperature in K to a 3x3 susceptibility tensor.
    """

    susceptibilities = {}

    with open(file_name, "r") as f:
        for line in f:
            if f"QDPT WITH {section.upper()}" in line:
                while (
                    "TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR"
                    not in line
                ):
                    line = next(f)
                # Move down until we reach the first temperature header line
                while "TEMPERATURE/K" not in line:
                    line = next(f)
                while "TEMPERATURE/K" in line:
                    _temp = float(line.split("TEMPERATURE/K:")[1])
                    line = next(f)
                    line = next(f)
                    # Read tensor
                    row_1 = [float(val) for val in line.split()]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()]
                    susceptibilities[_temp] = np.array([row_1, row_2, row_3])
                    line = next(f)
                    line = next(f)

    return susceptibilities
