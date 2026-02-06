# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse nuclear shielding data from Gaussian outputs.

Provides helpers to extract isotropic shielding values from Gaussian
quantum-chemistry calculation files.
"""

import numpy as np


def read_gaussian09_log_cs(file_name):
    """Read chemical shielding data from a Gaussian 09 log file.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "Magnetic shielding tensor (ppm)" in line:
                while "Number of stored nuclei" not in line:
                    line = next(f)
                n_calcd = int(line.split()[4])
                while "Nucleus:" not in line:
                    line = next(f)
                for _ in range(n_calcd):
                    label = "{}{}".format(line.split()[2], line.split()[1])
                    for _ in range(13):
                        line = next(f)
                    # Read eigenvalues and convert to Anisotropic CS
                    evals = np.array([float(val) for val in line.split()[1:]])
                    evals = sorted(evals)
                    cs_aniso[label] = evals[2] - (evals[0] + evals[1]) / 2.0
                    line = next(f)
                    # Isotropic value
                    cs_iso[label] = float(line.split()[-1])
                    line = next(f)

    return cs_iso, cs_aniso


def read_gaussian16_log_cs(file_name):
    """Read chemical shielding data from a Gaussian 16 log file.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "Isotropic =" in line:
                line = line.replace("=-", "= -")
                cs_iso["{}{:d}".format(line.split()[1], int(line.split()[0]))] = float(
                    line.split()[4]
                )

    return cs_iso, cs_aniso
