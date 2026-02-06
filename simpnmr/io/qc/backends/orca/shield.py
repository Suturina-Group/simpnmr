# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse nuclear shielding data from ORCA outputs.

Provides helpers to extract isotropic shielding values from ORCA
quantum-chemistry calculation files.
"""

import numpy as np
import numpy.typing as npt


def read_orca5_output_cs(
    file_name: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract chemical shielding values from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}
    # Read Chemical Shielding data
    with open(file_name, "r") as f:
        for line in f:
            if "CHEMICAL SHIELDING SUMMARY (ppm)" in line:
                for _ in range(6):
                    line = next(f)

                while len(line.lstrip().rstrip()):
                    label = "{}{}".format(line.split()[1], int(line.split()[0]))
                    cs_iso[label] = float(line.split()[2])
                    cs_aniso[label] = float(line.split()[3])
                    line = next(f)

    return cs_iso, cs_aniso


def read_orca5_property_cs(
    file_name: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Read chemical shielding data from an ORCA property file.

    Args:
        file_name: Path to the ORCA property file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "EPRNMR_OrbitalShielding" in line:
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
