# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse nuclear shielding data from Gaussian outputs.

Provides helpers to extract isotropic shielding values from Gaussian
quantum-chemistry calculation files.
"""

import re

import numpy as np

# Gaussian prints the shielding tensor as labelled components, three per line:
#   XX=  904.8946   YX=  493.4538   ZX=   56.5242   (column X: sigma_xx, sigma_yx, sigma_zx)
#   XY=  ...         YY=  ...         ZY=  ...        (column Y)
#   XZ=  ...         YZ=  ...         ZZ=  ...        (column Z)
# so component "AB" maps to tensor[A, B] (sigma_AB, generally non-symmetric).
_CS_COMPONENT_RE = re.compile(r"([XYZ][XYZ])=\s*(-?\d+\.?\d*)")
_AXIS = {"X": 0, "Y": 1, "Z": 2}


def read_gaussian_log_cs(file_name):
    """Read chemical shielding data from a Gaussian log file (09 or 16).

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(cs_iso, cs_aniso, cs_tensor)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
            * `cs_tensor` maps atom labels to the full 3x3 shielding tensor
              (ppm) as a ``numpy`` array; generally non-symmetric.
    """

    cs_iso = {}
    cs_aniso = {}
    cs_tensor = {}

    with open(file_name, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Isotropic =" not in line or "Anisotropy =" not in line:
            continue
        parts = line.replace("=-", "= -").split()
        label = "{}{:d}".format(parts[1], int(parts[0]))
        cs_iso[label] = float(parts[4])
        cs_aniso[label] = float(parts[-1])

        # The three lines after the header carry the nine tensor components.
        components = {}
        for row in lines[i + 1 : i + 4]:
            for name, value in _CS_COMPONENT_RE.findall(row):
                components[name] = float(value)
        if len(components) == 9:
            tensor = np.zeros((3, 3), dtype=float)
            for name, value in components.items():
                tensor[_AXIS[name[0]], _AXIS[name[1]]] = value
            cs_tensor[label] = tensor

    return cs_iso, cs_aniso, cs_tensor
