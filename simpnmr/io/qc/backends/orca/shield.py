# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse nuclear shielding data from ORCA outputs.

Provides helpers to extract isotropic shielding values from ORCA
quantum-chemistry calculation files.
"""

import re

import numpy as np
import numpy.typing as npt

# ORCA per-nucleus header, e.g. " Nucleus  51H :" -> index 51, element H.
_ORCA_NUCLEUS_RE = re.compile(r"Nucleus\s+(\d+)([A-Za-z]{1,2})\s*:")


def read_orca5_output_cs(
    file_name: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, npt.NDArray]]:
    """Extract chemical shielding values from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(cs_iso, cs_aniso, cs_tensor)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
            * `cs_tensor` maps atom labels to the full 3x3 ``Total shielding
              tensor`` (ppm), stored raw (non-symmetric) as ORCA prints it.
    """

    cs_iso = {}
    cs_aniso = {}
    cs_tensor = {}
    with open(file_name, "r") as f:
        lines = f.readlines()

    # Summary table -> isotropic / anisotropic scalars.
    for i, line in enumerate(lines):
        if "CHEMICAL SHIELDING SUMMARY (ppm)" in line:
            for row in lines[i + 6 :]:
                if not row.strip():
                    break
                parts = row.split()
                label = "{}{}".format(parts[1], int(parts[0]))
                cs_iso[label] = float(parts[2])
                cs_aniso[label] = float(parts[3])
            break

    # Per-nucleus blocks -> full (raw, non-symmetric) Total shielding tensor.
    label = None
    for i, line in enumerate(lines):
        match = _ORCA_NUCLEUS_RE.search(line)
        if match:
            label = "{}{}".format(match.group(2), match.group(1))
        elif "Total shielding tensor (ppm)" in line and label is not None:
            rows = [r.split() for r in lines[i + 1 : i + 4]]
            if len(rows) == 3 and all(len(r) == 3 for r in rows):
                cs_tensor[label] = np.array(
                    [[float(v) for v in r] for r in rows], dtype=float
                )

    return cs_iso, cs_aniso, cs_tensor


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
