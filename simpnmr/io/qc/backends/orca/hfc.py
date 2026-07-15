# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse hyperfine coupling data from ORCA outputs.

Provides helpers to extract isotropic and deviatoric (traceless) hyperfine tensors from
ORCA quantum-chemistry calculation files.
"""

import logging

import numpy as np
import numpy.typing as npt

from simpnmr.core.util.strings import remove_letters, remove_numbers

logger = logging.getLogger(__name__)


def read_orca5_output_a_tensors(
    file_name: str,
) -> tuple[
    dict[str, npt.NDArray],
    dict[str, npt.NDArray],
    dict[str, npt.NDArray | None],
]:
    """Extract hyperfine (A) tensors from an ORCA 5 output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(a_fc, a_sd, a_orb)` where:
            * `a_fc` maps atom labels to full A(FC) tensors in MHz (shape `(3, 3)`).
            * `a_sd` maps atom labels to full A(SD) tensors in MHz (shape `(3, 3)`).
            * `a_orb` maps atom labels to full A(ORB) tensors in MHz (shape `(3, 3)`)
              when present, otherwise `None`.
    """

    a_fc_tensors: dict[str, npt.NDArray] = {}
    a_sd_tensors: dict[str, npt.NDArray] = {}
    a_orb_tensors: dict[str, npt.NDArray | None] = {}

    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                break
        else:
            raise ValueError(
                "Could not find 'ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE' section "
                f"in: {file_name}"
            )

        for line in f:
            if "Nucleus" not in line or ":" not in line:
                continue

            tmp = line.split()[1]
            label = "{}{}".format(remove_numbers(tmp), remove_letters(tmp))

            a_fc: list[float] | None = None
            a_sd: list[float] | None = None
            a_orb: list[float] | None = None
            r_rows: list[list[float]] = []

            while "Orientation:" not in line:
                line = next(f)
                stripped = line.strip()

                if stripped.startswith("A(FC)"):
                    parts = stripped.split()
                    a_fc = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif stripped.startswith("A(SD)"):
                    parts = stripped.split()
                    a_sd = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif stripped.startswith("A(ORB+DIA)"):
                    parts = stripped.split()
                    a_orb = [float(parts[1]), float(parts[2]), float(parts[3])]

            if a_fc is None or a_sd is None:
                raise ValueError(
                    "Could not find A(FC)/A(SD) principal values "
                    f"for nucleus {label} in: {file_name}"
                )

            for _axis in ("X", "Y", "Z"):
                line = next(f)
                parts = line.split()
                if not parts or parts[0] != _axis:
                    raise ValueError(
                        "Unexpected Orientation format for "
                        "nucleus {label} in: {file_name}"
                    )
                r_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])

            r_mat = np.array(r_rows, dtype=float)

            fc_pas = np.diag(np.array(a_fc, dtype=float))
            sd_pas = np.diag(np.array(a_sd, dtype=float))

            a_fc_tensors[label] = r_mat @ fc_pas @ r_mat.T
            a_sd_tensors[label] = r_mat @ sd_pas @ r_mat.T

            if a_orb is None:
                a_orb_tensors[label] = None
            else:
                orb_pas = np.diag(np.array(a_orb, dtype=float))
                a_orb_tensors[label] = r_mat @ orb_pas @ r_mat.T

    return a_fc_tensors, a_sd_tensors, a_orb_tensors


def read_orca6_output_a_tensors(
    file_name: str,
) -> tuple[
    dict[str, npt.NDArray],
    dict[str, npt.NDArray],
    dict[str, npt.NDArray | None],
]:
    """Extract hyperfine (A) tensors from an ORCA 6 output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(a_fc, a_sd, a_orb)` where:
            * `a_fc` maps atom labels to full A(FC) tensors in MHz (shape `(3, 3)`).
            * `a_sd` maps atom labels to full A(SD) tensors in MHz (shape `(3, 3)`).
            * `a_orb` maps atom labels to full A(ORB) tensors in MHz (shape `(3, 3)`)
              when present, otherwise `None`.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                n_calcd = int(line.split()[5][1:])

    a_fc_tensors: dict[str, npt.NDArray] = {}
    a_sd_tensors: dict[str, npt.NDArray] = {}
    a_orb_tensors: dict[str, npt.NDArray | None] = {}

    # Read hyperfine data
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                for _ in range(n_calcd):
                    while "Nucleus" not in line:
                        line = next(f)
                    tmp = line.split()[1]
                    label = "{}{}".format(remove_numbers(tmp), remove_letters(tmp))
                    # Parse principal hyperfine components and orientation.
                    a_fc: list[float] | None = None
                    a_sd: list[float] | None = None
                    a_orb: list[float] | None = None
                    r_rows: list[list[float]] = []

                    # Collect principal values reported by ORCA (PAS)
                    # Advance until we reach the Orientation block, collecting
                    # principal values on the way.
                    while "Orientation:" not in line:
                        line = next(f)
                        stripped = line.strip()
                        if stripped.startswith("A(FC)"):
                            parts = stripped.split()
                            a_fc = [float(parts[1]), float(parts[2]), float(parts[3])]
                        elif stripped.startswith("A(SD)"):
                            parts = stripped.split()
                            a_sd = [float(parts[1]), float(parts[2]), float(parts[3])]
                        elif stripped.startswith("A(ORB+DIA)"):
                            parts = stripped.split()
                            a_orb = [float(parts[1]), float(parts[2]), float(parts[3])]

                    if a_fc is None or a_sd is None:
                        raise ValueError(
                            "Could not find A(FC)/A(SD) principal values "
                            f"for nucleus {label}"
                        )

                    for _axis in ("X", "Y", "Z"):
                        line = next(f)
                        parts = line.split()
                        if not parts or parts[0] != _axis:
                            raise ValueError(
                                f"Unexpected Orientation format for nucleus {label}"
                            )
                        r_rows.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )

                    r_mat = np.array(r_rows, dtype=float)

                    fc_pas = np.diag(np.array(a_fc, dtype=float))
                    sd_pas = np.diag(np.array(a_sd, dtype=float))

                    a_fc_tensors[label] = r_mat @ fc_pas @ r_mat.T
                    a_sd_tensors[label] = r_mat @ sd_pas @ r_mat.T

                    if a_orb is None:
                        a_orb_tensors[label] = None
                    else:
                        orb_pas = np.diag(np.array(a_orb, dtype=float))
                        a_orb_tensors[label] = r_mat @ orb_pas @ r_mat.T

    return a_fc_tensors, a_sd_tensors, a_orb_tensors
