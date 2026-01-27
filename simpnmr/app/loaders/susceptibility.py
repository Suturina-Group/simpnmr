# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""High-level loaders for susceptibility data.

This module sits in the application layer and bridges file I/O to domain objects.

Rules:
- The domain layer (``simpnmr.core.domain``) must not import from ``simpnmr.io``.
- This loader may import both ``simpnmr.io`` readers and domain classes.

Supported formats:
- ``csv`` / ``csv_*``: susceptibility tensors stored in CSV.
- ``orca_<section>``: susceptibility tensors read from ORCA output, where
  ``<section>`` is typically ``cas``, ``nev``, or ``auto``.

The loader returns a list of ``Susceptibility`` domain objects.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import scipy.constants as constants

from simpnmr.core.domain.tensors import Susceptibility
from simpnmr.io.csv.susceptibility import read_susceptibilities_csv
from simpnmr.io.qc import qc_readers as rdrs

_ORCA_PREFIX: Final[str] = "orca_"


def load_susceptibilities(
    susceptibility_file: str,
    susceptibility_format: str,
) -> list[Susceptibility]:
    """Load susceptibility tensors from a given file/format.

    Args:
        susceptibility_file: Path to the susceptibility source (CSV or ORCA output).
        susceptibility_format: Format identifier.
            - CSV: any string containing ``"csv"``.
            - ORCA: strings starting with ``"orca_"`` (e.g., ``"orca_cas"``).
            - Molcas: any string containing ``"molcas"`` (unsupported).

    Returns:
        List of :class:`~simpnmr.core.domain.tensors.Susceptibility`.

    Raises:
        ValueError: If the format is unsupported or no data is found.
    """

    fmt = susceptibility_format.lower().strip()

    if "molcas" in fmt:
        raise ValueError("Molcas files are not currently supported")

    if "csv" in fmt:
        # Returns (tensor, temperature)
        rows = read_susceptibilities_csv(susceptibility_file)
        return [Susceptibility(tensor, temperature=t) for tensor, t in rows]

    if fmt.startswith(_ORCA_PREFIX) or fmt == "orca":
        section = "auto"
        if fmt.startswith(_ORCA_PREFIX) and len(fmt) > len(_ORCA_PREFIX):
            section = fmt.split(_ORCA_PREFIX, 1)[1]

        # ORCA reader returns temperature -> tensor (XT), typically in cm^3 mol^-1 K.
        tensors = rdrs.read_orca_susceptibility(susceptibility_file, section)

        if not tensors:
            raise ValueError("No susceptibility data found in ORCA output")

        # ORCA units of X*T are cm^3 mol^-1 K. Convert to Angstrom^3 K.
        conv = 1e-24 * constants.Avogadro / (4.0 * np.pi)
        conv = 1.0 / conv

        suscs: list[Susceptibility] = []
        for temperature, tensor_xt in tensors.items():
            # Convert X*T -> X by dividing by temperature.
            suscs.append(
                Susceptibility(
                    tensor_xt / float(temperature) * conv,
                    temperature=float(temperature),
                )
            )
        return suscs

    raise ValueError(
        f"Unsupported susceptibility_format='{susceptibility_format}'. "
        "Expected 'csv'/'csv_*' or 'orca_<section>' (e.g., 'orca_cas')."
    )
