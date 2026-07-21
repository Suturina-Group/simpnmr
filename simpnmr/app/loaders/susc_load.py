# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load magnetic susceptibility tensors from CSV or QC output.

Reads external data and returns Susceptibility domain objects.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simpnmr.app.policies.susc import resolve_susceptibility_source
from simpnmr.core.build.susc import (
    build_chi_d_tensor_from_csv,
    build_chi_d_tensor_from_orca,
    build_chi_iso_from_csv,
    build_chi_iso_g_corr,
    build_chi_iso_spin_only,
)
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.io.csv.susc import read_susceptibilities_csv
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def load_susceptibilities(
    susceptibility_file: str,
    susceptibility_format: str | None = None,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an explicit source file.

    This function acts as the orchestrator for susceptibility loading. It
    resolves the source backend and delegates to the corresponding CSV or ORCA
    loader helper.

    Args:
        susceptibility_file: Path to the susceptibility source file.
        susceptibility_format: Optional format identifier. If not provided, the
            format is detected from ``susceptibility_file``.
        electronic: Optional electronic-state context used for isotropic
            susceptibility enrichment.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility enrichment.

    Returns:
        Loaded susceptibility domain objects.

    Raises:
        ValueError: If the source format is unsupported.
    """
    backend, section = resolve_susceptibility_source(
        susceptibility_file,
        susceptibility_format,
    )

    if backend == "csv":
        return load_susceptibility_csv(
            susceptibility_file,
            electronic=electronic,
            g_tensor=g_tensor,
        )

    if backend == "orca":
        return load_susceptibility_orca(
            susceptibility_file,
            section=section,
            electronic=electronic,
            g_tensor=g_tensor,
        )

    raise ValueError(f"Unsupported susceptibility backend: {backend!r}")


def load_susceptibility_csv(
    susceptibility_file: str,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from a CSV source file.

    The CSV loader always constructs the tensor-backed susceptibility object
    first. If a CSV isotropic susceptibility value is present, it is attached
    directly as the canonical isotropic susceptibility. Otherwise, when
    electronic-state data is available, the loader first attaches the spin-only
    isotropic susceptibility reference channel. If a g-tensor is also
    available, it then attaches the g-corrected isotropic susceptibility
    channel and promotes it to the canonical ``susc.iso`` while preserving
    ``susc.iso_spin_only`` for downstream delta-shift reporting. If only the
    spin-only channel is available, that channel becomes the canonical
    ``susc.iso``. If insufficient data is available, the isotropic
    susceptibility channel is skipped and only the tensor-backed susceptibility
    object is returned.

    Args:
        susceptibility_file: Path to the CSV susceptibility source file.
        electronic: Optional electronic-state context used for isotropic
            susceptibility enrichment.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility enrichment.

    Returns:
        Loaded susceptibility domain objects.
    """
    rows = read_susceptibilities_csv(susceptibility_file)
    suscs: list[Susceptibility] = []

    def _row_iso(row):
        # g-corrected isotropic susceptibility takes precedence over plain chi_iso
        _, _, chi_iso, chi_iso_g_corr = row
        return chi_iso_g_corr if chi_iso_g_corr is not None else chi_iso

    has_csv_chi_iso = any(_row_iso(r) is not None for r in rows)
    has_rows_without_csv_chi_iso = any(_row_iso(r) is None for r in rows)

    if has_csv_chi_iso:
        if has_rows_without_csv_chi_iso:
            logger.info(
                "CSV source provides chi_iso for some rows; those rows will use "
                "the isotropic susceptibility read directly from CSV"
            )
        else:
            logger.info("Isotropic susceptibility is loaded directly from CSV")

    if has_rows_without_csv_chi_iso:
        if electronic is not None:
            logger.info(
                "CSV rows without chi_iso will build a spin-only isotropic "
                "susceptibility reference channel"
            )
            if g_tensor is not None:
                logger.info(
                    "g-tensor available; CSV rows without chi_iso will also "
                    "build a g-tensor-corrected isotropic susceptibility while "
                    "preserving the spin-only reference"
                )
        else:
            logger.info(
                "CSV rows without chi_iso have insufficient electronic-state "
                "data; isotropic magnetic susceptibility will be skipped for "
                "those rows"
            )

    for tensor, temperature, chi_iso, chi_iso_g_corr in rows:
        susc = build_chi_d_tensor_from_csv(
            temperature=float(temperature),
            tensor=tensor,
        )

        if chi_iso_g_corr is not None:
            # Explicit g-corrected isotropic susceptibility (e.g. from a fit):
            # adopt it as the canonical iso so the Fermi contact is g-corrected.
            susc = build_chi_iso_from_csv(susc, chi_iso=float(chi_iso_g_corr))
            susc.iso_g_corr = float(chi_iso_g_corr)
        elif chi_iso is not None:
            susc = build_chi_iso_from_csv(susc, chi_iso=float(chi_iso))
        elif electronic is not None:
            susc = build_chi_iso_spin_only(
                susc,
                spin=electronic.spin_S,
                orbit=electronic.orbit_L,
                total_momentum_J=electronic.total_J,
            )
            if g_tensor is not None:
                susc = build_chi_iso_g_corr(
                    susc,
                    spin=electronic.spin_S,
                    orbit=electronic.orbit_L,
                    total_momentum_J=electronic.total_J,
                    g_tensor=g_tensor,
                )
                susc.iso = susc.iso_g_corr
            else:
                susc.iso = susc.iso_spin_only
        else:
            pass

        suscs.append(susc)

    return suscs


def load_susceptibility_orca(
    susceptibility_file: str,
    *,
    section: str | None,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an ORCA output file.

    The ORCA loader always constructs the tensor-backed susceptibility object
    first. When electronic-state data is available, the loader first attaches
    the spin-only isotropic susceptibility reference channel. If a g-tensor is
    also available, it then attaches the g-corrected isotropic susceptibility
    channel and promotes it to the canonical ``susc.iso`` while preserving
    ``susc.iso_spin_only`` for downstream delta-shift reporting. If only the
    spin-only channel is available, that channel becomes the canonical
    ``susc.iso``. If insufficient data is available, the isotropic
    susceptibility channel is skipped and only the tensor-backed susceptibility
    object is returned.

    Args:
        susceptibility_file: Path to the ORCA output file.
        section: Resolved ORCA QDPT section label to read.
        electronic: Optional electronic-state context used for isotropic
            susceptibility enrichment.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility enrichment.

    Returns:
        Loaded ORCA susceptibility domain objects.

    Raises:
        ValueError: If the ORCA section is missing or no susceptibility data is
            parsed.
    """

    if section is None:
        raise ValueError("ORCA susceptibility section is required but was None")

    # ORCA reader returns temperature -> tensor (XT), typically in cm^3 mol^-1 K.
    tensors = rdrs.read_orca_susceptibility(susceptibility_file, section)

    if not tensors:
        raise ValueError("No susceptibility data found in ORCA output")

    if electronic is not None:
        logger.info("Building spin-only isotropic susceptibility reference channel")
        if g_tensor is not None:
            logger.info(
                "g-tensor available; building g-tensor-corrected isotropic "
                "susceptibility while preserving spin-only reference"
            )
    else:
        logger.info(
            "Insufficient electronic-state data is available; isotropic "
            "magnetic susceptibility was skipped"
        )

    suscs: list[Susceptibility] = []
    for temperature, tensor_xt in tensors.items():
        susc = build_chi_d_tensor_from_orca(
            temperature=float(temperature),
            tensor_xt=tensor_xt,
        )

        if electronic is not None:
            susc = build_chi_iso_spin_only(
                susc,
                spin=electronic.spin_S,
                orbit=electronic.orbit_L,
                total_momentum_J=electronic.total_J,
            )
            if g_tensor is not None:
                susc = build_chi_iso_g_corr(
                    susc,
                    spin=electronic.spin_S,
                    orbit=electronic.orbit_L,
                    total_momentum_J=electronic.total_J,
                    g_tensor=g_tensor,
                )
                susc.iso = susc.iso_g_corr
            else:
                susc.iso = susc.iso_spin_only
        else:
            pass

        suscs.append(susc)

    return suscs
