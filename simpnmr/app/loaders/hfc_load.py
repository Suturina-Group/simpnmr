# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Application-layer loader for enriching an existing Molecule with hyperfine data.

This loader encapsulates the selection of hyperfine source (DFT/QC, point-dipole
approximation, or CSV) and applies hyperfine-related information to a base domain
`Molecule` that has already been constructed earlier in the pipeline.

Domain rules are implemented in builders and domain objects; this module
orchestrates IO + builders according to the provided configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from simpnmr.app.policies.hfc import (
    OrbitalContribution,
    normalise_orbital_contribution,
)
from simpnmr.core.build.hfc import (
    build_hfc_from_csv,
    build_hfc_from_pdip,
    build_hfc_from_qca,
)
from simpnmr.core.domain.mol import Molecule
from simpnmr.io.csv.mol import read_molecule_csv
from simpnmr.io.qc import gateway as rdrs
from simpnmr.io.qc.backends.orca.detect import detect_hfc_has_orb

logger = logging.getLogger(__name__)


def load_hyperfines(
    molecule: Molecule,
    config: Any,
) -> Molecule:
    """Enrich an existing Molecule with hyperfine information.

    Args:
        molecule: Base Molecule that has already been constructed earlier in the
            pipeline.
        config: Application configuration object. Expected attributes:
            - hyperfine_method: One of {"dft", "pdip", "csv"}.
            - hyperfine_file: Path to the input file.
            - hyperfine_orbital_contribution: Orbital-contribution policy for
              QC-derived hyperfine data.

    Returns:
        The input Molecule enriched with hyperfine data as required by the
        selected method.

    Raises:
        ValueError: If the hyperfine method is unsupported, or if the requested
            orbital-contribution policy is incompatible with the input data.
    """
    method = config.hyperfine_method

    # DFT/QC-derived hyperfine data.
    if method == "dft":
        requested_orbital_contribution = normalise_orbital_contribution(
            config.hyperfine_orbital_contribution
        )
        has_orb = detect_hfc_has_orb(config.hyperfine_file)

        if requested_orbital_contribution is OrbitalContribution.ON and not has_orb:
            raise ValueError(
                "Requested orbital hyperfine contribution, but no A(ORB) term "
                "was found in the input file."
            )

        if requested_orbital_contribution is OrbitalContribution.AUTO:
            effective_orbital_contribution = (
                OrbitalContribution.ON if has_orb else OrbitalContribution.OFF
            )
        else:
            effective_orbital_contribution = requested_orbital_contribution

        if effective_orbital_contribution is OrbitalContribution.ON:
            logger.info("Hyperfine contributions used: A(FC), A(SD), A(ORB)")
        else:
            logger.info("Hyperfine contributions used: A(FC), A(SD)")

        # Record final, domain-level hyperfine model choice
        hyperfine_orbital_availability = (
            "available"
            if effective_orbital_contribution is OrbitalContribution.ON
            else "unavailable"
        )

        qc_hyperfine_data = rdrs.QCA.guess_from_file(
            config.hyperfine_file,
            spin=getattr(config, "hyperfine_spin", None),
        )

        molecule = build_hfc_from_qca(
            molecule,
            qc_hyperfine_data,
            converter="MHz_to_Ang-3",
            orbital_contribution=effective_orbital_contribution.value,
        )

        molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
            hyperfine_orbital_availability
        )

    # Point dipole approximation.
    elif method == "pdip":
        molecule = build_hfc_from_pdip(molecule)

        molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
            "unavailable"
        )

    # CSV-provided hyperfines.
    elif method == "csv":
        csv_payload = read_molecule_csv(config.hyperfine_file)
        molecule = build_hfc_from_csv(molecule, csv_payload)
        molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
            "unavailable"
        )

    else:
        raise ValueError(
            "Unsupported hyperfine_method. Expected one of {'dft', 'pdip', 'csv'}, "
            f"got: {method!r}"
        )

    return molecule
