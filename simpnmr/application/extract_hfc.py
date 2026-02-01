# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Extract hyperfine coupling data from QC output.

Reads a quantum-chemistry result file, constructs a Molecule, and writes
hyperfine data to CSV.
"""

import os

from simpnmr.application.loaders.molecule import load_molecule_from_qca
from simpnmr.application.setup.options import ExtractHFCRunOptions
from simpnmr.io.csv.molecule import save_molecule_to_csv


def run_extract_hfc(
    calculation_data: str,
    options: ExtractHFCRunOptions,
) -> int:
    """
    Extract hyperfine data from a quantum-chemistry output file and write a CSV.

    Args:
        calculation_data: Path to QC output file.
        options: Runtime-related options.

    Returns:
        Exit code.
    """

    # Create molecule object and convert units
    molecule = load_molecule_from_qca(
        calculation_data,
        converter="MHz_to_Ang-3",
    )

    file_head = os.path.splitext(os.path.basename(calculation_data))[0]
    out_name = f"hyperfine_{file_head}.csv"

    save_molecule_to_csv(
        molecule=molecule,
        file_name=out_name,
        verbose=True,
        delimiter=options.runtime.csv_delimiter,
    )

    return 0
