import os

from simpnmr.application.setup.options import ExtractHFCRunOptions
from simpnmr.core.domain.molecule import Molecule
from simpnmr.io.csv.molecule import save_molecule_to_csv
from simpnmr.io.qc import qc_readers as rdrs


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
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(calculation_data)

    # Create molecule object and convert units
    molecule = Molecule.from_QCA(
        calc_data,
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
