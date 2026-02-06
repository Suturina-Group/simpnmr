"""ORCA backend detection.

This module contains small helpers to detect whether a given text output belongs
to the ORCA quantum-chemistry package.

Detection is intentionally simple and mirrors the legacy logic previously
implemented in `simpnmr.io.qc.readers`.
"""

from __future__ import annotations

ORCA_SIGNATURE = "* O   R   C   A *"


def is_orca_output(file_name: str) -> bool:
    """Return whether the provided file looks like an ORCA output.

    The implementation intentionally performs a straightforward substring search
    over the file contents, matching the legacy reader behavior.

    Args:
        file_name: Path to the candidate text output file.

    Returns:
        True if the ORCA signature string is found, otherwise False.
    """

    with open(file_name, "r") as f:
        for line in f:
            if ORCA_SIGNATURE in line:
                return True

    return False
