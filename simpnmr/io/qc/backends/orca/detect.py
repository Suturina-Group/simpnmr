"""ORCA backend detection.

This module contains small helpers to detect whether a given text output belongs
to the ORCA quantum-chemistry package.

Detection is intentionally simple and mirrors the legacy logic previously
implemented in `simpnmr.io.qc.readers`.
"""

from __future__ import annotations

ORCA_SIGNATURE = "* O   R   C   A *"
ORCA_PROPERTY_SIGNATURE = "!PROPERTIES!"

ORCA_A5_SIGNATURE = (
    "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,         ##   #,  ,#"
)
ORCA_A6_SIGNATURE = (
    "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,     #,   #   #,  ,#"
)


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


def is_orca_property(file_name: str) -> bool:
    """Return whether the provided file looks like an ORCA property output.

    This mirrors the legacy reader behavior, which distinguished ORCA property
    files by the presence of the "!PROPERTIES!" marker.

    Args:
        file_name: Path to the candidate text output file.

    Returns:
        True if the ORCA property marker is found, otherwise False.
    """

    with open(file_name, "r") as f:
        for line in f:
            if ORCA_PROPERTY_SIGNATURE in line:
                return True

    return False


def is_orca_a5_output(file_name: str) -> bool:
    with open(file_name, "r") as f:
        for line in f:
            if ORCA_A5_SIGNATURE in line:
                return True
    return False


def is_orca_a6_output(file_name: str) -> bool:
    with open(file_name, "r") as f:
        for line in f:
            if ORCA_A6_SIGNATURE in line:
                return True
    return False
