"""ORCA backend detection.

This module contains small helpers to detect whether a given text output belongs
to the ORCA quantum-chemistry package.

Detection is intentionally simple and mirrors the legacy logic previously
implemented in `simpnmr.io.qc.readers`.
"""

from __future__ import annotations

import re
from typing import List

ORCA_SIGNATURE = "* O   R   C   A *"

ORCA_A5_SIGNATURE = (
    "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,         ##   #,  ,#"
)
ORCA_A6_SIGNATURE = (
    "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,     #,   #   #,  ,#"
)
# ORCA 6.1+ changed the ASCII banner; fall back to version-string detection.
ORCA_A6_VERSION_SIGNATURE = "Program Version 6."

A_ORB_SIGNATURE = "A(ORB)"

QDPT_WITH_RE = re.compile(r"QDPT WITH\s+(?P<method>[A-Z0-9_+-]+)")


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


def is_orca_a5_output(file_name: str) -> bool:
    with open(file_name, "r") as f:
        for line in f:
            if ORCA_A5_SIGNATURE in line:
                return True
    return False


def is_orca_a6_output(file_name: str) -> bool:
    found_banner = False
    with open(file_name, "r") as f:
        for line in f:
            if ORCA_A6_SIGNATURE in line:
                return True
            if ORCA_A6_VERSION_SIGNATURE in line:
                found_banner = True
    return found_banner


def detect_hfc_has_orb(file_name: str) -> bool:
    """Detect whether an ORCA output contains an orbital (A(ORB)) hyperfine term."""
    with open(file_name, "r", errors="ignore") as f:
        for line in f:
            if A_ORB_SIGNATURE in line:
                return True
    return False


def detect_susc_methods(file_name: str) -> List[str]:
    """Detect available susceptibility methods in an ORCA output.

    The function performs a lightweight scan of the ORCA text output and
    collects all QDPT susceptibility blocks (e.g. CASSCF, NEVPT2).

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        Sorted list of detected method labels in lowercase (e.g. ["casscf", "nevpt2"]).
    """

    methods: set[str] = set()

    with open(file_name, "r", errors="ignore") as f:
        for line in f:
            if "QDPT WITH" not in line:
                continue
            match = QDPT_WITH_RE.search(line)
            if match is None:
                continue
            methods.add(match.group("method").lower())

    return sorted(methods)
