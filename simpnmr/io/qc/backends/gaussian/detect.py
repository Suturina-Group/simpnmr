"""Gaussian backend detection.

This module contains small helpers to detect whether a given text output belongs
to the Gaussian quantum-chemistry package.

Detection is intentionally simple and mirrors the legacy logic previously
implemented in `simpnmr.io.qc.readers`.
"""

from __future__ import annotations

GAUSSIAN_SIGNATURE = "Gaussian(R)"
GAUSSIAN_09_SIGNATURE = "Gaussian(R) 09 program"
GAUSSIAN_16_SIGNATURE = "Gaussian(R) 16 program"


def is_gaussian_log(file_name: str) -> bool:
    """Return whether the provided file looks like a Gaussian log.

    The implementation intentionally performs a straightforward substring search
    over the file contents, matching the legacy reader behavior.

    Args:
        file_name: Path to the candidate text output file.

    Returns:
        True if a Gaussian signature string is found, otherwise False.
    """

    with open(file_name, "r") as f:
        for line in f:
            if GAUSSIAN_SIGNATURE in line:
                return True

    return False


def is_gaussian_09(file_name: str) -> bool:
    """Return whether the provided file looks like a Gaussian 09 log.

    This mirrors the legacy reader behavior based on the
    'Gaussian(R) 09 program' marker.
    """

    with open(file_name, "r") as f:
        for line in f:
            if GAUSSIAN_09_SIGNATURE in line:
                return True

    return False


def is_gaussian_16(file_name: str) -> bool:
    """Return whether the provided file looks like a Gaussian 16 log.

    This mirrors the legacy reader behavior based on the
    'Gaussian(R) 16 program' marker.
    """

    with open(file_name, "r") as f:
        for line in f:
            if GAUSSIAN_16_SIGNATURE in line:
                return True

    return False
