import numpy as np
from scipy import constants

from simpnmr.core.main import Susceptibility
from simpnmr.io.qc import qc_readers as rdrs  # TODO move this function here


def load_orca_susc(file_name: str, section: str) -> list:
    """Loads susceptibility tensors from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.
        section: Section identifier to extract (e.g., ``"cas"`` or ``"nev"``).

    Returns:
        A list of susceptibility tensors (one per temperature).
    """

    # Extract all possible susceptibility tensors from ORCA output file
    tensors = rdrs.read_orca_susceptibility(file_name, section)

    # Orca units of XT are cm3 mol-1 K, so convert to Angstrom^3 K
    conv = 1e-24 * constants.Avogadro / (4 * np.pi)
    conv = 1 / conv

    suscs = [
        Susceptibility(tensor / temperature * conv, temperature=temperature)
        for temperature, tensor in tensors.items()
    ]

    return suscs
