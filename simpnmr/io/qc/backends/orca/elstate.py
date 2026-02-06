# read electornic state data

from simpnmr.io.qc.errors import DataNotFoundError


def read_orca_spin(file_name: str) -> float:
    """Read the spin quantum number S from an ORCA output file.

    This parses an input-style line in the output and supports both:
        * `* xyz charge mult`
        * `* xyzfile charge mult filename.xyz`

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        Spin quantum number S, derived from the spin multiplicity (2S+1).

    Raises:
        DataNotFoundError: If a multiplicity cannot be determined from the file.
    """
    spin = None

    with open(file_name, "r") as f:
        for line in f:
            # Normalise whitespace to make matching robust
            compact = line.replace(" ", "").lower()
            if "*xyz" in compact:
                # Example lines:
                #   * xyz 0 2
                #   * xyzfile 0 2 ptbu3_opt_solv_optim.xyz
                tokens = line.split()
                # Collect all integer tokens (charge, multiplicity, etc.)
                int_tokens = []
                for tok in tokens:
                    stripped = tok.lstrip("+-")
                    if stripped.isdigit():
                        int_tokens.append(int(tok))
                if len(int_tokens) >= 2:
                    mult = int_tokens[1]  # second integer is multiplicity
                    spin = (mult - 1) / 2.0
                    break

    if spin is None:
        raise DataNotFoundError(
            message="Could not determine spin multiplicity from ORCA output",
            path=file_name,
            backend="orca",
            kind="spin",
        )

    return spin
