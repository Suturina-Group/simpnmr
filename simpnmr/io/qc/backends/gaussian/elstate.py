def read_gaussian_log_spin(file_name: str) -> int:
    """Read the spin multiplicity (2S+1) from a Gaussian .log file.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        Spin multiplicity (2S+1).
    """

    # Read number of atoms
    with open(file_name, "r") as f:
        for line in f:
            if "Multiplicity =" in line:
                mult = int(line.split()[-1])

    return mult
