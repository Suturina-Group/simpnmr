import numpy as np
import numpy.typing as npt

from simpnmr.tools.coords import xyz_fmt as xyzf


def read_gaussian_log_xyz(file_name: str) -> tuple[npt.NDArray[np.str_], npt.NDArray]:
    """Read atomic labels and coordinates from a Gaussian .log file.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(labels, coords)` where:
            * `labels` is an array of atomic symbols with length `n_atoms`.
            * `coords` is an array of shape `(n_atoms, 3)` with coordinates in Å.
    """

    # Read number of atoms
    with open(file_name, "r") as f:
        for line in f:
            if "NAtoms=" in line:
                spl_line = line.split()
                n_atoms = int(spl_line[spl_line.index("NAtoms=") + 1])
                break

    # Get coordinates
    headers = ["Standard orientation:", "Input orientation:"]
    with open(file_name, "r") as f:
        for line in f:
            if any([he in line for he in headers]):
                coords = []
                a_nums = []

                # Skip header
                for _ in range(4):
                    line = next(f)

                for _ in range(n_atoms):
                    line = next(f)
                    coords.append([float(coord) for coord in line.split()[3:]])
                    a_nums.append(int(line.split()[1]))

    f.close()

    # Convert atomic numbers to atomic labels
    labels = xyzf.num_to_lab(a_nums)

    labels = np.asarray(labels)
    coords = np.asarray(coords)

    return labels, coords
