# read_geometry → StructureData

import numpy as np
import numpy.typing as npt


def read_orca5_output_xyz(file_name: str) -> tuple[npt.NDArray[np.str_], npt.NDArray]:
    """Read the final Cartesian coordinates from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(labels, coords)` where:
            * `labels` is an array of atomic symbols (no indices).
            * `coords` is an array of shape `(n_atoms, 3)` in Å.
    """

    labels, coords = [], []

    with open(file_name, "r") as f:
        for line in f:
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                labels, coords = [], []
                line = next(f)
                line = next(f)
                while len(line.split()):
                    labels.append(line.split()[0])
                    coords.append(line.split()[1:])
                    line = next(f)

    coords = [[float(trio[0]), float(trio[1]), float(trio[2])] for trio in coords]

    labels = np.array(labels)
    coords = np.array(coords)

    return labels, coords
