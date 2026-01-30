"""TODO"""

from simpnmr.core.domain.molecule import Molecule
from simpnmr.io.xyz import xyz


def load_molecule_from_xyz(
    file_name: str,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    labels, coords = xyz.load_xyz(file_name)
    return Molecule.from_labels_coords(labels, coords, elements=elements)
