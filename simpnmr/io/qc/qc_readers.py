# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read structures and magnetic properties from quantum-chemistry outputs.

Provides readers and lightweight containers to extract coordinates, shielding,
hyperfine tensors, spin data, susceptibility tensors, and g-tensors from supported
QC program outputs.
"""

# TODO: Refactor in progress — split this module by responsibility and layer

import datetime
import logging
import sys
from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as la
import numpy.typing as npt

from simpnmr.core.utils.strings import remove_letters, remove_numbers
from simpnmr.core.utils.text import subtitle, title
from simpnmr.tools.coords_tools import xyz_format as xyzf

from ...__version__ import __version__

logger = logging.getLogger(__name__)


class QCStructure(ABC):
    """Abstract base class for quantum-chemistry structure readers.

    Subclasses implement `_read` to extract atomic labels and coordinates from a
    supported file type.
    """

    def __init__(self, file_name, labels, coords):
        self.file_name = file_name
        self.labels = labels
        self.coords = coords
        self.n_atoms = len(labels)

        return

    @staticmethod
    def guess_from_file(file_name: str) -> "QCCS":
        """Guess a compatible structure reader and parse the file.

        Args:
            file_name: Path to the file to examine.

        Returns:
            QCCS: Parsed structure object.

        Raises:
            SystemExit: If no supported reader matches the file content.
        """

        SUPPORTED_READERS: list[QCA] = [
            OrcaOutputStructure,
            GaussianLogStructure,
        ]

        data = None

        with open(file_name, "r") as f:
            for line in f:
                for obj in SUPPORTED_READERS:
                    if obj.COMMON_STR in line:
                        # Load quantum chemical hyperfine data
                        data = obj.read(file_name)
                        break

        if data is None:
            sys.exit(f"Cannot find data in {file_name}")

        return data

    "string name of filetype"
    FILETYPE: str

    "string to look for in file which identifies type of file"
    COMMON_STR: str

    "String name of file which has been read"
    file_name: str

    "Number of atoms in system"
    n_atoms: int

    "Atomic labels, with indexing numbers"
    labels: npt.NDArray[np.str_]

    "Atomic coordinates (3xn_atoms)"
    coords: npt.NDArray

    @classmethod
    def read(cls, file_name: str):
        """Read a file using the subclass implementation.

        This method wraps the user-implemented `_read` and validates that the
        required attributes exist on the returned instance.

        Note:
            Do not edit this method.
        """

        instance = cls._read(file_name)

        attributes = [
            "FILETYPE",
            "COMMON_STR",
            "file_name",
            "n_atoms",
            "labels",
            "coords",
        ]

        for attribute in attributes:
            try:
                getattr(instance, attribute)
            except AttributeError:
                sys.exit(
                    "ERROR: Attribute {} is missing from {}".format(attribute, cls)
                )

        return instance

    @classmethod
    @abstractmethod
    def _read(file_name: str):
        """Parse a QC file and construct a structure instance.

        Args:
            file_name: Path to the QC output file.

        Returns:
            QCStructure: Parsed structure instance.
        """
        raise NotImplementedError


class GaussianLogStructure(QCStructure):
    """
    Structure object for Gaussian LOG files
    """

    FILETYPE = "Gaussian LOG"

    COMMON_STR = "Gaussian(R)"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        labels, coords = read_gaussian_log_xyz(file_name)
        labels = np.array(xyzf.add_label_indices(labels))

        return cls(file_name, labels, coords)


class OrcaOutputStructure(QCStructure):
    """
    Structure object for Orca Output files
    """

    FILETYPE = "Orca OUTPUT"

    COMMON_STR = "* O   R   C   A *"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        old_labels, coords = read_orca5_output_xyz(file_name)
        old_labels = np.array(
            xyzf.add_label_indices(old_labels, style="sequential", start_index=0)
        )
        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )

        return cls(file_name, new_labels, coords)


class QCCS(ABC):
    """Abstract base class for chemical shielding readers.

    Subclasses implement `_read` to extract isotropic and anisotropic chemical
    shielding values together with atomic labels and coordinates.
    """

    def __init__(self, file_name, labels, coords, cs_iso, cs_aniso, cs_units):
        """Initialize a chemical shielding container.

        Args:
            file_name: Source file name.
            labels: Atom labels (with indices).
            coords: Atomic coordinates as an (n_atoms, 3) array.
            cs_iso: Isotropic chemical shielding values by label.
            cs_aniso: Anisotropic chemical shielding values by label.
            cs_units: Units for shielding values.
        """

        self.file_name = file_name
        self.labels = labels
        self.coords = coords
        self.n_atoms = len(labels)
        self.cs_iso = cs_iso
        self.cs_aniso = cs_aniso
        self.cs_units = cs_units

        return

    @staticmethod
    def guess_from_file(file_name: str) -> "QCCS":
        """Guess a compatible shielding reader and parse the file.

        Args:
            file_name: Path to the file to examine.

        Returns:
            QCCS: Parsed chemical shielding object.

        Raises:
            SystemExit: If no supported reader matches the file content.
        """

        SUPPORTED_CS_OBJ: list[QCA] = [
            OrcaOutputCS,
            OrcaPropertyCS,
            Gaussian09LogCS,
            Gaussian16LogCS,
        ]

        data = None

        with open(file_name, "r") as f:
            for line in f:
                for obj in SUPPORTED_CS_OBJ:
                    if obj.COMMON_STR in line:
                        # Load quantum chemical hyperfine data
                        data = obj.read(file_name)
                        break

        if data is None:
            sys.exit(f"Cannot find data in {file_name}")

        return data

    def __str__(self):
        """Return a human-readable representation of the parsed shielding data."""

        string = ""

        string += title("Quantum Chemistry Chemical Shielding Data")

        string += "Data was read from: {}\n".format(self.file_name)

        string += "As filetype: {}\n".format(self.FILETYPE)

        string += subtitle("Coordinates (Å)")

        for label, coord in zip(self.labels, self.coords):
            string += "{:5}  {: 10.6f}  {: 10.6f}  {: 10.6f}\n".format(label, *coord)

        string += subtitle("Isotropic Chemical Shielding ({})".format(self.cs_units))

        for label, val in self.cs_iso.items():
            string += "{:5} {: .6f}\n".format(label, val)

        string += subtitle("Anisotropic Chemical Shielding ({})".format(self.cs_units))

        for label, val in self.cs_aniso.items():
            string += "{:5} {: .6f}\n".format(label, val)

        return string

    "string name of filetype"
    FILETYPE: str

    "string to look for in file which identifies type of file"
    COMMON_STR: str

    "String name of file which has been read"
    file_name: str

    "Number of atoms in system"
    n_atoms: int

    "Atomic labels, with indexing numbers"
    labels: npt.NDArray[np.str_]

    "Atomic coordinates (3xn_atoms)"
    coords: npt.NDArray

    "Isotropic Chemical Shielding values"
    cs_iso: dict[str, float]

    "Anisotropic Chemical Shielding values"
    cs_aniso: dict[str, float]

    """
    Units of Isotropic Chemical Shielding (cs)
    """
    cs_units: str

    @classmethod
    def read(cls, file_name: str):
        """Read a file using the subclass implementation.

        This method wraps the user-implemented `_read` and validates that the
        required attributes exist on the returned instance.

        Note:
            Do not edit this method.
        """

        instance = cls._read(file_name)

        attributes = [
            "FILETYPE",
            "COMMON_STR",
            "file_name",
            "n_atoms",
            "labels",
            "coords",
            "cs_iso",
            "cs_aniso",
            "cs_units",
        ]

        for attribute in attributes:
            try:
                getattr(instance, attribute)
            except AttributeError:
                sys.exit(
                    "ERROR: Attribute {} is missing from {}".format(attribute, cls)
                )

        return instance

    @classmethod
    @abstractmethod
    def _read(file_name: str):
        """Parse a QC file and construct a shielding instance.

        Args:
            file_name: Path to the QC output file.

        Returns:
            QCCS: Parsed shielding instance.
        """
        raise NotImplementedError


class OrcaOutputCS(QCCS):
    """
    Chemical Shielding object for Orca OUTPUT files
    """

    FILETYPE = "Orca OUTPUT"

    COMMON_STR = "* O   R   C   A *"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        old_labels, coords = read_orca5_output_xyz(file_name)
        old_labels = np.array(
            xyzf.add_label_indices(old_labels, style="sequential", start_index=0)
        )
        cs_iso, cs_aniso = read_orca5_output_cs(file_name)

        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )

        converter = {ol: nl for ol, nl in zip(old_labels, new_labels)}

        cs_iso = {converter[label]: val for label, val in cs_iso.items()}

        cs_aniso = {converter[label]: val for label, val in cs_aniso.items()}

        cs_units = "ppm"

        return cls(file_name, new_labels, coords, cs_iso, cs_aniso, cs_units)


class OrcaPropertyCS(QCCS):
    """
    Chemical Shielding object for Orca PROPERTY files
    """

    FILETYPE = "Orca PROPERTY"

    COMMON_STR = "!PROPERTIES!"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        old_labels, coords = read_orca5_property_xyz(file_name)
        cs_iso, cs_aniso = read_orca5_property_cs(file_name)

        # Convert orca labelling 1-> natoms for all atoms
        # to 1-n_atoms per element
        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        cs_iso = {converter[label]: value for label, value in cs_iso.items()}
        cs_aniso = {converter[label]: tensor for label, tensor in cs_aniso.items()}

        cs_units = "ppm"

        return cls(file_name, new_labels, coords, cs_iso, cs_aniso, cs_units)


class Gaussian16LogCS(QCCS):
    """
    Chemical Shielding object for Gaussian LOG files
    """

    FILETYPE = "Gaussian LOG"

    COMMON_STR = "Gaussian(R) 16 program"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        labels, coords = read_gaussian_log_xyz(file_name)
        labels = np.array(xyzf.add_label_indices(labels))
        cs_iso, cs_aniso = read_gaussian16_log_cs(file_name)

        cs_units = "ppm"

        return cls(file_name, labels, coords, cs_iso, cs_aniso, cs_units)


class Gaussian09LogCS(QCCS):
    """
    Chemical Shielding object for Gaussian LOG files
    """

    FILETYPE = "Gaussian LOG"

    COMMON_STR = "Gaussian(R) 09 program"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        labels, coords = read_gaussian_log_xyz(file_name)
        labels = np.array(xyzf.add_label_indices(labels))
        cs_iso, cs_aniso = read_gaussian09_log_cs(file_name)

        cs_units = "ppm"

        return cls(file_name, labels, coords, cs_iso, cs_aniso, cs_units)


class QCSpin(ABC):
    """Abstract base class for spin data readers."""

    def __init__(self, file_name: str, S: float, multiplicity: int):
        self.file_name = file_name
        self.S = S
        self.multiplicity = multiplicity

    @staticmethod
    def guess_from_file(file_name: str) -> "QCSpin":
        SUPPORTED_SPIN_OBJS: list[type["QCSpin"]] = [GaussianLogSpin, OrcaSpin]

        data = None
        with open(file_name, "r") as f:
            for line in f:
                for obj in SUPPORTED_SPIN_OBJS:
                    if obj.COMMON_STR in line:
                        data = obj.read(file_name)
                        break
                if data is not None:
                    break
        if data is None:
            sys.exit(f"Cannot find spin data in {file_name}")

        return data

    FILETYPE: str
    COMMON_STR: str
    file_name: str
    S: float
    multiplicity: int | None

    @classmethod
    def read(cls, file_name: str) -> "QCSpin":
        instance = cls._read(file_name)
        for attribute in ["FILETYPE", "COMMON_STR", "file_name", "S", "multiplicity"]:
            try:
                getattr(instance, attribute)
            except AttributeError:
                sys.exit(f"Attribute {attribute} is missing from {cls}")
        return instance

    @classmethod
    @abstractmethod
    def _read(cls, file_name: str) -> "QCSpin":
        raise NotImplementedError


class GaussianLogSpin(QCSpin):
    """
    Spin object for Gaussian LOG files
    """

    FILETYPE = "Gaussian LOG"
    COMMON_STR = "Gaussian(R)"

    @classmethod
    def _read(cls, file_name: str) -> "GaussianLogSpin":
        multiplicity = read_gaussian_log_spin(file_name)
        S = (multiplicity - 1) / 2.0
        return cls(file_name, S, multiplicity)


class OrcaSpin(QCSpin):
    """
    Spin object for Orca OUTPUT files
    """

    FILETYPE = "Orca OUTPUT"
    COMMON_STR = "* O   R   C   A *"

    @classmethod
    def _read(cls, file_name: str) -> "OrcaSpin":
        S = read_orca_spin(file_name)
        multiplicity = int(2 * S + 1)
        return cls(file_name, S, multiplicity)


class QCA(ABC):
    """Abstract base class for hyperfine (A-tensor) readers."""

    def __init__(self, file_name, labels, coords, a_iso, a_dip, a_units):
        """Initialize a hyperfine (A-tensor) container.

        Args:
            file_name: Source file name.
            labels: Atom labels (with indices).
            coords: Atomic coordinates as an (n_atoms, 3) array.
            a_iso: Isotropic hyperfine couplings by label.
            a_dip: Dipolar (traceless) hyperfine tensors by label.
            a_units: Units for hyperfine values.
        """

        self.file_name = file_name
        self.labels = labels
        self.coords = coords
        self.n_atoms = len(labels)
        self.a_iso = a_iso
        self.a_dip = a_dip
        self.a_units = a_units

        return

    @staticmethod
    def guess_from_file(file_name: str) -> "QCA":
        """Guess a compatible hyperfine reader and parse the file.

        Args:
            file_name: Path to the file to examine.

        Returns:
            QCA: Parsed hyperfine (A-tensor) object.

        Raises:
            SystemExit: If no supported reader matches the file content.
        """

        SUPPORTED_A_OBJS: list[QCA] = [
            GaussianLogA,
            Orca5OutputA,
            Orca6OutputA,
            Orca5PropertyA,
        ]

        with open(file_name, "r") as f:
            for line in f:
                for obj in SUPPORTED_A_OBJS:
                    if obj.COMMON_STR in line:
                        # Load quantum chemical hyperfine data
                        data = obj.read(file_name)
                        break

        if data is None:
            sys.exit(f"Cannot find data in {file_name}")

        return data

    def save_to_csv(
        self,
        file_name: str = "dft_hyperfines.csv",
        verbose: bool = True,
        comment: str = "",
        delimiter: str = ",",
    ) -> None:
        """Save hyperfine data to a CSV file.

        Args:
            file_name: Output CSV file name.
            verbose: If True, prints the output file name.
            comment: Optional additional comment line (including comment marker).
            delimiter: Delimiter used in the CSV.
        """

        # Save hyperfine data to file
        out = np.array(
            [
                "{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                    label, iso, *tensor[0, :], *tensor[1, 1:], tensor[2, 2]
                )
                for iso, (label, tensor) in zip(self.a_iso.values(), self.a_dip.items())
            ]
        )

        _comments = (
            f"#This file was generated with SimpNMR v{__version__} on {{}}\n".format(
                datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
            )
        )

        _comments += comment + "\n"

        header = (
            f"atom_label, "
            f"Aiso ({self.a_units}), "
            f"Adip_xx ({self.a_units}), "
            f"Adip_xy ({self.a_units}), "
            f"Adip_xz ({self.a_units}), "
            f"Adip_yy ({self.a_units}), "
            f"Adip_yz ({self.a_units}), "
            f"Adip_zz ({self.a_units})"
        )

        # Save to file
        np.savetxt(
            file_name,
            out,
            delimiter=delimiter,
            header=header,
            fmt="%s",
            comments=_comments,
        )

        if verbose:
            logger.info("Raw DFT Hyperfine data written to %s", file_name)

        return

    def __str__(self):
        """Return a human-readable representation of the parsed hyperfine data."""

        string = ""

        string += title("Quantum Chemistry Hyperfine Data")

        string += "Data was read from: {}\n".format(self.file_name)

        string += "As filetype: {}\n".format(self.FILETYPE)

        string += subtitle("Coordinates (Å)")

        for label, coord in zip(self.labels, self.coords):
            string += "{:5}  {: 10.6f}  {: 10.6f}  {: 10.6f}\n".format(label, *coord)

        string += subtitle("Isotropic A values ({})".format(self.a_units))

        for label, val in self.a_iso.items():
            string += "{:5} {: .6f}\n".format(label, val)

        string += subtitle("Anisotropic (dipolar) A Tensor ({})".format(self.a_units))

        for label, tensor in self.a_dip.items():
            string += "\n      {: .6f} {: .6f} {: .6f}\n".format(*tensor[0])
            string += "{:5} {: .6f} {: .6f} {: .6f}\n".format(label, *tensor[1])
            string += "      {: .6f} {: .6f} {: .6f}\n".format(*tensor[2])

        return string

    "string name of filetype"
    FILETYPE: str

    "string to look for in file which identifies type of file"
    COMMON_STR: str

    "String name of file which has been read"
    file_name: str

    "Number of atoms in system"
    n_atoms: int

    "Atomic labels, with indexing numbers"
    labels: npt.NDArray[np.str_]

    "Atomic coordinates (3xn_atoms)"
    coords: npt.NDArray

    "Isotropic Hyperfine coupling values"
    a_iso: dict[str, float]

    """Anisotropic (dipolar) Hyperfine coupling tensors (traceless)
    keys are string label with index number, values are (3x3) arrays"""
    a_dip: dict[str, npt.NDArray]

    """
    Units of A_iso and A_dip
    """
    a_units: str

    @classmethod
    def read(cls, file_name: str):
        """Read a file using the subclass implementation.

        This method wraps the user-implemented `_read` and validates that the
        required attributes exist on the returned instance.

        Note:
            Do not edit this method.
        """

        instance = cls._read(file_name)

        attributes = [
            "FILETYPE",
            "COMMON_STR",
            "file_name",
            "n_atoms",
            "labels",
            "coords",
            "a_iso",
            "a_dip",
            "a_units",
        ]

        for attribute in attributes:
            try:
                getattr(instance, attribute)
            except AttributeError:
                sys.exit(
                    "ERROR: Attribute {} is missing from {}".format(attribute, cls)
                )

        return instance

    @classmethod
    @abstractmethod
    def _read(file_name: str):
        """Parse a QC file and construct a hyperfine instance.

        Args:
            file_name: Path to the QC output file.

        Returns:
            QCA: Parsed hyperfine (A-tensor) instance.
        """
        raise NotImplementedError


class GaussianLogA(QCA):
    """
    A QCA object for Gaussian LOG files
    """

    FILETYPE = "Gaussian LOG"

    COMMON_STR = "Gaussian(R)"

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        labels, coords = read_gaussian_log_xyz(file_name)
        labels = np.array(xyzf.add_label_indices(labels))
        a_iso_raw, a_dip_raw = read_gaussian_log_a_tensors(file_name)

        mult = read_gaussian_log_spin(file_name)
        n_unpaired = mult - 1

        # Convert to dict
        a_iso = {label: val for label, val in zip(labels, a_iso_raw)}

        # Convert to dict
        # and normalise by number of unpaired electrons
        a_dip = {
            label: tensor * 1.0 / n_unpaired for label, tensor in zip(labels, a_dip_raw)
        }

        a_units = "MHz"

        return cls(file_name, labels, coords, a_iso, a_dip, a_units)


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


def read_gaussian_log_a_tensors(file_name: str) -> tuple[npt.NDArray, npt.NDArray]:
    """Extract isotropic and dipolar hyperfine (A) tensors from a Gaussian log.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` is an array of shape `(n_atoms,)` with isotropic values in MHz.
            * `a_dip` is an array of shape `(n_atoms, 3, 3)` with dipolar
            tensors in MHz.
    """

    # Read number of atoms
    with open(file_name, "r") as f:
        for line in f:
            if "NAtoms=" in line:
                spl_line = line.split()
                n_atoms = int(spl_line[spl_line.index("NAtoms=") + 1])
                break

    a_iso = np.zeros(n_atoms)

    # Read isotropic part
    with open(file_name, "r") as f:
        for line in f:
            if "Isotropic Fermi Contact Couplings" in line:
                line = next(f)
                for it in range(n_atoms):
                    line = next(f)
                    a_iso[it] = float(line.split()[3])  # MHz

    a_dip = np.zeros([n_atoms, 3, 3])
    # Read traceless tensor as eigenvalues and eigenvectors
    track = 0
    with open(file_name, "r") as f:
        for line in f:
            if "Anisotropic Spin Dipole Couplings" in line:
                track += 1
                # Make sure in spin density part!
            if "Anisotropic Spin Dipole Couplings" in line and track == 2:
                line = next(f)
                line = next(f)
                line = next(f)
                line = next(f)
                for it in range(n_atoms):
                    line = next(f)
                    val_1 = float(line.split()[2])  # MHz
                    vecs_1 = [float(val) for val in line.split()[-3:]]
                    line = next(f)
                    val_2 = float(line.split()[4])  # MHz
                    vecs_2 = [float(val) for val in line.split()[-3:]]
                    line = next(f)
                    val_3 = float(line.split()[2])  # MHz
                    vecs_3 = [float(val) for val in line.split()[-3:]]
                    vals = np.array([val_1, val_2, val_3])
                    vecs = np.array([vecs_1, vecs_2, vecs_3]).T

                    # Transform back to coordinate frame in MHz
                    a_dip[it, :, :] = vecs @ np.diag(vals) @ la.inv(vecs)
                    line = next(f)

    if track != 2:
        logger.warning(
            (
                "Cannot find Dipolar Hyperfine Tensor in log file \n"
                "Check prop=epr is in routecard!"
            )
        )

    return a_iso, a_dip


class Orca5OutputA(QCA):
    """
    A Tensor object for Orca 5 OUTPUT files
    """

    FILETYPE = "Orca OUTPUT"

    COMMON_STR = (
        "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,         ##   #,  ,#"
    )

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        old_labels, coords = read_orca5_output_xyz(file_name)
        old_labels = np.array(
            xyzf.add_label_indices(old_labels, style="sequential", start_index=0)
        )
        a_iso, a_dip = read_orca5_output_a_tensors(file_name)

        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        a_iso = {converter[label]: value for label, value in a_iso.items()}
        a_dip = {converter[label]: tensor for label, tensor in a_dip.items()}

        a_units = "MHz"

        return cls(file_name, new_labels, coords, a_iso, a_dip, a_units)


class Orca6OutputA(QCA):
    """
    A Tensor object for Orca 6 OUTPUT files
    """

    FILETYPE = "Orca OUTPUT"

    COMMON_STR = (
        "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,     #,   #   #,  ,#"
    )

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        old_labels, coords = read_orca5_output_xyz(file_name)
        old_labels = np.array(
            xyzf.add_label_indices(old_labels, style="sequential", start_index=0)
        )
        a_iso, a_dip = read_orca6_output_a_tensors(file_name)

        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        a_iso = {converter[label]: value for label, value in a_iso.items()}
        a_dip = {converter[label]: tensor for label, tensor in a_dip.items()}

        a_units = "MHz"

        return cls(file_name, new_labels, coords, a_iso, a_dip, a_units)


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


def read_orca6_output_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract hyperfine (A) tensors from an ORCA 6 output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dip` maps atom labels to 3x3 traceless dipolar tensors in MHz.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                n_calcd = int(line.split()[5][1:])

    a_iso = {}
    a_dip = {}

    # Read hyperfine data
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                for _ in range(n_calcd):
                    while "Nucleus" not in line:
                        line = next(f)
                    tmp = line.split()[1]
                    label = "{}{}".format(remove_numbers(tmp), remove_letters(tmp))
                    for _ in range(8):
                        line = next(f)

                    # Raw matrix in MHz
                    row_1 = [float(val) for val in line.split()]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()]

                    full = np.array([row_1, row_2, row_3])

                    a_iso[label] = 1 / 3 * np.trace(full)
                    a_dip[label] = full - np.eye(3) * a_iso[label]

    return a_iso, a_dip


def read_orca5_output_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract hyperfine (A) tensors from an ORCA 5 output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dip` maps atom labels to 3x3 traceless dipolar tensors in MHz.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "Number of nuclei for epr/nmr" in line:
                n_calcd = int(line.split()[-1])

    a_iso = {}
    a_dip = {}

    # Read hyperfine data
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                line = next(f)
                line = next(f)
                line = next(f)
                for it in range(n_calcd):
                    line = next(f)
                    tmp = line.split()[1]
                    label = "{}{}".format(remove_numbers(tmp), remove_letters(tmp))
                    for _ in range(5):
                        line = next(f)

                    # Raw matrix in MHz
                    row_1 = [float(val) for val in line.split()]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()]

                    for _ in range(5):
                        line = next(f)
                    a_iso[label] = float(line.split()[-1])

                    for _ in range(9):
                        line = next(f)

                    full = np.array([row_1, row_2, row_3])

                    a_dip[label] = full - np.eye(3) * a_iso[label]

    return a_iso, a_dip


def read_orca5_output_cs(
    file_name: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract chemical shielding values from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}
    # Read Chemical Shielding data
    with open(file_name, "r") as f:
        for line in f:
            if "CHEMICAL SHIELDING SUMMARY (ppm)" in line:
                for _ in range(6):
                    line = next(f)

                while len(line.lstrip().rstrip()):
                    label = "{}{}".format(line.split()[1], int(line.split()[0]))
                    cs_iso[label] = float(line.split()[2])
                    cs_aniso[label] = float(line.split()[3])
                    line = next(f)

    return cs_iso, cs_aniso


class Orca5PropertyA(QCA):
    """
    A Tensor object for Orca PROPERTY files
    """

    FILETYPE = "Orca PROPERTY"

    COMMON_STR = (
        "            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,         ##   #,  ,#"
    )

    @classmethod
    def _read(cls, file_name: str):
        # Read raw data
        old_labels, coords = read_orca5_property_xyz(file_name)
        a_iso, a_dip = read_orca5_property_a_tensors(file_name)

        # Convert orca labelling 1-> natoms for all atoms
        # to 1-n_atoms per element
        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        a_iso = {converter[label]: value for label, value in a_iso.items()}
        a_dip = {converter[label]: tensor for label, tensor in a_dip.items()}

        a_units = "MHz"

        return cls(file_name, new_labels, coords, a_iso, a_dip, a_units)


def read_orca5_property_xyz(file_name: str) -> tuple[npt.NDArray[np.str_], npt.NDArray]:
    """Read the final Cartesian coordinates from an ORCA property file.

    Args:
        file_name: Path to the ORCA property file.

    Returns:
        A tuple `(labels, coords)` where:
            * `labels` is an array of atom labels with indices.
            * `coords` is an array of shape `(n_atoms, 3)` in Å.
    """

    labels, coords = [], []

    with open(file_name, "r") as f:
        for line in f:
            if "!GEOMETRY!" in line:
                line = next(f)
                n_atoms = int(line.split()[-1])
                for _ in range(2):
                    line = next(f)
                for _ in range(n_atoms):
                    line = next(f)
                    labels.append("{}{}".format(line.split()[1], line.split()[0]))
                    coords.append(line.split()[2:])

    coords = [[float(trio[0]), float(trio[1]), float(trio[2])] for trio in coords]

    labels = np.array(labels)
    coords = np.array(coords)

    return labels, coords


def read_orca5_property_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Read hyperfine coupling tensors from an ORCA property file.

    Args:
        file_name: Path to the ORCA property file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dip` maps atom labels to 3x3 traceless dipolar tensors in MHz.
    """

    a_dip = {}
    a_iso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "EPRNMR_ATensor" in line:
                while "Number of stored nuclei" not in line:
                    line = next(f)
                n_calcd = int(line.split()[4])
                while "Nucleus:" not in line:
                    line = next(f)
                for _ in range(n_calcd):
                    label = "{}{}".format(line.split()[2], line.split()[1])
                    for _ in range(6):
                        line = next(f)
                    # Raw values
                    row_1 = [float(val) for val in line.split()[1:]]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()[1:]]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()[1:]]
                    a_dip[label] = np.array([row_1, row_2, row_3])
                    for _ in range(9):
                        line = next(f)
                    # Isotropic value
                    a_iso[label] = float(line.split()[-1])
                    a_dip[label] -= np.eye(3) * a_iso[label]
                    line = next(f)

    return a_iso, a_dip


def read_orca5_property_cs(
    file_name: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Read chemical shielding data from an ORCA property file.

    Args:
        file_name: Path to the ORCA property file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "EPRNMR_OrbitalShielding" in line:
                while "Number of stored nuclei" not in line:
                    line = next(f)
                n_calcd = int(line.split()[4])
                while "Nucleus:" not in line:
                    line = next(f)
                for _ in range(n_calcd):
                    label = "{}{}".format(line.split()[2], line.split()[1])
                    for _ in range(13):
                        line = next(f)
                    # Read eigenvalues and convert to Anisotropic CS
                    evals = np.array([float(val) for val in line.split()[1:]])
                    evals = sorted(evals)
                    cs_aniso[label] = evals[2] - (evals[0] + evals[1]) / 2.0
                    line = next(f)
                    # Isotropic value
                    cs_iso[label] = float(line.split()[-1])
                    line = next(f)

    return cs_iso, cs_aniso


def read_gaussian09_log_cs(file_name):
    """Read chemical shielding data from a Gaussian 09 log file.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "Magnetic shielding tensor (ppm)" in line:
                while "Number of stored nuclei" not in line:
                    line = next(f)
                n_calcd = int(line.split()[4])
                while "Nucleus:" not in line:
                    line = next(f)
                for _ in range(n_calcd):
                    label = "{}{}".format(line.split()[2], line.split()[1])
                    for _ in range(13):
                        line = next(f)
                    # Read eigenvalues and convert to Anisotropic CS
                    evals = np.array([float(val) for val in line.split()[1:]])
                    evals = sorted(evals)
                    cs_aniso[label] = evals[2] - (evals[0] + evals[1]) / 2.0
                    line = next(f)
                    # Isotropic value
                    cs_iso[label] = float(line.split()[-1])
                    line = next(f)

    return cs_iso, cs_aniso


def read_gaussian16_log_cs(file_name):
    """Read chemical shielding data from a Gaussian 16 log file.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(cs_iso, cs_aniso)` where:
            * `cs_iso` maps atom labels to isotropic shielding in ppm.
            * `cs_aniso` maps atom labels to anisotropic shielding in ppm.
    """

    cs_iso = {}
    cs_aniso = {}

    with open(file_name, "r") as f:
        for line in f:
            if "Isotropic =" in line:
                line = line.replace("=-", "= -")
                cs_iso["{}{:d}".format(line.split()[1], int(line.split()[0]))] = float(
                    line.split()[4]
                )

    return cs_iso, cs_aniso


def read_orca_susceptibility(file_name: str, section: str) -> dict[float, np.ndarray]:
    """Extract temperature-dependent molar magnetic susceptibility tensors.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        Dictionary mapping temperature in K to a 3x3 susceptibility tensor.
    """

    susceptibilities = {}

    with open(file_name, "r") as f:
        for line in f:
            if f"QDPT WITH {section.upper()}" in line:
                while (
                    "TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR"
                    not in line
                ):
                    line = next(f)
                # Move down until we reach the first temperature header line
                while "TEMPERATURE/K" not in line:
                    line = next(f)
                while "TEMPERATURE/K" in line:
                    _temp = float(line.split("TEMPERATURE/K:")[1])
                    line = next(f)
                    line = next(f)
                    # Read tensor
                    row_1 = [float(val) for val in line.split()]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()]
                    susceptibilities[_temp] = np.array([row_1, row_2, row_3])
                    line = next(f)
                    line = next(f)

    return susceptibilities


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
        RuntimeError: If a multiplicity cannot be determined from the file.
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
        raise RuntimeError(
            f"Could not determine spin multiplicity from ORCA output '{file_name}'"
        )

    return spin


def read_orca_g_tensor(file_name: str, section: str) -> np.ndarray | None:
    """Extract the electronic g-tensor from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        A 3x3 g-tensor as a NumPy array if found, otherwise None.
    """

    g_tensor = None

    try:
        with open(file_name, "r") as f:
            for line in f:
                # Find the correct QDPT section
                if f"QDPT WITH {section.upper()}" in line:
                    # Go down to the G-matrix header
                    for line in f:
                        if "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN" in line:
                            break
                    # Find "g-matrix:"
                    for line in f:
                        if "g-matrix:" in line:
                            # Next three lines are the rows of the tensor
                            row_1 = [float(val) for val in next(f).split()]
                            row_2 = [float(val) for val in next(f).split()]
                            row_3 = [float(val) for val in next(f).split()]
                            g_tensor = np.array([row_1, row_2, row_3])
                            break
                    break
    except Exception as e:
        logger.warning(
            "Failed to parse ORCA g-tensor — proceeding without g-tensor: %s",
            e,
        )

    return g_tensor


def read_eff_hamiltonian_tensor(file_name: str, section: str) -> np.ndarray | None:
    """Extract the raw effective Hamiltonian tensor from an ORCA output file.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        A 3x3 effective Hamiltonian tensor in cm-1 as a NumPy array if found,
        otherwise None.
    """

    eff_H_raw = None

    with open(file_name, "r") as f:
        for line in f:
            # Find the correct QDPT section
            if f"QDPT WITH {section.upper()}" in line:
                # Go down to the G-matrix header
                for line in f:
                    if (
                        "Effective Hamiltonian from projected relativistic states "
                        "and relativistic energies:" in line
                    ):
                        break
                # Find "Raw matrix"
                for line in f:
                    if "Raw matrix (cm-1):" in line:
                        # Next three lines are the rows of the tensor
                        row_1 = [float(val) for val in next(f).split()]
                        row_2 = [float(val) for val in next(f).split()]
                        row_3 = [float(val) for val in next(f).split()]
                        eff_H_raw = np.array([row_1, row_2, row_3])
                        break
                break

    return eff_H_raw
