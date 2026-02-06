# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read structures and magnetic properties from quantum-chemistry outputs.

Provides readers and lightweight containers to extract coordinates, shielding,
hyperfine tensors, spin data, susceptibility tensors, and g-tensors from supported
QC program outputs.
"""

# TODO: Refactor in progress — split this module by responsibility and layer

import logging
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from simpnmr.core.util.text import subtitle, title
from simpnmr.io.qc.backends.gaussian.detect import (  # noqa
    GAUSSIAN_09_SIGNATURE,
    GAUSSIAN_16_SIGNATURE,
    GAUSSIAN_SIGNATURE,
    is_gaussian_09,
    is_gaussian_16,
    is_gaussian_log,
)
from simpnmr.io.qc.backends.gaussian.elstate import read_gaussian_log_spin  # noqa
from simpnmr.io.qc.backends.gaussian.geom import read_gaussian_log_xyz  # noqa
from simpnmr.io.qc.backends.gaussian.hfc import read_gaussian_log_a_tensors  # noqa
from simpnmr.io.qc.backends.gaussian.shield import (  # noqa
    read_gaussian09_log_cs,
    read_gaussian16_log_cs,
)
from simpnmr.io.qc.backends.orca.detect import (
    ORCA_A5_SIGNATURE,  # noqa
    ORCA_A6_SIGNATURE,  # noqa
    ORCA_SIGNATURE,  # noqa
    is_orca_a5_output,  # noqa
    is_orca_a6_output,  # noqa
    is_orca_output,  # noqa
    is_orca_property,  # noqa
)
from simpnmr.io.qc.backends.orca.elstate import read_orca_spin  # noqa
from simpnmr.io.qc.backends.orca.geom import (  # noqa
    read_orca5_output_xyz,
    read_orca5_property_xyz,
)
from simpnmr.io.qc.backends.orca.gtensor import read_orca_g_tensor  # noqa
from simpnmr.io.qc.backends.orca.ham import read_eff_hamiltonian_tensor  # noqa
from simpnmr.io.qc.backends.orca.hfc import (  # noqa
    read_orca5_output_a_tensors,
    read_orca5_property_a_tensors,
    read_orca6_output_a_tensors,
)
from simpnmr.io.qc.backends.orca.shield import (  # noqa
    read_orca5_output_cs,
    read_orca5_property_cs,
)
from simpnmr.io.qc.backends.orca.susc import read_orca_susceptibility  # noqa
from simpnmr.io.qc.errors import (
    ReaderContractError,
    UnsupportedFileError,
)
from simpnmr.tools.coords import xyz_fmt as xyzf

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
    def guess_from_file(file_name: str) -> "QCStructure":
        """Guess a compatible structure reader and parse the file.

        Args:
            file_name: Path to the file to examine.

        Returns:
            QCStructure: Parsed structure object.

        Raises:
            UnsupportedFileError: If no supported reader matches the file content.
        """

        if is_orca_output(file_name):
            return OrcaOutputStructure.read(file_name)

        if is_gaussian_log(file_name):
            return GaussianLogStructure.read(file_name)

        raise UnsupportedFileError(
            message="Unsupported QC file for geometry "
            "reader (no known signature found)",
            path=file_name,
            kind="geom",
        )

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
            except AttributeError as e:
                raise ReaderContractError(
                    message=(
                        f"Reader contract violation: '{cls.__name__}' is missing "
                        f"required attribute '{attribute}'"
                    ),
                    path=file_name,
                    kind="geom",
                    details={"class": cls.__name__, "attribute": attribute},
                ) from e

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

    COMMON_STR = GAUSSIAN_SIGNATURE

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

    COMMON_STR = ORCA_SIGNATURE

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
        # Stage 1: detect backend.
        if is_orca_output(file_name):
            # ORCA: distinguish PROPERTY vs OUTPUT using the legacy marker.
            if is_orca_property(file_name):
                return OrcaPropertyCS.read(file_name)

            return OrcaOutputCS.read(file_name)

        if is_gaussian_log(file_name):
            # Gaussian: distinguish 09 vs 16 using backend detect helpers.
            if is_gaussian_16(file_name):
                return Gaussian16LogCS.read(file_name)

            if is_gaussian_09(file_name):
                return Gaussian09LogCS.read(file_name)

            raise UnsupportedFileError(
                message="Unsupported QC file for shielding "
                "reader (no known signature found)",
                path=file_name,
                kind="shield",
            )

        raise UnsupportedFileError(
            message="Unsupported QC file for shielding "
            "reader (no known signature found)",
            path=file_name,
            kind="shield",
        )

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
            except AttributeError as e:
                raise ReaderContractError(
                    message=(
                        f"Reader contract violation: '{cls.__name__}' is missing "
                        f"required attribute '{attribute}'"
                    ),
                    path=file_name,
                    kind="shield",
                    details={"class": cls.__name__, "attribute": attribute},
                ) from e

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

    COMMON_STR = ORCA_SIGNATURE

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

    COMMON_STR = GAUSSIAN_16_SIGNATURE

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

    COMMON_STR = GAUSSIAN_09_SIGNATURE

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
        if is_gaussian_log(file_name):
            return GaussianLogSpin.read(file_name)

        if is_orca_output(file_name):
            return OrcaSpin.read(file_name)

        raise UnsupportedFileError(
            message="Unsupported QC file for spin reader (no known signature found)",
            path=file_name,
            kind="spin",
        )

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
            except AttributeError as e:
                raise ReaderContractError(
                    message=(
                        f"Reader contract violation: '{cls.__name__}' is missing "
                        f"required attribute '{attribute}'"
                    ),
                    path=file_name,
                    kind="spin",
                    details={"class": cls.__name__, "attribute": attribute},
                ) from e
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
    COMMON_STR = GAUSSIAN_SIGNATURE

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
    COMMON_STR = ORCA_SIGNATURE

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
            UnsupportedFileError: If no supported reader matches the file content.
        """

        if is_gaussian_log(file_name):
            return GaussianLogA.read(file_name)

        if is_orca_output(file_name):
            # ORCA: prefer explicit PROPERTY marker when present.
            if is_orca_property(file_name):
                return Orca5PropertyA.read(file_name)

            # ORCA OUTPUT: distinguish 5 vs 6 using legacy banner markers.
            if is_orca_a6_output(file_name):
                return Orca6OutputA.read(file_name)

            if is_orca_a5_output(file_name):
                return Orca5OutputA.read(file_name)

            raise UnsupportedFileError(
                message=(
                    "Unsupported QC file for hyperfine "
                    "reader (no known signature found)"
                ),
                path=file_name,
                kind="hfc",
            )

        raise UnsupportedFileError(
            message=(
                "Unsupported QC file for hyperfine reader (no known signature found)"
            ),
            path=file_name,
            kind="hfc",
        )

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
            except AttributeError as e:
                raise ReaderContractError(
                    message=(
                        f"Reader contract violation: '{cls.__name__}' is missing "
                        f"required attribute '{attribute}'"
                    ),
                    path=file_name,
                    kind="hfc",
                    details={"class": cls.__name__, "attribute": attribute},
                ) from e

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

    COMMON_STR = GAUSSIAN_SIGNATURE

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


class Orca5OutputA(QCA):
    """
    A Tensor object for Orca 5 OUTPUT files
    """

    FILETYPE = "Orca OUTPUT"

    COMMON_STR = ORCA_A5_SIGNATURE

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

    COMMON_STR = ORCA_A6_SIGNATURE

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


class Orca5PropertyA(QCA):
    """
    A Tensor object for Orca PROPERTY files
    """

    FILETYPE = "Orca PROPERTY"

    COMMON_STR = ORCA_A5_SIGNATURE

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
