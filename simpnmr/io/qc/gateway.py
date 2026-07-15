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
from simpnmr.io.qc.backends.orca.gtensor import (  # noqa
    read_g_tensor_ab_initio,
)
from simpnmr.io.qc.backends.orca.gtensor import (
    read_g_tensor_dft as read_orca_g_tensor_dft,
)
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


def read_g_tensor_dft(
    file_name: str,
) -> (
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]
    | None
):
    """Read decomposed DFT g-tensor contributions from a supported QC file.

    This gateway is the intermediate layer between backend-specific readers and
    app-level loaders. It exposes the decomposed DFT g-tensor contribution
    tensors as returned by the backend parser and keeps them separate from any
    later builder step that assembles the full physical g-tensor.

    Args:
        file_name: Path to the QC output file.

    Returns:
        Tuple ``(g_rmc, g_dso, g_pso)`` of (3, 3) arrays when supported,
        otherwise None.

    Raises:
        UnsupportedFileError: If the file is not a supported QC output for this
            reader.
    """

    if is_orca_output(file_name):
        return read_orca_g_tensor_dft(file_name)
    else:
        return None


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

    def __init__(
        self, file_name, labels, coords, cs_iso, cs_aniso, cs_units, cs_tensor=None
    ):
        """Initialize a chemical shielding container.

        Args:
            file_name: Source file name.
            labels: Atom labels (with indices).
            coords: Atomic coordinates as an (n_atoms, 3) array.
            cs_iso: Isotropic chemical shielding values by label.
            cs_aniso: Anisotropic chemical shielding values by label.
            cs_units: Units for shielding values.
            cs_tensor: Full 3x3 shielding tensors by label, or ``None`` when the
                reader only provides isotropic/anisotropic scalars.
        """

        self.file_name = file_name
        self.labels = labels
        self.coords = coords
        self.n_atoms = len(labels)
        self.cs_iso = cs_iso
        self.cs_aniso = cs_aniso
        self.cs_units = cs_units
        self.cs_tensor = cs_tensor if cs_tensor is not None else {}

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

    "Full 3x3 Chemical Shielding tensors by label (empty if reader gives scalars)"
    cs_tensor: dict[str, npt.NDArray]

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
        cs_iso, cs_aniso, cs_tensor = read_orca5_output_cs(file_name)

        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )

        converter = {ol: nl for ol, nl in zip(old_labels, new_labels)}

        cs_iso = {converter[label]: val for label, val in cs_iso.items()}

        cs_aniso = {converter[label]: val for label, val in cs_aniso.items()}

        cs_tensor = {
            converter[label]: t
            for label, t in cs_tensor.items()
            if label in converter
        }

        cs_units = "ppm"

        return cls(
            file_name, new_labels, coords, cs_iso, cs_aniso, cs_units, cs_tensor=cs_tensor
        )


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
        cs_iso, cs_aniso, cs_tensor = read_gaussian16_log_cs(file_name)

        cs_units = "ppm"

        return cls(
            file_name, labels, coords, cs_iso, cs_aniso, cs_units, cs_tensor=cs_tensor
        )


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
    """Abstract base class for component-based hyperfine (A-tensor) readers."""

    def __init__(self, file_name, labels, coords, a_fc, a_sd, a_orb, a_units):
        """Initialize a component-based hyperfine (A-tensor) container.

        Args:
            file_name: Source file name.
            labels: Atom labels (with indices).
            coords: Atomic coordinates as an (n_atoms, 3) array.
            a_fc: Fermi-contact hyperfine tensors by label (3x3).
            a_sd: Spin-dipolar hyperfine tensors by label (3x3).
            a_orb: Orbital hyperfine tensors by label (3x3) or None when unavailable.
            a_units: Units for hyperfine values.
        """

        self.file_name = file_name
        self.labels = labels
        self.coords = coords
        self.n_atoms = len(labels)
        self.a_fc = a_fc
        self.a_sd = a_sd
        self.a_orb = a_orb
        self.a_units = a_units

        return

    @staticmethod
    def guess_from_file(file_name: str, *, spin: float | None = None) -> "QCA":
        """Guess a compatible hyperfine reader and parse the file.

        Args:
            file_name: Path to the file to examine.
            spin: Optional spin quantum number S. Used only for Gaussian logs
                that lack a 'Multiplicity =' line, to derive n_unpaired for
                A(SD) normalisation. If the file is missing the line and spin
                is not provided, a ValueError is raised with a hint to add
                'spin:' under 'hyperfine:' in the YAML.

        Returns:
            QCA: Parsed component-based hyperfine object.

        Raises:
            UnsupportedFileError: If no supported reader matches the file content.
        """

        if is_gaussian_log(file_name):
            if spin is not None:
                return GaussianLogA.read_with_spin(file_name, spin)
            return GaussianLogA.read(file_name)

        if is_orca_output(file_name):
            # ORCA: prefer explicit PROPERTY marker when present.
            if is_orca_property(file_name):
                return Orca5PropertyA.read(file_name)

            # ORCA OUTPUT
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

        string += subtitle("A(FC) Tensor ({})".format(self.a_units))
        for label, tensor in self.a_fc.items():
            string += "\n      {: .6f} {: .6f} {: .6f}\n".format(*tensor[0])
            string += "{:5} {: .6f} {: .6f} {: .6f}\n".format(label, *tensor[1])
            string += "      {: .6f} {: .6f} {: .6f}\n".format(*tensor[2])

        string += subtitle("A(SD) Tensor ({})".format(self.a_units))
        for label, tensor in self.a_sd.items():
            string += "\n      {: .6f} {: .6f} {: .6f}\n".format(*tensor[0])
            string += "{:5} {: .6f} {: .6f} {: .6f}\n".format(label, *tensor[1])
            string += "      {: .6f} {: .6f} {: .6f}\n".format(*tensor[2])

        string += subtitle("A(ORB) Tensor ({})".format(self.a_units))
        for label, tensor in self.a_orb.items():
            if tensor is None:
                string += "{:5} None\n".format(label)
                continue
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

    "Fermi-contact hyperfine tensors (3x3) by atom label"
    a_fc: dict[str, npt.NDArray]

    "Spin-dipolar hyperfine tensors (3x3) by atom label"
    a_sd: dict[str, npt.NDArray]

    "Orbital hyperfine tensors (3x3) by atom label, or None when unavailable"
    a_orb: dict[str, npt.NDArray | None]

    "Units of component hyperfine tensors"
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
            "a_fc",
            "a_sd",
            "a_orb",
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
            QCA: Parsed component-based hyperfine instance.
        """
        raise NotImplementedError


class GaussianLogA(QCA):
    """
    A QCA object for Gaussian LOG files
    """

    FILETYPE = "Gaussian LOG"

    COMMON_STR = GAUSSIAN_SIGNATURE

    @classmethod
    def _read(cls, file_name: str, n_unpaired: int | None = None):
        # Read raw data
        labels, coords = read_gaussian_log_xyz(file_name)
        labels = np.array(xyzf.add_label_indices(labels))
        a_fc_raw, a_sd_raw = read_gaussian_log_a_tensors(file_name)

        mult = read_gaussian_log_spin(file_name)
        if mult is None:
            if n_unpaired is None:
                raise ValueError(
                    f"Could not find 'Multiplicity =' in Gaussian log: {file_name}. "
                    "Add 'spin: <value>' under 'hyperfine:' in your YAML to specify "
                    "the spin multiplicity manually."
                )
            logger.warning(
                "Multiplicity not found in %s; using n_unpaired=%d from config spin.",
                file_name,
                n_unpaired,
            )
        else:
            n_unpaired = mult - 1

        # Gaussian provides isotropic Fermi-contact values and traceless dipolar
        # tensors separately. Adapt these raw quantities to the canonical QCA
        # component contract expected downstream: A(FC) as an isotropic 3x3 tensor
        # and A(SD) as the traceless spin-dipolar tensor.
        a_fc = {
            label: np.eye(3, dtype=float) * float(fc_iso)
            for label, fc_iso in zip(labels, a_fc_raw)
        }
        a_sd = {
            label: tensor * 1.0 / n_unpaired for label, tensor in zip(labels, a_sd_raw)
        }
        a_orb = {label: None for label in labels}

        a_units = "MHz"

        return cls(file_name, labels, coords, a_fc, a_sd, a_orb, a_units)

    @classmethod
    def read_with_spin(cls, file_name: str, spin: float) -> "GaussianLogA":
        """Read a Gaussian log, using *spin* (S) to derive n_unpaired when
        'Multiplicity =' is absent from the file."""
        return cls._read(file_name, n_unpaired=round(2 * spin))


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
        a_fc, a_sd, a_orb = read_orca5_output_a_tensors(file_name)

        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        a_fc = {converter[label]: value for label, value in a_fc.items()}
        a_sd = {converter[label]: value for label, value in a_sd.items()}
        a_orb = {converter[label]: value for label, value in a_orb.items()}
        a_units = "MHz"

        return cls(file_name, new_labels, coords, a_fc, a_sd, a_orb, a_units)


class Orca6OutputA(QCA):
    """
    A Tensor object for Orca 6 OUTPUT files
    """

    FILETYPE = "Orca OUTPUT"

    COMMON_STR = ORCA_A6_SIGNATURE

    @classmethod
    def _read(cls, file_name: str):
        return cls.read_with_options(file_name)

    @classmethod
    def read_with_options(cls, file_name: str) -> "Orca6OutputA":
        """Read ORCA6 component hyperfine tensors (A(FC), A(SD), A(ORB)).

        Args:
            file_name: Path to the ORCA6 output file.

        Returns:
            Orca6OutputA: Parsed ORCA6 component hyperfine tensor container.
        """

        # Read raw data
        old_labels, coords = read_orca5_output_xyz(file_name)
        old_labels = np.array(
            xyzf.add_label_indices(old_labels, style="sequential", start_index=0)
        )
        a_fc, a_sd, a_orb = read_orca6_output_a_tensors(file_name)

        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        a_fc = {converter[label]: value for label, value in a_fc.items()}
        a_sd = {converter[label]: value for label, value in a_sd.items()}
        a_orb = {converter[label]: value for label, value in a_orb.items()}
        a_units = "MHz"

        return cls(file_name, new_labels, coords, a_fc, a_sd, a_orb, a_units)


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
        a_fc, a_sd, a_orb = read_orca5_property_a_tensors(file_name)

        # Convert orca labelling 1-> natoms for all atoms
        # to 1-n_atoms per element
        new_labels = np.array(
            xyzf.add_label_indices(xyzf.remove_label_indices(old_labels))
        )
        converter = {old: new for old, new in zip(old_labels, new_labels)}

        a_fc = {converter[label]: value for label, value in a_fc.items()}
        a_sd = {converter[label]: value for label, value in a_sd.items()}
        a_orb = {converter[label]: value for label, value in a_orb.items()}
        a_units = "MHz"

        return cls(file_name, new_labels, coords, a_fc, a_sd, a_orb, a_units)
