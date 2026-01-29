# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO"""

import copy
import datetime
import logging
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

from simpnmr.__version__ import __version__
from simpnmr.core.constants import isotopes, periodic_table
from simpnmr.core.convertors import hyperfine as hfc
from simpnmr.core.domain.tensors import Hyperfine, Shift, Susceptibility
from simpnmr.core.utils.arrays import flatten
from simpnmr.io.csv.utils import read_csv_safe
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.mappers import dataframes as ser
from simpnmr.mappers import label_format as lf
from simpnmr.tools.coords_tools import xyz_format as xyzf

logger = logging.getLogger(__name__)


# RETURN TO: Create a Relaxation class and use it in Nucleus
# This will allow us to use something like nuc.r1 and nuc.r2 to access relaxation rates
class Relaxation:
    """Holds calculated relaxation rates for a nucleus.

    Attributes:
        r1: Longitudinal relaxation rate (s^-1).
        r2: Transverse relaxation rate (s^-1).
    """

    def __init__(
        self,
        r1=None,
        r2=None,
        dipolar_r1=None,
        contact_r1=None,
        curie_r1=None,
        dipolar_r2=None,
        contact_r2=None,
        curie_r2=None,
    ):
        self.r1 = r1
        self.r2 = r2
        self.dipolar_r1 = dipolar_r1
        self.contact_r1 = contact_r1
        self.curie_r1 = curie_r1
        self.dipolar_r2 = dipolar_r2
        self.contact_r2 = contact_r2
        self.curie_r2 = curie_r2
        pass


# Add setters and properties as needed


class Nucleus:
    r"""Container for nucleus-specific data.

    Args:
        label: Atomic label with index (e.g., ``"H2"``).
        coord: Cartesian coordinates of the nucleus.
        A: Hyperfine coupling as a `Hyperfine` instance.
        shift: Chemical shift container. Defaults to a zeroed `Shift`.
        chem_label: Optional chemical label (e.g., ``"tBu3"``).
        chem_math_label: Optional mathtext (LaTeX-like) label used in plots,
            e.g. ``$\\mathregular{tBu_3}$``.
        isotope: Isotope label formatted as nucleon number then symbol
            (e.g., ``"13C"``).

    Attributes:
        label: Atomic label with index (e.g., ``"H2"``).
        label_nn: Atomic label without the index (e.g., ``"H"``).
        chem_label: Chemical label if provided, otherwise falls back to `label`.
        chem_math_label: Mathtext label if provided, otherwise falls back to
            `chem_label`.
        coord: Coordinates as a length-3 NumPy array.
        A: Hyperfine coupling container.
        shift: Chemical shift container.
        isotope: Isotope label (e.g., ``"13C"``).
    """

    def __init__(
        self,
        label: str,
        coord: list[float],
        A: Hyperfine,
        shift: Shift = Shift(),
        chem_label: str = None,
        chem_math_label: str = None,
        isotope: str = None,
    ) -> None:
        # Label with and without indexing
        self.label = label
        self.label_nn = xyzf.remove_label_indices(self.label)

        # Hyperfine coupling tensor for current nucleus
        self.A = copy.deepcopy(A)

        # Chemical shift
        self.shift = copy.deepcopy(shift)

        # Coordinates of nucleus
        self.coord = coord

        # Chemical labels, normal and mathtext
        if chem_label is None:
            self._chem_label = None
        else:
            self.chem_label = chem_label
        if chem_math_label is None:
            self._chem_math_label = None
        else:
            self.chem_math_label = chem_math_label

        # If isotope is provided then set, else set as default
        if isotope is None:
            self.isotope = isotopes.DEFAULT_ISOTOPES[self.label_nn]

        return

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, inlabel: str):
        if not isinstance(inlabel, str):
            raise TypeError("label must be string")
        self._label = str(inlabel)
        return

    @property
    def chem_label(self) -> str:
        if self._chem_label is None:
            return self.label
        return self._chem_label

    @chem_label.setter
    def chem_label(self, inchem_label: str):
        if not isinstance(inchem_label, (type(None), str)):
            raise TypeError("chem_label must be string")
        self._chem_label = inchem_label
        return

    @property
    def chem_math_label(self) -> str:
        if self._chem_math_label is None:
            return self.chem_label
        return self._chem_math_label

    @chem_math_label.setter
    def chem_math_label(self, inchem_math_label: str):
        if not isinstance(inchem_math_label, (type(None), str)):
            raise TypeError("chem_math_label must be string")
        self._chem_math_label = inchem_math_label
        return

    @property
    def coord(self) -> NDArray:
        return self._coord

    @coord.setter
    def coord(self, incoord: ArrayLike):
        incoord = np.asarray(incoord)

        if len(incoord.shape) > 1:
            raise ValueError("Nucleus coordinates must be (1x3) array")

        elif incoord.shape[0] != 3:
            raise ValueError("Nucleus coordinates must be (1x3) array")
        self._coord = incoord
        return

    @property
    def A(self) -> Hyperfine:
        return self._A

    @A.setter
    def A(self, inA: Hyperfine):
        if not isinstance(inA, Hyperfine):
            raise TypeError("A must be Hyperfine object")
        self._A = inA
        return

    @property
    def shift(self) -> Shift:
        return self._shift

    @shift.setter
    def shift(self, inShift: Shift):
        if not isinstance(inShift, Shift):
            raise TypeError("shift must be a Shift object")
        self._shift = inShift

    @property
    def isotope(self) -> str:
        return self._isotope

    @isotope.setter
    def isotope(self, value: str):
        if re.sub("[0-9]", "", value) != self.label_nn:
            raise ValueError("Isotope label does not match atomic label")
        elif value not in isotopes.SUPPORTED_ISOTOPES:
            raise ValueError(f"Unsupported isotope {value}")
        else:
            self._isotope = value

    @classmethod
    def from_a_values(
        cls, a_isos: dict[str, float], a_dips: dict[str, NDArray], coords: NDArray
    ) -> list["Nucleus"]:
        """Build nuclei from isotropic and dipolar hyperfine data.

        Args:
            a_isos: Mapping from atom label to isotropic hyperfine coupling
                (ppm Å^-3).
            a_dips: Mapping from atom label to dipolar hyperfine tensor as a
                3x3 array (ppm Å^-3).
            coords: Coordinates for each nucleus. The ordering must match the
                ordering of the dictionaries.

        Returns:
            A list of `Nucleus` instances.
        """

        tensors = {
            label: Hyperfine(a_dips[label] + np.eye(3) * a_isos[label])
            for label in a_dips
        }

        nuclei = [
            cls(key, coord, value)
            for (key, value), coord in zip(tensors.items(), coords)
        ]

        if not len(nuclei):
            raise ValueError("No Nuclei selected!")

        return nuclei


class ElectronicState:
    """Electronic/magnetic state of the system.

    Stores global quantum numbers and magnetic-model metadata.
    """

    def __init__(
        self,
        spin_S: float | None = None,
        orbit_L: float | None = None,
        total_J: float | None = None,
        model: str | None = None,
    ) -> None:
        self.spin_S = spin_S
        self.orbit_L = orbit_L
        self.total_J = total_J
        self.model = model

        if self.model is not None and self.model not in {
            "spin_only",
            "orbital",
            "total_J",
        }:
            raise ValueError(
                "ElectronicState.model must be one of 'spin_only', "
                "'orbital', or 'total_J'"
            )

        return


class Molecule:
    """Molecular container holding structure and NMR-active nuclei.

    Args:
        labels: Atomic labels (no indices).
        coords: Atomic coordinates as an ``(n_atoms, 3)`` array in Å.
        nuclei: List of NMR-active `Nucleus` objects.

    Attributes:
        labels: Atomic labels with indices.
        coords: Atomic coordinates as an ``(n_atoms, 3)`` array in Å.
        n_atoms: Number of atoms.
        nuclei: NMR-active nuclei.
        susc: Magnetic susceptibility tensor for the molecule.
        electronic: Electronic state metadata (spin/orbit/J model selection).
    """

    def __init__(
        self, labels: NDArray[np.str_], coords: NDArray, nuclei: list[Nucleus]
    ) -> None:
        self.labels = xyzf.add_label_indices(labels)
        self.coords = coords

        # List of Nucleus objects
        self.nuclei = nuclei

        # Susceptibility object
        self.susc = copy.deepcopy(Susceptibility())

        # List of quantum number objects
        self.electronic = ElectronicState()

    pass

    @property
    def n_atoms(self):
        return len(self.labels)

    def __str__(self):
        string = ""

        string += lf.title("Molecule Hyperfine Data")

        string += lf.subtitle("Isotropic A values (ppm Å^-3)")

        for nuc in self.nuclei:
            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f"{nuc.chem_label} ({nuc.label})"

            string += f"{label} {nuc.A.iso: .6f}\n"

        string += lf.subtitle("Anisotropic (dipolar) A Tensor (ppm Å^-3)")

        for nuc in self.nuclei:
            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f"{nuc.chem_label} ({nuc.label})"

            string += "\n{:} {: .6f} {: .6f} {: .6f}\n".format(
                " " * len(label), *nuc.A.dip[0]
            )
            string += "{:} {: .6f} {: .6f} {: .6f}\n".format(label, *nuc.A.dip[1])
            string += "{:} {: .6f} {: .6f} {: .6f}\n".format(
                " " * len(label), *nuc.A.dip[2]
            )

        return string

    @classmethod
    def from_xyz(cls, xyz_file: str, elements: list[str] | str = "all") -> "Molecule":
        """Create a `Molecule` from an XYZ file.

        Args:
            xyz_file: Path to an XYZ file containing the full structure.
            elements: Elements/labels to include. Use ``"all"`` for all atoms.

        Returns:
            A `Molecule` instance.
        """

        # Load xyz file
        labels, coords = xyzf.load_xyz(xyz_file)

        # Generate Molecule using ALL labels and coords
        base = cls.from_labels_coords(labels, coords, elements)

        return base

    @classmethod
    def from_labels_coords(
        cls,
        labels: ArrayLike,
        coords: ArrayLike,
        elements: list[str] | str = "all",
    ) -> "Molecule":
        """Create a `Molecule` from labels and coordinates.

        Args:
            labels: Atomic labels.
            coords: Atomic coordinates as an ``(n_atoms, 3)`` array-like in Å.
            elements: Elements/labels to include. Use ``"all"`` to include all.

        Returns:
            A `Molecule` instance.
        """

        # Normalize inputs to stable Python types.
        labels_list: list[str] = [str(lab) for lab in list(np.asarray(labels))]

        if isinstance(elements, str):
            elements = [elements]

        coords = np.asarray(coords)

        elements_to_include = []
        for ele in elements:
            if ele == "all":
                elements_to_include = labels_list
                break
            elif "all_" in ele or ele in periodic_table.elements:
                if "all_" in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [la for la in labels_list if _e == xyzf.remove_label_indices(la)]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        # Generate list of Nuclei, one for each atom
        # selecting only those elements requested by user
        nuclei = [
            Nucleus(label, coord, Hyperfine())
            for label, coord in zip(labels_list, coords)
            if label in elements_to_include
        ]

        # Generate Molecule using ALL labels and coords
        base = cls(labels_list, coords, nuclei)

        return base

    @classmethod
    def from_csv(cls, file_name: str, elements: list[str] | str = "all") -> "Molecule":
        """Create a `Molecule` from a CSV file containing structure and tensors.

        Args:
            file_name: Path to the CSV file.
            elements: Elements/labels to include. Use ``"all"`` to include all.

        Returns:
            A `Molecule` instance.

        Raises:
            ValueError: If required columns are missing or hyperfine headers are
                incomplete.
        """

        data = read_csv_safe(file_name)

        required_cols = ["atom_label ()", "x (Å)", "y (Å)", "z (Å)"]
        split_hyperfine_cols = [
            "Aiso (ppm Å^-3)",
            "Adip_xx (ppm Å^-3)",
            "Adip_xy (ppm Å^-3)",
            "Adip_xz (ppm Å^-3)",
            "Adip_yy (ppm Å^-3)",
            "Adip_yz (ppm Å^-3)",
            "Adip_zz (ppm Å^-3)",
        ]
        full_hyperfine_cols = [
            "A_xx (ppm Å^-3)",
            "A_xy (ppm Å^-3)",
            "A_xz (ppm Å^-3)",
            "A_yy (ppm Å^-3)",
            "A_yz (ppm Å^-3)",
            "A_zz (ppm Å^-3)",
        ]

        # Standardise column names
        name_convertor = {
            "atom_labels": "atom_label ()",
            "atom_labels ()": "atom_label ()",
            "chem_label": "chem_label ()",
            "chem_labels ()": "chem_label ()",
            "chem_math_label": "chem_math_label ()",
            "chem_math_labels ()": "chem_math_label ()",
            "x": "x (Å)",
            "x (A)": "x (Å)",
            "y": "y (Å)",
            "y (A)": "y (Å)",
            "z": "z (Å)",
            "z (A)": "z (Å)",
        }
        others = {}
        for key, val in name_convertor.items():
            others[key.capitalize()] = val
            others[val.capitalize()] = val
        name_convertor.update(others)
        data.rename(columns=name_convertor, inplace=True)

        missing = [col for col in required_cols if col not in data.columns]
        if any(missing):
            raise ValueError(f"Missing header(s) {missing} in {file_name}")

        if all([col in data.columns for col in split_hyperfine_cols]):
            split = True
        elif all([col in data.columns for col in full_hyperfine_cols]):
            split = False
        else:
            raise ValueError(f"Incomplete hyperfine headers in {file_name}")

        labels = data["atom_label ()"]

        elements_to_include = []
        for ele in elements:
            if ele == "all":
                elements_to_include = labels
                break
            elif "all_" in ele or ele in periodic_table.elements:
                if "all_" in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [la for la in labels if _e == xyzf.remove_label_indices(la)]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        # Generate list of Nuclei, one for each atom
        # selecting only those elements requested by user
        coords = np.array([data["x (Å)"], data["y (Å)"], data["z (Å)"]])

        if split:
            tensors = [
                np.array(
                    [
                        [
                            row["Adip_xx (ppm Å^-3)"],
                            row["Adip_xy (ppm Å^-3)"],
                            row["Adip_xz (ppm Å^-3)"],
                        ],
                        [
                            row["Adip_xy (ppm Å^-3)"],
                            row["Adip_yy (ppm Å^-3)"],
                            row["Adip_yz (ppm Å^-3)"],
                        ],
                        [
                            row["Adip_xz (ppm Å^-3)"],
                            row["Adip_yz (ppm Å^-3)"],
                            row["Adip_zz (ppm Å^-3)"],
                        ],
                    ]
                )
                + np.eye(3) * row["Aiso (ppm Å^-3)"]
                for _, row in data.iterrows()
            ]
        else:
            tensors = [
                np.array(
                    [
                        [
                            row["A_xx (ppm Å^-3)"],
                            row["A_xy (ppm Å^-3)"],
                            row["A_xz (ppm Å^-3)"],
                        ],
                        [
                            row["A_xy (ppm Å^-3)"],
                            row["A_yy (ppm Å^-3)"],
                            row["A_yz (ppm Å^-3)"],
                        ],
                        [
                            row["A_xz (ppm Å^-3)"],
                            row["A_yz (ppm Å^-3)"],
                            row["A_zz (ppm Å^-3)"],
                        ],
                    ]
                )
                for _, row in data.iterrows()
            ]

        coords = coords.T

        nuclei = [
            Nucleus(label, coord, Hyperfine(tensor))
            for label, coord, tensor in zip(labels, coords, tensors)
            if label in elements_to_include
        ]

        # Add chem labels if present
        if "chem_label ()" in data.columns:
            for nucleus, (_, row) in zip(nuclei, data.iterrows()):
                nucleus.chem_label = row["chem_label ()"]
        if "chem_math_label ()" in data.columns:
            for nucleus, (_, row) in zip(nuclei, data.iterrows()):
                nucleus.chem_math_label = row["chem_math_label ()"]

        # Generate Molecule using ALL labels and coords
        base = cls(labels, coords, nuclei)

        return base

    @classmethod
    def from_QCA(
        cls,
        ab_initio: rdrs.QCA,
        converter: str = "Null",
        elements: list[str] | str = "all",
    ) -> "Molecule":
        """Create a `Molecule` from ab initio hyperfine data.

        Args:
            ab_initio: Parsed hyperfine container from `simpnmr.readers`.
            converter: Unit converter identifier. Use ``"Null"`` to apply no
                conversion.
            elements: Elements/labels to include. Use ``"all"`` to include all.

        Returns:
            A `Molecule` instance.
        """

        if isinstance(elements, str):
            elements = [elements]

        elements_to_include = []
        for ele in elements:
            if ele == "all":
                elements_to_include = copy.copy(ab_initio.labels)
                break
            elif "all_" in ele or ele in periodic_table.elements:
                if "all_" in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [
                    la for la in ab_initio.labels if _e == xyzf.remove_label_indices(la)
                ]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        # Convert units
        if converter.lower() != "null":
            a_isos = hfc.a_tensor_mhz_to_angstrom(ab_initio.a_iso)
            a_dips = hfc.a_tensor_mhz_to_angstrom(ab_initio.a_dip)
        else:
            a_isos = ab_initio.a_iso
            a_dips = ab_initio.a_dip

        a_isos = {key: val for key, val in a_isos.items() if key in elements_to_include}
        a_dips = {key: val for key, val in a_dips.items() if key in elements_to_include}

        coords = [
            coord
            for label, coord in zip(ab_initio.labels, ab_initio.coords)
            if label in elements_to_include
        ]

        # Generate list of Nuclei, one for each atom
        nuclei = Nucleus.from_a_values(a_isos, a_dips, coords)

        # Create molecule - uses all atoms, regardless of user labels
        base = cls(ab_initio.labels, ab_initio.coords, nuclei)

        return base

    @property
    def susc(self) -> Susceptibility:
        return self._susc

    @susc.setter
    def susc(self, new_susc: Susceptibility):
        if not isinstance(new_susc, Susceptibility):
            raise TypeError("Molecule.susc must be of type Susceptibility")
        self._susc = new_susc
        return

    def load_diamagnetic_shifts(
        self,
        file_name: str,
        file_type: str = "csv",
        ref_file_name: str = "",
        ref_file_type: str = "csv",
    ) -> None:
        """Load diamagnetic shifts from a file and assign them to nuclei.

        Args:
            file_name: Input file containing diamagnetic shifts.
            file_type: Input type. Supported values are ``"csv"`` and ``"dft"``.
            ref_file_name: Optional reference file used to subtract reference
                shifts.
            ref_file_type: Reference file type. Supported values are ``"csv"`` and
                ``"dft"``.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If `file_type` or `ref_file_type` is unsupported.
        """

        if file_type == "csv":
            dia = read_csv_safe(file_name)
            if "atom_label" in dia.keys():
                dia.set_index("atom_label", inplace=True)
                for nuc in self.nuclei:
                    nuc.shift.dia = dia["shift"][nuc.label]
            elif "chem_label" in dia.keys():
                dia.set_index("chem_label", inplace=True)
                for nuc in self.nuclei:
                    nuc.shift.dia = dia["shift"][nuc.chem_label]
            else:
                raise KeyError(
                    "atom_label or chem_label not present in diamagnetic shift file"
                )
        elif file_type == "dft":
            data = rdrs.QCCS.guess_from_file(file_name)

            _relabel = {
                new: old
                for old, new in zip(
                    data.cs_iso.keys(), xyzf.add_label_indices(data.cs_iso.keys())
                )
            }

            for nuc in self.nuclei:
                try:
                    nuc.shift.dia = data.cs_iso[_relabel[nuc.label]]
                except KeyError:
                    raise KeyError(
                        f"Cannot find {nuc.label} in reference diamagnetic shift file"
                    )
        else:
            raise ValueError("Unknown file_type")

        if len(ref_file_name):
            if ref_file_type == "csv":
                ref = read_csv_safe(ref_file_name)

                # Average by nucleus
                ref["atom_label"] = xyzf.remove_label_indices(ref["atom_label"])

                ref = ref.groupby("atom_label").mean().reset_index()

                for nuc in self.nuclei:
                    try:
                        nuc.shift.dia = ref["shift"][nuc.label_nn] - nuc.shift.dia
                    except KeyError:
                        raise KeyError(
                            f"Cannot find {nuc.label_nn} in reference diamagnetic "
                            "shift file"
                        )

            elif ref_file_type == "dft":
                ref_data = rdrs.QCCS.guess_from_file(ref_file_name)

                ref_labels = list(ref_data.cs_iso.keys())
                ref_labels_nn = xyzf.remove_label_indices(ref_labels)

                avg_ref_iso = dict.fromkeys(ref_labels_nn, 0)

                for lab, lab_nn in zip(ref_labels, ref_labels_nn):
                    avg_ref_iso[lab_nn] += ref_data.cs_iso[lab]

                for lab_nn in np.unique(ref_labels_nn):
                    avg_ref_iso[lab_nn] /= ref_labels_nn.count(lab_nn)

                for nuc in self.nuclei:
                    try:
                        nuc.shift.dia = avg_ref_iso[nuc.label_nn] - nuc.shift.dia
                    except KeyError:
                        raise KeyError(
                            f"Cannot find {nuc.label_nn} in reference diamagnetic "
                            "shift file"
                        )
            else:
                raise ValueError("Unknown file_type")
        return

    def average_shifts(self):
        """Average total shifts over nuclei sharing the same chemical label.

        The mean value is stored in `Nucleus.shift.avg`.
        """

        cl_to_shifts = {nuc.chem_label for nuc in self.nuclei}
        cl_to_shifts = {cl: [] for cl in cl_to_shifts}
        for nuc in self.nuclei:
            cl_to_shifts[nuc.chem_label].append(nuc.shift.total)

        cl_to_shifts = {cl: np.mean(shifts) for cl, shifts in cl_to_shifts.items()}
        for nuc in self.nuclei:
            nuc.shift.avg = cl_to_shifts[nuc.chem_label]

        return

    def average_hyperfine(self, av_chemlabels: list[str] | list[list[str]]):
        """Average hyperfine tensors for specified nuclei.

        Args:
            av_chemlabels: Chemical labels specifying which nuclei are averaged.
                If a flat list is provided, each entry is averaged separately.
                If a list of lists is provided, each sublist defines a group of
                labels that are averaged together.

        Raises:
            TypeError: If `av_chemlabels` contains unsupported types.
            ValueError: If any requested label is not present in the molecule.
        """

        # Convert all entries into lists
        av_chemlabels = [
            [ent] if not isinstance(ent, list) else ent for ent in av_chemlabels
        ]

        # Check formatting - either list of lists or just list
        # list of lists - sublists group dissimilar labels which will be
        # averaged together
        # list - entries are averaged separately
        if not all(isinstance(ent, (list, str)) for ent in av_chemlabels):
            raise TypeError(
                "Unknown type passed to average_hyperfine, "
                "labels should be list[list[str]] or list[str]"
            )

        # Check sublists are all string
        if any(
            [not isinstance(subent, str) for ent in av_chemlabels for subent in ent]
        ):
            raise TypeError(
                "Unknown type passed to average_hyperfine, "
                "labels should be list[list[str]] or list[str]"
            )

        # Check labels exist in molecule
        _fl_av_chemlabels = flatten(av_chemlabels)
        all_chemlabels = [nuc.chem_label for nuc in self.nuclei]
        if any([cl not in all_chemlabels for cl in _fl_av_chemlabels]):
            print(set(all_chemlabels).difference(set(_fl_av_chemlabels)))
            raise ValueError("Attempted average using unknown chem_label")

        # Average hyperfines and diamagnetic shifts
        for ents in av_chemlabels:
            avg_atens = np.mean(
                [nuc.A.tensor for nuc in self.nuclei if nuc.chem_label in ents],
                axis=0,
            )
            for nuc in self.nuclei:
                if nuc.chem_label in ents:
                    nuc.A.tensor = avg_atens

        return

    def rotate_hyperfines(self, rot_mat: ArrayLike):
        """Rotate all hyperfine tensors using a rotation matrix.

        This applies the standard second-rank tensor rotation:

            ``A' = R @ A @ R.T``

        where `R` maps components from the old frame into the new frame.

        Args:
            rot_mat: Rotation matrix ``R`` with shape ``(3, 3)``.

        Raises:
            ValueError: If `rot_mat` is not a ``(3, 3)`` matrix.
        """

        rot_mat = np.asarray(rot_mat)
        if rot_mat.shape != (3, 3):
            raise ValueError("rot_mat must be a (3x3) rotation matrix")

        for nuc in self.nuclei:
            nuc.A.tensor = rot_mat @ nuc.A.tensor @ rot_mat.T

        return

    def calc_pdip(self, centre_labels: list[str]):
        """Add point-dipole dipolar hyperfine contributions for all nuclei.

        Args:
            centre_labels: Labels of paramagnetic centers.

        Raises:
            ValueError: If `centre_labels` is empty, if multiple matches are
                found for a center label, or if a center label is not found.
        """

        if not len(centre_labels):
            raise ValueError(
                "Error: No paramagnetic centres specified for point dipole"
            )

        # Find user specified centre(s)
        for centre in centre_labels:
            it = [i for i, x in enumerate(self.labels) if x == centre]

            if len(it) > 1:
                raise ValueError("Error: More than one of specified label found")
            elif not len(it):
                raise ValueError(f"Cant find {centre} in labels")

            for nuc in self.nuclei:
                if nuc.label in centre_labels:
                    continue
                val = Hyperfine.calc_pdip(nuc.coord, self.coords[it[0]])
                val *= 1e6 / len(centre_labels)
                nuc.A.tensor += val
        return

    def calculate_shifts(self, shift_terms="full"):
        """Compute paramagnetic chemical shift components for all nuclei.

        Args:
            shift_terms: Shift terms to calculate. Supported values are
                ``"full"``, ``"pc"``, and ``"fc"``. ``"full"`` expands to
                ``["pc", "fc"]``.

        Raises:
            ValueError: If an unsupported shift term is provided.
        """

        if isinstance(shift_terms, str):
            shift_terms = [shift_terms]

        # Swap full for actual terms
        shift_terms = [
            nst for st in shift_terms for nst in (st if st != "full" else ["pc", "fc"])
        ]

        if "pc" in shift_terms:
            for nuc in self.nuclei:
                nuc.shift.pc = Shift.calc_pcs(nuc.A, self.susc)
        if "fc" in shift_terms:
            for nuc in self.nuclei:
                nuc.shift.fc = Shift.calc_fcs(nuc.A, self.susc)

        if "fc" not in shift_terms and "pc" not in shift_terms:
            raise ValueError("Unknown shift specified")

        return

    def add_chem_labels_from_file(self, file_name: str) -> None:
        """Assign chemical labels to nuclei using a CSV file.

        The CSV must include columns ``atom_label`` and ``chem_label``. If a
        ``chem_math_label`` column is present, it is also loaded.

        Args:
            file_name: Path to the CSV file.

        Raises:
            KeyError: If duplicate atom labels exist or required label entries
                are missing.
            ValueError: If coordinates are provided and do not match the current
                structure.
        """

        _tmp = read_csv_safe(file_name)

        # Check for duplicate atom labels
        if any([val > 1 for val in _tmp["atom_label"].value_counts()]):
            _dupes = _tmp["atom_label"].value_counts().gt(1)
            raise KeyError(f"Duplicate Atom label(s) {_dupes} in chemlabels file")

        # Check for missing/empty entries in chem_label
        if any(_tmp["chem_label"].isnull()):
            raise KeyError(
                "Missing chem_label for {}".format(
                    _tmp[_tmp["chem_label"].isnull()]["atom_label"][0]
                )
            )

        # Check for missing/empty entries in chem_math_label
        if "chem_math_label" in _tmp.keys():
            if any(_tmp["chem_math_label"].isnull()):
                raise KeyError(
                    "Missing chem_math_label for {}".format(
                        _tmp[_tmp["chem_math_label"].isnull()]["atom_label"][0]
                    )
                )

        al_to_cl = {
            al: cl
            for al, cl in zip(_tmp["atom_label"], _tmp["chem_label"])
            if al in [nuc.label for nuc in self.nuclei]
        }

        # Add chem label to each atom
        for nuc in self.nuclei:
            if nuc.label in al_to_cl.keys():
                nuc.chem_label = al_to_cl[nuc.label]

        # Add math label to each atom
        # from supplied math labels
        if "chem_math_label" in _tmp.keys():
            al_to_cml = {
                al: cl for al, cl in zip(_tmp["atom_label"], _tmp["chem_math_label"])
            }
            for nuc in self.nuclei:
                if nuc.label in al_to_cl.keys():
                    nuc.chem_math_label = al_to_cml[nuc.label].lstrip().rstrip()
        # or if math labels are not provided, set to the same as math labels
        else:
            for nuc in self.nuclei:
                if not len(nuc.chem_math_label):
                    nuc.chem_math_label = copy.deepcopy(nuc.chem_label)

        # If coordinates are provided in chem_labels file, then check these
        # against the current molecular coordinates
        if all(clab in _tmp.keys() for clab in ["x", "y", "z"]):
            _tmp.set_index("atom_label")
            for nuc in self.nuclei:
                _coord = [
                    _tmp.loc[nuc.label]["x"],
                    _tmp.loc[nuc.label]["y"],
                    _tmp.loc[nuc.label]["z"],
                ]
                diff = np.sum(_coord - nuc.coord)
                if diff > 1e-8:
                    raise ValueError(
                        f"Coordinates of {nuc.label} in chem_labels file "
                        "do not match those of molecule."
                    )

        return

    def save_hyperfines_to_csv(
        self,
        file_name: str = "dft_hyperfines.csv",
        verbose: bool = True,
        comment: str = "",
        delimiter: str = ",",
    ) -> None:
        """Save hyperfine data for all nuclei to a CSV file.

        Args:
            file_name: Output CSV file name.
            verbose: If True, prints the output file path.
            comment: Optional additional comment line (including comment marker).
            delimiter: CSV delimiter.
        """

        df = ser.build_hyperfines_df(self)

        _comment = (
            f"#This file was generated with SimpNMR v{__version__} at {{}}\n".format(
                datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
            )
        )

        _comment += comment + "\n"

        with open(file_name, "w") as _f:
            _f.write(_comment)

            df.to_csv(_f, sep=delimiter, header=True, float_format="%.5f", index=None)

        if verbose:
            logger.info("Molecule data written to %s", file_name)
            logger.info("Converted hyperfine data written to %s", file_name)

        return

    def to_csv(
        self,
        file_name: str = "molecule.csv",
        verbose: bool = True,
        comment: str = "",
        delimiter: str = ",",
    ) -> None:
        """Save molecule structure, hyperfine data, and shifts to a CSV file.

        Args:
            file_name: Output CSV file name.
            verbose: If True, prints the output file path.
            comment: Optional additional comment line (including comment marker).
            delimiter: CSV delimiter.
        """

        df = ser.build_molecule_df(self)

        _comment = (
            f"# This file was generated with SimpNMR v{__version__} at {{}}\n".format(
                datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
            )
        )

        _comment += comment + "\n"

        with open(file_name, "w") as _f:
            _f.write(_comment)

            df.to_csv(_f, sep=delimiter, header=True, float_format="%.5f", index=None)

        if verbose:
            logger.info("Molecule data written to %s", file_name)

        return

    def save_chemcraft_xyz(self, file_name: str, verbose: bool = True):
        """Save an XYZ file with Chemcraft-compatible chemical labels.

        Chemcraft can display per-atom labels if an extra quoted string is appended to
        each coordinate line. This writer appends `Nucleus.chem_label` for nuclei where
        it is available.

        Args:
            file_name: Output XYZ file path.
            verbose: If True, prints the output file path.

        Returns:
            None.
        """

        _clabs = {nuc.label: nuc.chem_label for nuc in self.nuclei}
        with open(file_name, "w") as f:
            for lab, trio in zip(self.labels, self.coords):
                f.write(
                    "{:5} {:15.7f} {:15.7f} {:15.7f}".format(
                        xyzf.lab_to_num(lab), *trio
                    )
                )
                if lab in _clabs.keys():
                    f.write('      "{}"\n'.format(_clabs[lab]))
                else:
                    f.write("\n")

        if verbose:
            logger.info("Molecule CHEMCRAFT.xyz file written to %s", file_name)
        return
