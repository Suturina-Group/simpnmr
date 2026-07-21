# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define domain models for molecular structure and NMR-active nuclei.

Provides Molecule, Nucleus, and ElectronicState containers used across the library.
"""

import copy
import logging
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

from simpnmr.core.const import isotopes, ptable
from simpnmr.core.domain.relax import RelaxationEvaluation
from simpnmr.core.domain.tensor import Hyperfine, Shift, Susceptibility
from simpnmr.core.util import transform as tfm
from simpnmr.core.util.arrays import flatten
from simpnmr.core.util.text import subtitle, title
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)


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
        shift: Shift = Shift(),  # TODO: switch to Shift | None = None once deepcopy removed #noqa
        chem_label: str = None,
        chem_math_label: str = None,
        isotope: str = None,
    ) -> None:
        # Label with and without indexing
        self.label = label
        self.label_nn = xyzf.remove_label_indices(self.label)

        # [REDUCE] Avoid deepcopy in domain unless ownership/mutability requires it.
        self.A = copy.deepcopy(A)

        # [REDUCE] Avoid deepcopy in domain; construct copies in factories if needed.
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
            self.isotope = isotopes.DEFAULT_ISOTOPES.get(self.label_nn)

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
    def isotope(self, value: str | None):
        if value is None:
            self._isotope = None
        elif re.sub("[0-9]", "", value) != self.label_nn:
            raise ValueError("Isotope label does not match atomic label")
        elif value not in isotopes.SUPPORTED_ISOTOPES:
            raise ValueError(f"Unsupported isotope {value}")
        else:
            self._isotope = value


class ElectronicState:
    """Electronic/magnetic state of the system (spin Hamiltonian metadata).

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


class SpinHamiltonian:
    """Spin-Hamiltonian parameters used across pNMR models.

    This container is intended to hold magnetic-model tensors/parameters that are
    shared across the molecule (e.g., g-tensor, ZFS parameters). It is independent
    from `ElectronicState`, which stores quantum-number metadata.

    Attributes:
        g_tensor_ab_initio: Optional 3x3 ab-initio g-tensor matrix.
        g_tensor_ab_initio_iso: Optional isotropic ab-initio g component.
        g_tensor_ab_initio_ax: Optional axial ab-initio g component.
        g_tensor_ab_initio_rho: Optional rhombic ab-initio g component.
        g_tensor_dft: Optional 3x3 DFT-derived g-tensor matrix.
        D_tensor: Optional 3x3 zero-field splitting (ZFS) D tensor.
        E: Optional scalar E parameter (alternative ZFS representation).
    """

    def __init__(
        self,
        g_tensor_ab_initio: ArrayLike | None = None,
        g_tensor_ab_initio_iso: float | None = None,
        g_tensor_ab_initio_ax: float | None = None,
        g_tensor_ab_initio_rho: float | None = None,
        g_tensor_dft: ArrayLike | None = None,
        D_tensor: ArrayLike | None = None,
        E: float | None = None,
    ) -> None:
        self.g_tensor_ab_initio = g_tensor_ab_initio
        self.g_tensor_ab_initio_iso = g_tensor_ab_initio_iso
        self.g_tensor_ab_initio_ax = g_tensor_ab_initio_ax
        self.g_tensor_ab_initio_rho = g_tensor_ab_initio_rho
        self.g_tensor_dft = g_tensor_dft
        self.D_tensor = D_tensor
        self.E = E

    @property
    def g_tensor_ab_initio(self) -> NDArray | None:
        return self._g_tensor_ab_initio

    @g_tensor_ab_initio.setter
    def g_tensor_ab_initio(self, value: ArrayLike | None) -> None:
        if value is None:
            self._g_tensor_ab_initio = None
            return

        arr = np.asarray(value, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(
                "SpinHamiltonian.g_tensor_ab_initio must be a (3, 3) matrix"
            )
        self._g_tensor_ab_initio = arr
        return

    @property
    def g_tensor_ab_initio_iso(self) -> float | None:
        return self._g_tensor_ab_initio_iso

    @g_tensor_ab_initio_iso.setter
    def g_tensor_ab_initio_iso(self, value: float | None) -> None:
        if value is None:
            self._g_tensor_ab_initio_iso = None
            return
        self._g_tensor_ab_initio_iso = float(value)
        return

    @property
    def g_tensor_ab_initio_ax(self) -> float | None:
        return self._g_tensor_ab_initio_ax

    @g_tensor_ab_initio_ax.setter
    def g_tensor_ab_initio_ax(self, value: float | None) -> None:
        if value is None:
            self._g_tensor_ab_initio_ax = None
            return
        self._g_tensor_ab_initio_ax = float(value)
        return

    @property
    def g_tensor_ab_initio_rho(self) -> float | None:
        return self._g_tensor_ab_initio_rho

    @g_tensor_ab_initio_rho.setter
    def g_tensor_ab_initio_rho(self, value: float | None) -> None:
        if value is None:
            self._g_tensor_ab_initio_rho = None
            return
        self._g_tensor_ab_initio_rho = float(value)
        return

    @property
    def g_tensor_dft(self) -> NDArray | None:
        return self._g_tensor_dft

    @g_tensor_dft.setter
    def g_tensor_dft(self, value: ArrayLike | None) -> None:
        if value is None:
            self._g_tensor_dft = None
            return

        arr = np.asarray(value, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("SpinHamiltonian.g_tensor_dft must be a (3, 3) matrix")
        self._g_tensor_dft = arr
        return

    @property
    def D_tensor(self) -> NDArray | None:
        return self._D_tensor

    @D_tensor.setter
    def D_tensor(self, value: ArrayLike | None) -> None:
        if value is None:
            self._D_tensor = None
            return

        arr = np.asarray(value, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("SpinHamiltonian.D_tensor must be a (3, 3) matrix")
        self._D_tensor = arr
        return


class Molecule:
    def _calculate_fc_gcorr_delta(self) -> None:
        """Compute spin-only FC reference and g-correction delta for all nuclei.

        This diagnostic is only available when the canonical susceptibility was
        built with both a stored spin-only isotropic susceptibility baseline and
        a stored g-corrected isotropic susceptibility.

        The canonical molecule state is preserved. Spin-only FC values are
        evaluated using a copied susceptibility object with ``chi.iso`` replaced
        by ``chi.iso_spin_only``.
        """
        if self.susc.iso_g_corr is None or self.susc.iso_spin_only is None:
            return

        chi_spin_only = copy.deepcopy(self.susc)
        chi_spin_only.iso = chi_spin_only.iso_spin_only

        for nuc in self.nuclei:
            fc_spin_only = Shift.calc_fcs(nuc.A, chi_spin_only)
            nuc.shift.fc_spin_only = fc_spin_only
            nuc.shift.fc_delta_g_corr = nuc.shift.fc - fc_spin_only

        return

    """Molecular container holding structure, available HFC data, and runtime nuclei.

    Args:
        labels: Atomic labels for the full structure (no indices).
        coords: Atomic coordinates for the full structure as an ``(n_atoms, 3)``
            array in Å.
        nuclei: List of runtime `Nucleus` objects used by magnetic workflows.

    Attributes:
        labels: Full atomic labels with indices for the whole structure.
        coords: Full atomic coordinates as an ``(n_atoms, 3)`` array in Å.
        paramagnetic_centre: Optional Cartesian coordinates of the canonical
            paramagnetic centre as a length-3 array in Å.
        chi_source_labels: Optional full atomic labels from the
            susceptibility/chi source geometry.
        chi_source_coords: Optional full atomic coordinates from the
            susceptibility/chi source geometry, stored as an ``(n_atoms, 3)``
            array in Å.
        n_atoms: Number of atoms in the full structure.
        available_hfc_by_label: Canonical hyperfine payload store keyed by atom
            label for all HFC data available from the source.
        nuclei: Runtime nuclei used by downstream magnetic workflows.
        susc: Magnetic susceptibility tensor for the molecule.
        electronic: Electronic state metadata (spin/orbit/J model selection).
        sh: Spin-Hamiltonian parameters (e.g. g-tensor, ZFS),
            shared across the molecule.
        relaxation: Optional relaxation evaluation results attached during
            prediction workflows. Defaults to ``None`` when relaxation is not
            computed.
        metadata: Dictionary for domain-level metadata and model provenance.
            Stores final, effective modelling decisions that affect downstream
            physics (e.g. availability of orbital hyperfine contributions).
    """

    def __init__(
        self, labels: NDArray[np.str_], coords: NDArray, nuclei: list[Nucleus]
    ) -> None:
        self.labels = xyzf.add_label_indices(labels)
        self.coords = coords
        self.paramagnetic_centre = None
        self.chi_source_labels = None
        self.chi_source_coords = None

        # List of Nucleus objects
        self.nuclei = nuclei

        # Canonical HFC store for all source-available hyperfine data.
        self.available_hfc_by_label: dict[str, Hyperfine] = {}

        # Susceptibility object
        self.susc = copy.deepcopy(Susceptibility())

        # Electronic state and spin Hamiltonian as separate attributes
        self.electronic = ElectronicState()
        self.sh = SpinHamiltonian()
        self.relaxation: RelaxationEvaluation | None = None

        # Domain-level metadata
        self.metadata: dict[str, dict[str, object]] = {}

    @property
    def n_atoms(self):
        return len(self.labels)

    @property
    def paramagnetic_centre(self) -> NDArray | None:
        return self._paramagnetic_centre

    @paramagnetic_centre.setter
    def paramagnetic_centre(self, value: ArrayLike | None) -> None:
        if value is None:
            self._paramagnetic_centre = None
            return

        arr = np.asarray(value, dtype=float)
        if len(arr.shape) != 1 or arr.shape[0] != 3:
            raise ValueError("paramagnetic_centre must be a length-3 array")
        self._paramagnetic_centre = arr
        return

    @property
    def chi_source_labels(self) -> NDArray[np.str_] | None:
        return self._chi_source_labels

    @chi_source_labels.setter
    def chi_source_labels(self, value: ArrayLike | None) -> None:
        if value is None:
            self._chi_source_labels = None
            return

        arr = np.asarray(value)
        if len(arr.shape) != 1:
            raise ValueError("chi_source_labels must be a 1D array")
        self._chi_source_labels = np.asarray([str(label) for label in arr])
        self._validate_chi_source_geometry()
        return

    @property
    def chi_source_coords(self) -> NDArray | None:
        return self._chi_source_coords

    @chi_source_coords.setter
    def chi_source_coords(self, value: ArrayLike | None) -> None:
        if value is None:
            self._chi_source_coords = None
            return

        arr = np.asarray(value, dtype=float)
        if len(arr.shape) != 2 or arr.shape[1] != 3:
            raise ValueError("chi_source_coords must be an (n_atoms, 3) array")
        self._chi_source_coords = arr
        self._validate_chi_source_geometry()
        return

    def _validate_chi_source_geometry(self) -> None:
        """Validate the optional chi-source geometry stored on the molecule.

        Raises:
            ValueError: If chi-source labels/coordinates disagree in length or
                do not match the full molecule atom count.
        """
        if self.chi_source_labels is None or self.chi_source_coords is None:
            return

        if len(self.chi_source_labels) != len(self.chi_source_coords):
            raise ValueError(
                "chi_source_labels and chi_source_coords must have matching lengths"
            )

        if len(self.chi_source_labels) != self.n_atoms:
            raise ValueError(
                "chi-source geometry must match the full molecule atom count"
            )

        # TODO(domain): Support automatic chi-source label alignment when the
        # susceptibility-source geometry contains the same atoms but arrives in
        # a different order. For now, require the indexed label order to match
        # Molecule.labels exactly.
        indexed_chi_labels = xyzf.add_label_indices(self.chi_source_labels)
        if list(indexed_chi_labels) != list(self.labels):
            raise ValueError(
                "chi_source_labels must match Molecule.labels in the same order"
            )

    def __str__(self):
        string = ""

        string += title("Molecule Hyperfine Data")

        string += subtitle("Isotropic A values (ppm Å^-3)")

        for nuc in self.nuclei:
            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f"{nuc.chem_label} ({nuc.label})"
            a_iso = 1.0 / 3.0 * np.trace(nuc.A.fc)
            string += f"{label} {a_iso: .6f}\n"

        string += subtitle("Anisotropic (traceless) A Tensor (ppm Å^-3)")

        for nuc in self.nuclei:
            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f"{nuc.chem_label} ({nuc.label})"

            string += "\n{:} {: .6f} {: .6f} {: .6f}\n".format(
                " " * len(label), *nuc.A.sd[0]
            )
            string += "{:} {: .6f} {: .6f} {: .6f}\n".format(label, *nuc.A.sd[1])
            string += "{:} {: .6f} {: .6f} {: .6f}\n".format(
                " " * len(label), *nuc.A.sd[2]
            )

        return string

    @classmethod
    def from_labels_coords(
        cls,
        labels: ArrayLike,
        coords: ArrayLike,
        elements: list[str] | str = "all",
        exclude: list[str] | None = None,
    ) -> "Molecule":
        """Create a `Molecule` from labels and coordinates.

        Args:
            labels: Atomic labels.
            coords: Atomic coordinates as an ``(n_atoms, 3)`` array-like in Å.
            elements: Elements/labels to include. Use ``"all"`` to include all.
            exclude: Explicit atom labels to remove after the include filter.

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
            elif "all_" in ele or ele in ptable.elements:
                if "all_" in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [la for la in labels_list if _e == xyzf.remove_label_indices(la)]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        exclude_set: set[str] = set(exclude) if exclude else set()

        # Generate list of Nuclei, one for each atom
        # selecting only those elements requested by user
        nuclei = [
            Nucleus(label, coord, Hyperfine())
            for label, coord in zip(labels_list, coords)
            if label in elements_to_include and label not in exclude_set
        ]

        # Generate Molecule using ALL labels and coords
        base = cls(labels_list, coords, nuclei)

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

    def set_available_hfc_by_label(
        self,
        hfc_by_label: dict[str, Hyperfine],
    ) -> None:
        """Set canonical available HFC payload and project it onto runtime nuclei.

        Args:
            hfc_by_label: Hyperfine payload keyed by atom label for all HFC data
                available from the source.
        """
        self.available_hfc_by_label = {
            str(label): copy.deepcopy(hfc) for label, hfc in hfc_by_label.items()
        }

        for nuc in self.nuclei:
            label = str(nuc.label)
            if label not in self.available_hfc_by_label:
                continue
            nuc.A = copy.deepcopy(self.available_hfc_by_label[label])

    def apply_frame_rotation(self, rot_mat: ArrayLike) -> None:
        """Apply a frame rotation to canonical molecule state.

        The canonical coordinate set and canonical available HFC store are
        rotated first. Runtime nucleus coordinates and runtime hyperfine payload
        are then re-projected from the rotated canonical state.

        Args:
            rot_mat: Rotation matrix with shape ``(3, 3)``.

        Raises:
            ValueError: If the rotation matrix does not have shape ``(3, 3)``.
        """
        rot_arr = np.asarray(rot_mat, dtype=float)
        if rot_arr.shape != (3, 3):
            raise ValueError("rot_mat must be a (3, 3) matrix")

        self.coords = tfm.rotate_coords(self.coords, rot_arr)

        if self.paramagnetic_centre is not None:
            self.paramagnetic_centre = tfm.rotate_coords(
                np.asarray([self.paramagnetic_centre], dtype=float),
                rot_arr,
            )[0]

        if self.chi_source_coords is not None:
            self.chi_source_coords = tfm.rotate_coords(
                self.chi_source_coords,
                rot_arr,
            )

        rotated_hfc_by_label: dict[str, Hyperfine] = {}
        for label, hfc in self.available_hfc_by_label.items():
            rotated_hfc = copy.deepcopy(hfc)
            rotated_hfc.fc = tfm.rotate_tensor(rotated_hfc.fc, rot_arr)
            rotated_hfc.sd = tfm.rotate_tensor(rotated_hfc.sd, rot_arr)
            rotated_hfc.orb = tfm.rotate_tensor(rotated_hfc.orb, rot_arr)
            if rotated_hfc.tensor_full is not None:
                rotated_hfc.tensor_full = tfm.rotate_tensor(
                    rotated_hfc.tensor_full,
                    rot_arr,
                )
            rotated_hfc_by_label[str(label)] = rotated_hfc

        self.set_available_hfc_by_label(rotated_hfc_by_label)

        coord_by_label = {
            str(label): np.asarray(coord, dtype=float)
            for label, coord in zip(self.labels, self.coords)
        }
        for nuc in self.nuclei:
            label = str(nuc.label)
            if label in coord_by_label:
                nuc.coord = coord_by_label[label]

        self.metadata["frame"] = "chi"

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
            group = [nuc for nuc in self.nuclei if nuc.chem_label in ents]

            avg_fc = np.mean([nuc.A.fc for nuc in group], axis=0)
            avg_sd = np.mean([nuc.A.sd for nuc in group], axis=0)
            avg_orb = np.mean([nuc.A.orb for nuc in group], axis=0)

            have_full = all(nuc.A.tensor_full is not None for nuc in group)
            avg_full = None
            if have_full:
                avg_full = np.mean([nuc.A.tensor_full for nuc in group], axis=0)

            for nuc in group:
                nuc.A.fc = avg_fc
                nuc.A.sd = avg_sd
                nuc.A.orb = avg_orb
                if have_full:
                    nuc.A.tensor_full = avg_full

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
            nuc.A.fc = rot_mat @ nuc.A.fc @ rot_mat.T
            nuc.A.sd = rot_mat @ nuc.A.sd @ rot_mat.T
            nuc.A.orb = rot_mat @ nuc.A.orb @ rot_mat.T
            if nuc.A.tensor_full is not None:
                nuc.A.tensor_full = rot_mat @ nuc.A.tensor_full @ rot_mat.T

        return

    def calc_pdip(self) -> None:
        """Add point-dipole dipolar hyperfine contributions to canonical HFC state.

        Raises:
            ValueError: If `Molecule.paramagnetic_centre` is not set.
        """
        if self.paramagnetic_centre is None:
            raise ValueError("Molecule.paramagnetic_centre must be set")

        centre_coord = np.asarray(self.paramagnetic_centre, dtype=float)

        updated_hfc_by_label = {
            str(label): copy.deepcopy(hfc)
            for label, hfc in self.available_hfc_by_label.items()
        }

        for nuc in self.nuclei:
            label = str(nuc.label)
            hfc = copy.deepcopy(updated_hfc_by_label.get(label, Hyperfine()))
            val = Hyperfine.calc_pdip(nuc.coord, centre_coord)
            val *= 1e6
            hfc.sd = hfc.sd + val
            hfc.tensor_full = hfc.fc + hfc.sd + hfc.orb
            updated_hfc_by_label[label] = hfc

        self.set_available_hfc_by_label(updated_hfc_by_label)
        return

    def calculate_shifts(self):
        """Compute chemical shift components for all nuclei.

        This method computes all standard shift contributions that are available
        from the current molecule domain state. Fermi-contact and pseudocontact
        contributions are always evaluated.
        When both spin-only and g-corrected isotropic susceptibility values are
        available, the method also stores spin-only FC reference values and the
        corresponding FC g-correction deltas.
        Orbital contributions are evaluated
        only when the hyperfine metadata reports orbital contribution as
        available and a DFT-derived g-tensor is present in the spin-Hamiltonian
        container.

        Raises:
            ValueError: If orbital contribution is marked as available but the
                required DFT-derived g-tensor is missing.
        """

        hyperfine_meta = self.metadata.get("hyperfine", {})
        orb_available = hyperfine_meta.get("orbital_contribution") == "available"

        g_tensor_dft = None
        if orb_available:
            g_tensor_dft = self.sh.g_tensor_dft
            if g_tensor_dft is None:
                raise ValueError(
                    "Orbital contribution is marked as available, but "
                    "SpinHamiltonian.g_tensor_dft is missing."
                )

        for nuc in self.nuclei:
            nuc.shift.pc = Shift.calc_pcs(nuc.A, self.susc)
            nuc.shift.fc = Shift.calc_fcs(nuc.A, self.susc)
            nuc.shift.pc_tensor = Shift.calc_pcs_tensor(nuc.A, self.susc)
            nuc.shift.fc_tensor = Shift.calc_fcs_tensor(nuc.A, self.susc)

            if orb_available:
                nuc.shift.orb_iso = Shift.calc_orb_iso(nuc.A, self.susc, g_tensor_dft)
                nuc.shift.orb_aniso = Shift.calc_orb_aniso(
                    nuc.A, self.susc, g_tensor_dft
                )
                nuc.shift.orb_tensor = Shift.calc_orb_tensor(
                    nuc.A, self.susc, g_tensor_dft
                )
            else:
                nuc.shift.orb_iso = 0.0
                nuc.shift.orb_aniso = 0.0
                nuc.shift.orb_tensor = np.zeros((3, 3), dtype=float)

        self._calculate_fc_gcorr_delta()
        return

    def apply_diamagnetic_shifts(
        self,
        dia_by_key: dict,
        key_kind: str,
        ref_avg_by_isotope: dict[str, float] | None = None,
    ) -> None:
        """Apply diamagnetic shifts to nuclei.

        Args:
            dia_by_key: Mapping from label key -> diamagnetic shielding (or
                shift for pre-referenced CSV inputs).  Keys are either plain
                strings (label only) or ``(label, isotope)`` tuples when the
                dia CSV includes an ``isotope`` column.
            key_kind: ``'atom_label'`` (uses ``nuc.label``) or
                ``'chem_label'`` (uses ``nuc.chem_label``).
            ref_avg_by_isotope: Optional mapping isotope -> averaged reference
                shielding (e.g. ``{"1H": 31.74, "13C": 188.07}``).
                When provided, applies: dia = ref - dia, converting absolute
                DFT shieldings into chemical shifts relative to the reference.

        Raises:
            KeyError: If a required key is missing in the provided mapping(s).
            ValueError: If key_kind is unsupported.
        """
        if key_kind not in ("atom_label", "chem_label"):
            raise ValueError("key_kind must be 'atom_label' or 'chem_label'")

        missing: list[str] = []
        for nuc in self.nuclei:
            key = nuc.label if key_kind == "atom_label" else nuc.chem_label
            iso_key = (key, nuc.isotope) if nuc.isotope is not None else None
            if iso_key is not None and iso_key in dia_by_key:
                nuc.shift.dia = float(dia_by_key[iso_key])
            elif key in dia_by_key:
                nuc.shift.dia = float(dia_by_key[key])
            else:
                missing.append(key)
        if missing:
            raise KeyError(
                f"Diamagnetic shift missing for {len(missing)} nucleus/"
                f"nuclei: {', '.join(missing)}.\n"
                "These atoms are in your molecule but not covered by the "
                "diamagnetic file. Check your nuclei:isotope / "
                "nuclei:include_groups / nuclei:exclude_groups settings."
            )

        if ref_avg_by_isotope is not None:
            for nuc in self.nuclei:
                try:
                    nuc.shift.dia = (
                        float(ref_avg_by_isotope[nuc.isotope]) - nuc.shift.dia
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Cannot find isotope {nuc.isotope!r} in reference "
                        "diamagnetic shift mapping. "
                        "Add it to diamagnetic_ref:values or provide a "
                        "per-isotope reference file."
                    ) from exc

        return

    def apply_chem_labels(
        self,
        al_to_cl: dict[str, str],
        al_to_cml: dict[str, str] | None = None,
        al_to_isotope: dict[str, str] | None = None,
    ) -> None:
        """Apply chemical label mappings to nuclei.

        This is a pure domain operation: callers must provide pre-parsed
        mappings (e.g. from an application loader).

        Args:
            al_to_cl: Mapping atom_label -> chem_label.
            al_to_cml: Optional mapping atom_label -> chem_math_label.
            al_to_isotope: Optional mapping atom_label -> isotope string
                (e.g. ``"1H"``).  When provided, overrides the nucleus
                default isotope.  Invalid or unsupported entries are logged
                and skipped.

        Returns:
            None.
        """

        # Apply chem_label
        for nuc in self.nuclei:
            cl = al_to_cl.get(nuc.label)
            if cl is not None:
                nuc.chem_label = cl

        # Apply chem_math_label (if provided)
        if al_to_cml is not None:
            for nuc in self.nuclei:
                cml = al_to_cml.get(nuc.label)
                if cml is not None:
                    nuc.chem_math_label = str(cml).strip()
        else:
            # If math labels are not provided, ensure a sensible fallback.
            for nuc in self.nuclei:
                if not len(nuc.chem_math_label):
                    nuc.chem_math_label = nuc.chem_label

        # Apply per-nucleus isotopes (if provided)
        if al_to_isotope is not None:
            for nuc in self.nuclei:
                iso = al_to_isotope.get(nuc.label)
                if iso is not None:
                    try:
                        nuc.isotope = iso
                    except ValueError as exc:
                        logger.warning(
                            "Cannot set isotope '%s' for nucleus '%s': %s",
                            iso,
                            nuc.label,
                            exc,
                        )

        return
