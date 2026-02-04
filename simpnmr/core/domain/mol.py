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
from simpnmr.core.domain.tensor import Hyperfine, Shift, Susceptibility
from simpnmr.core.util.arrays import flatten
from simpnmr.core.util.text import subtitle, title
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)


class Relaxation:
    """Holds calculated relaxation rates for a nucleus.

    Attributes:
        r1: Longitudinal relaxation rate (s^-1).
        r2: Transverse relaxation rate (s^-1).
    """

    def __init__(
        self,
        r1: float | None = None,
        r2: float | None = None,
        dipolar_r1: float | None = None,
        contact_r1: float | None = None,
        curie_r1: float | None = None,
        dipolar_r2: float | None = None,
        contact_r2: float | None = None,
        curie_r2: float | None = None,
    ) -> None:
        self.r1 = r1
        self.r2 = r2
        self.dipolar_r1 = dipolar_r1
        self.contact_r1 = contact_r1
        self.curie_r1 = curie_r1
        self.dipolar_r2 = dipolar_r2
        self.contact_r2 = contact_r2
        self.curie_r2 = curie_r2


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

    @property
    def n_atoms(self):
        return len(self.labels)

    def __str__(self):
        string = ""

        string += title("Molecule Hyperfine Data")

        string += subtitle("Isotropic A values (ppm Å^-3)")

        for nuc in self.nuclei:
            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f"{nuc.chem_label} ({nuc.label})"

            string += f"{label} {nuc.A.iso: .6f}\n"

        string += subtitle("Anisotropic (dipolar) A Tensor (ppm Å^-3)")

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
            elif "all_" in ele or ele in ptable.elements:
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
    def from_hyperfine_data(
        cls,
        *,
        labels: list[str],
        coords: ArrayLike,
        a_iso: dict[str, float],
        a_dip: dict[str, NDArray],
        elements: list[str] | str = "all",
    ) -> "Molecule":
        """Create a `Molecule` from already-parsed hyperfine data.

        This is a pure domain constructor: no file I/O, no QC parsing, no unit
        conversion. Callers must provide labels/coords and hyperfine tensors in
        consistent units.

        Args:
            labels: Atomic labels (with indices) in the same ordering as `coords`,
                e.g. ["H1", "C2", ...].
            coords: Atomic coordinates as an (n_atoms, 3) array-like in Å.
            a_iso: Mapping atom_label -> isotropic hyperfine coupling.
            a_dip: Mapping atom_label -> dipolar hyperfine tensor (3x3).
            elements: Elements/labels to include. Use "all" to include all atoms,
                "all_H" to include all H, etc., or explicit labels like "H7".

        Returns:
            A `Molecule` instance.

        Raises:
            ValueError: If no nuclei were selected.
        """
        if isinstance(elements, str):
            elements = [elements]

        labels_list = [str(lab) for lab in labels]
        coords_arr = np.asarray(coords)

        elements_to_include: list[str] = []
        for ele in elements:
            if ele == "all":
                elements_to_include = labels_list
                break
            if "all_" in ele or ele in ptable.elements:
                elem = ele[4:] if "all_" in ele else ele
                elements_to_include += [
                    lab for lab in labels_list if elem == xyzf.remove_label_indices(lab)
                ]
            else:
                elements_to_include.append(ele)

        # Filter hyperfine dicts by selection.
        a_iso_sel = {k: v for k, v in a_iso.items() if k in elements_to_include}
        a_dip_sel = {k: v for k, v in a_dip.items() if k in elements_to_include}

        # Filter coords by selection, preserving the original label ordering.
        coords_sel = [
            coord
            for lab, coord in zip(labels_list, coords_arr)
            if lab in elements_to_include
        ]

        nuclei = Nucleus.from_a_values(a_iso_sel, a_dip_sel, coords_sel)
        if not nuclei:
            raise ValueError("No Nuclei selected!")

        # Molecule keeps the full structure labels/coords.
        return cls(labels_list, coords_arr, nuclei)

    @property
    def susc(self) -> Susceptibility:
        return self._susc

    @susc.setter
    def susc(self, new_susc: Susceptibility):
        if not isinstance(new_susc, Susceptibility):
            raise TypeError("Molecule.susc must be of type Susceptibility")
        self._susc = new_susc
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

    def apply_diamagnetic_shifts(
        self,
        dia_by_key: dict[str, float],
        key_kind: str,
        ref_avg_by_label_nn: dict[str, float] | None = None,
    ) -> None:
        """Apply diamagnetic shifts to nuclei.

        Args:
            dia_by_key: Mapping from label key -> dia shift.
            key_kind: 'atom_label' (uses nuc.label) or 'chem_label'
            (uses nuc.chem_label).
            ref_avg_by_label_nn: Optional mapping nuc.label_nn -> averaged
            reference shift.
                If provided, applies: dia := ref - dia.

        Raises:
            KeyError: If a required key is missing in the provided mapping(s).
            ValueError: If key_kind is unsupported.
        """
        if key_kind not in ("atom_label", "chem_label"):
            raise ValueError("key_kind must be 'atom_label' or 'chem_label'")

        for nuc in self.nuclei:
            key = nuc.label if key_kind == "atom_label" else nuc.chem_label
            try:
                nuc.shift.dia = float(dia_by_key[key])
            except KeyError as exc:
                raise KeyError(
                    f"Cannot find {key} in diamagnetic shift mapping"
                ) from exc

        if ref_avg_by_label_nn is not None:
            for nuc in self.nuclei:
                try:
                    nuc.shift.dia = (
                        float(ref_avg_by_label_nn[nuc.label_nn]) - nuc.shift.dia
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Cannot find {nuc.label_nn} in reference diamagnetic "
                        "shift mapping"
                    ) from exc

        return

    def apply_chem_labels(
        self,
        al_to_cl: dict[str, str],
        al_to_cml: dict[str, str] | None = None,
    ) -> None:
        """Apply chemical label mappings to nuclei.

        This is a pure domain operation: callers must provide pre-parsed
        mappings (e.g. from an application loader).

        Args:
            al_to_cl: Mapping atom_label -> chem_label.
            al_to_cml: Optional mapping atom_label -> chem_math_label.

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

        return
