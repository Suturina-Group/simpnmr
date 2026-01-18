# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Core data structures and utilities for paramagnetic NMR.

This module defines objects and methods for working with paramagnetic NMR
spectra, hyperfine couplings, and magnetic susceptibility tensors.
"""

import copy
import csv
import datetime
import logging
import os
import re
from itertools import chain, permutations, product

import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.constants as constants
from numpy.typing import ArrayLike, NDArray

from . import utils as ut
from .__version__ import __version__
from .io.csv import experiment
from .io.qc import qc_readers as rdrs
from .scripts.coords_tools import atoms
from .scripts.coords_tools import label_format as lf
from .scripts.coords_tools import xyz_format as xyzf

logger = logging.getLogger(__name__)


class Signal:
    """Represents a single NMR signal.

    Args:
        shift: Chemical shift in ppm.
        width: Signal width in ppm.
        area: Signal area (integral).
        assignment: Chemical label of the signal.
        l_to_g: Lorentzian-to-Gaussian lineshape ratio.
        r1: Longitudinal relaxation rate in s^-1.

    Attributes:
        shift: Chemical shift in ppm.
        width: Signal width in ppm.
        area: Signal area (integral).
        assignment: Chemical label of the signal.
        l_to_g: Lorentzian-to-Gaussian lineshape ratio.
        r1: Longitudinal relaxation rate in s^-1.
    """

    def __init__(self, shift, width, area, assignment="UNK", l_to_g=1, r1=None):
        self.shift = shift
        self.width = width
        self.area = area
        self.assignment = assignment
        self.l_to_g = l_to_g
        self.r1 = r1

        return


class Experiment:
    """Represents a paramagnetic NMR experiment at a single temperature.

    Args:
        temperature: Experiment temperature in Kelvin.
        magnetic_field: Spectrometer magnetic field in Tesla.
        isotope: Isotope label (e.g., ``"13C"``).
        signals: List of assigned signals.
        spectrum: Optional experimental spectrum as an ``(N, 2)`` array where the
            first column is ppm and the second column is intensity.

    Attributes:
        temperature: Experiment temperature in Kelvin.
        signals: List of assigned signals.
        magnetic_field: Spectrometer magnetic field in Tesla.
        isotope: Isotope label.
        spectrum: Experimental spectrum as an ``(N, 2)`` array or ``None``.
    """

    def __init__(
        self,
        temperature: float,
        magnetic_field: float,
        isotope: str,
        signals: list[Signal],
        spectrum: ArrayLike = None,
    ) -> None:
        self._signals = signals
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.isotope = isotope.title()

        if spectrum is not None:
            self.spectrum = spectrum
        else:
            self._spectrum = None
        return

    def load_spectrum_from_file(self, file_name: str):
        """Loads spectrum data from a CSV file.

        The input file must contain two columns with no header: the first column
        is ppm (shift) and the second column is intensity.

        Args:
            file_name: Path to the CSV file.
        """

        # Read spectrum supporting both comma and any whitespace as separators
        df = pd.read_csv(
            file_name,
            sep=r"\s+",  # tabs/spaces
            header=None,
            comment="#",
            engine="python",
            quoting=csv.QUOTE_NONE,  # treat quotes as normal characters
            converters={
                0: lambda s: float(s.strip("\"'")),
                1: lambda s: float(s.strip("\"'")),
            },
        )

        if df.shape[1] != 2:
            raise ValueError(
                "Spectrum file must contain exactly two columns: ppm and intensity"
            )

        spectrum = df.to_numpy(dtype=float)

        self.spectrum = spectrum
        return

    def keys(self):
        """Returns the list of signal assignments.

        This enables selecting signals by assignment name.

        Returns:
            List of assignment strings.
        """
        return [signal.assignment for signal in self.signals]

    def __getitem__(self, item):
        # This is probably slow
        lookup = {signal.assignment: signal for signal in self.signals}
        return lookup[item]

    def __iter__(self):
        return iter(self.signals)

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value: ArrayLike):
        self._spectrum = np.asarray(value)

    @property
    def signals(self):
        return self._signals

    @signals.setter
    def signals(self, value):
        self._signals = value
        return

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        try:
            value = float(value)
        except TypeError:
            raise TypeError("temperature must be floatable")
        self._temperature = value
        return

    @property
    def magnetic_field(self) -> float:
        return self._magnetic_field

    @magnetic_field.setter
    def magnetic_field(self, value: float):
        try:
            value = float(value)
        except TypeError:
            raise TypeError("magnetic_field must be floatable")
        self._magnetic_field = value
        return

    @property
    def isotope(self) -> str:
        return self._isotope

    @isotope.setter
    def isotope(self, value: str):
        if not isinstance(value, str):
            raise ValueError("isotope must be str")
        self._isotope = value
        return

    def __str__(self):
        out = f"Temperature {self.temperature:f} K\n"
        out += "assignment, shift, width, area\n"
        width = max([len(signal.assignment) for signal in self.signals])
        for signal in self.signals:
            out += "{}, {: 10.4f}, {:7.4f}, {:5.2f}\n".format(
                signal.assignment.ljust(width), signal.shift, signal.width, signal.area
            )
        return out

    @classmethod
    def from_file(cls, file_names: str | list[str]) -> list["Experiment"]:
        """Creates experiments from one or more assignment CSV files.

        Each file is expected to contain signal assignments and parameters
        (shift, width, area, etc.). Additional metadata (temperature, field,
        isotope) is read via ``experiment.read_exp_metadata``.

        Args:
            file_names: Path to a CSV file or a list of CSV files.

        Returns:
            A list of `Experiment` objects (one per input file).

        Raises:
            ValueError: If `file_names` is an empty string or an empty list.
        """

        if not len(file_names):
            raise ValueError(ut.cstr("No files provided", "red"))

        if isinstance(file_names, str):
            file_names = [file_names]

        # Standardise column names
        name_convertor = {
            "shifts": "shift",
            "shifts (ppm)": "shift",
            "shift (ppm)": "shift",
            "ppm": "shift",
            "assignment": "assignment",
            "assignments": "assignment",
            "assignments ()": "assignment",
            "assignment ()": "assignment",
            "widths": "width",
            "width ()": "width",
            "width(Hz)": "width",
            "width (Hz)": "width",
            "widths ()": "width",
            "areas": "area",
            "area ()": "area",
            "areas ()": "area",
            "integral": "area",
            "integral ()": "area",
            "integrals ()": "area",
            "L/G ()": "L/G",
            "r1": "R1",
            "r1 (s^-1)": "R1",
            "R1 (s^-1)": "R1",
            "1/T1": "R1",
            "1/T1 (s^-1)": "R1",
        }
        others = {}
        for key, val in name_convertor.items():
            others[key.capitalize()] = val
            others[val.capitalize()] = val
        name_convertor.update(others)

        # Read each file
        final = []
        for file_name in file_names:
            _data = pd.read_csv(file_name, comment="#", skipinitialspace=True)
            _data.rename(columns=name_convertor, inplace=True)
            _temperature, _magnetic_field, _isotope = experiment.read_exp_metadata(
                file_name
            )
            _data["temperature"] = _temperature
            _data["magnetic_field"] = _magnetic_field
            _data["isotope"] = _isotope
            final.append(_data)

        # combine into a single dataframe
        data = pd.concat(final)
        data.reset_index(inplace=True)

        # Sort by temperature
        data = data.sort_values("temperature")

        # Add linewidth ratio if missing
        if "L/G" not in data.columns:
            data["L/G"] = 1.0

        # Split by mean temperature
        split_indices = ut.find_mean_values(data["temperature"], thresh=0.1)

        if len(split_indices):
            split_indices = [0] + split_indices
            split_indices.append(len(data))
            _exp = [
                data.iloc[split_indices[n] : split_indices[n + 1]]
                for n in range(len(split_indices) - 1)
            ]
        else:
            _exp = [data]

        # Then sort by shift
        for _e in _exp:
            _e.sort_values("shift")
            _e.reset_index(inplace=True)

        # and create experiments
        experiments = [
            cls(
                _e["temperature"][0],
                _e["magnetic_field"][0],
                _e["isotope"][0],
                [
                    Signal(
                        signal["shift"],
                        signal["width"],
                        signal["area"],
                        signal["assignment"],
                        l_to_g=signal["L/G"],
                        r1=signal.get("R1", None),
                    )
                    for _, signal in _e.iterrows()
                ],
            )
            for _e in _exp
        ]

        return experiments

    def to_csv(
        self,
        file_name: str,
        delimiter: str = ",",
        comment: str = "",
        verbose: bool = True,
    ) -> None:
        """Writes the experiment (assigned signals) to a CSV file.

        Args:
            file_name: Output file path.
            delimiter: CSV delimiter.
            comment: Optional additional comment line to append.
            verbose: If ``True``, prints the output file path.

        Returns:
            None.
        """

        data = {
            "assignment ()": [signal.assignment for signal in self.signals],
            "shift (ppm)": [signal.shift for signal in self.signals],
            "width (Hz)": [signal.width for signal in self.signals],
            "area ()": [signal.area for signal in self.signals],
            "L/G ()": [signal.l_to_g for signal in self.signals],
        }

        df = pd.DataFrame(data=data)
        df.sort_values(["shift (ppm)"], inplace=True)

        _comment = (
            f"#This file was generated with SimpNMR v{__version__} at {{}}\n".format(
                datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
            )
        )
        _comment += f"#temperature = {self.temperature:.3f}\n"
        _comment += f"#magnetic_field = {self.magnetic_field:.3f}\n"
        _comment += f"#isotope = {self.isotope}\n"

        _comment += comment + "\n"

        with open(file_name, "w") as _f:
            _f.write(_comment)

            df.to_csv(_f, sep=delimiter, header=True, float_format="%.5f", index=None)

        if verbose:
            logger.info("Assigned experiment saved to %s", file_name)

        return

    @classmethod
    def generate_permutations(
        cls, experiment: "Experiment", groups: list[list[str]] = []
    ) -> list["Experiment"]:
        """Generates all assignment permutations consistent with grouping.

        Args:
            experiment: The reference experiment whose assignments are permuted.
            groups: Groups of assignment labels that may be permuted within each
                group. Assignments not present in any group are treated as fixed.

        Returns:
            A list of permuted assignment lists.
        """

        # Add on fixed assignments by treating each as it were a group
        # of its own
        fixed = [
            [label]
            for label in experiment.keys()
            if label not in np.concatenate(groups)
        ]
        groups += fixed

        # Find all permutations subject to grouping constraints
        _tmp = [permutations(group) for group in groups]
        perms = [list(chain.from_iterable(e)) for e in product(*_tmp, repeat=1)]

        # Convert label groups into indices of experimental signals
        l2i = {label: it for it, label in enumerate(experiment.keys())}
        group_to_exp = [l2i[lab] for lab in np.concatenate(groups)]

        # Order which returns signals listed in groups
        # back to that of original experiment
        order = np.argsort(group_to_exp)

        # Reorder to match original experiment
        all_new_assgn = [[new_assgns[o] for o in order] for new_assgns in perms]

        return all_new_assgn

    @property
    def r1_by_assignment(self):
        """Returns a mapping from assignment to R1 value.

        Returns:
            Dictionary mapping signal assignments to R1 values.
        """
        r1_dict = {
            signal.assignment: signal.r1
            for signal in self.signals
            if signal.r1 is not None
        }
        return r1_dict


class Hyperfine:
    """Hyperfine coupling tensor for a single nucleus.

    Args:
        tensor: Hyperfine tensor as a ``(3, 3)`` NumPy array. Units are
            ``ppm Å^-3``.

    Attributes:
        tensor: Hyperfine tensor as a ``(3, 3)`` NumPy array (``ppm Å^-3``).
        iso: Isotropic hyperfine coupling (``ppm Å^-3``).
        dip: Dipolar hyperfine tensor (``ppm Å^-3``).
        eigvals: Eigenvalues of the total hyperfine tensor.
        eigvecs: Eigenvectors corresponding to ``eigvals``.
    """

    def __init__(self, tensor: NDArray = np.zeros([3, 3])) -> None:
        self._iso = None
        self._dip = None
        self._eigvals = None
        self._eigvecs = None
        self.tensor = copy.deepcopy(tensor)
        pass

    @property
    def tensor(self) -> NDArray:
        """Hyperfine coupling tensor as a ``(3, 3)`` array (``ppm Å^-3``)."""
        return self._tensor

    @tensor.setter
    def tensor(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("A must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("A must be np.array (3x3) of floats")
        self._tensor = intensor

        # Recalculate isotropic hyperfine
        self.calc_iso()
        # and dipolar hyperfine
        self.calc_dip()

        # and reset eigenvalues and eigenvectors to None
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def iso(self) -> float:
        """Isotropic hyperfine coupling constant (``ppm Å^-3``)."""
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
        return

    def calc_iso(self):
        """Computes and stores the isotropic component from `self.tensor`."""
        self.iso = self._calc_iso(self.tensor)
        return

    @staticmethod
    def _calc_iso(tensor: NDArray) -> float:
        """Computes the isotropic component of a hyperfine tensor.

        Args:
            tensor: Hyperfine tensor as a ``(3, 3)`` array.

        Returns:
            The isotropic value (trace/3).
        """
        return np.trace(tensor) / 3.0

    @property
    def dip(self) -> NDArray:
        """Dipolar hyperfine tensor as a ``(3, 3)`` array."""
        return self._dip

    @dip.setter
    def dip(self, tensor: NDArray):
        self._dip = tensor
        return

    def calc_dip(self):
        """Computes and stores the dipolar component from `self.tensor`."""
        self.dip = self._calc_dip(self.tensor)
        return

    @staticmethod
    def _calc_dip(tensor: NDArray) -> NDArray:
        """Computes the dipolar part of a hyperfine tensor.

        Args:
            tensor: Hyperfine tensor as a ``(3, 3)`` array.

        Returns:
            The dipolar tensor ``tensor - I * trace(tensor)/3``.
        """
        return tensor - np.eye(3) * Hyperfine._calc_iso(tensor)

    @property
    def eigvals(self) -> NDArray:
        """Eigenvalues of the hyperfine tensor."""
        # Recalculate if not populated
        if self._eigvals is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: ArrayLike):
        value = np.asarray(value)
        if np.size(value) != 3 or np.shape(value) != (3,):
            raise TypeError("Values must be 3 element arraylike")
        self._eigvals = value
        return

    @property
    def eigvecs(self) -> NDArray:
        """Eigenvectors of the hyperfine tensor (dimensionless)."""
        # Recalculate if not populated
        if self._eigvecs is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        self._eigvecs = intensor
        return

    @staticmethod
    def calc_pdip(r_nuc: ArrayLike, r_elec: ArrayLike = np.zeros(3)):
        """Computes the point-dipole approximation to the dipolar hyperfine tensor.

        Args:
            r_nuc: Nucleus coordinates.
            r_elec: Electron coordinates.

        Returns:
            Dipolar hyperfine tensor as a ``(3, 3)`` array.
        """

        r_nuc = np.asarray(r_nuc)
        r_elec = np.asarray(r_elec)

        r = r_nuc - r_elec

        rnorm = la.norm(r)

        pdip = 3 * np.outer(r, r) / rnorm**5 - np.eye(3) / rnorm**3
        pdip /= 4 * np.pi

        return pdip


class Susceptibility:
    """Magnetic susceptibility tensor.

    Args:
        tensor: Susceptibility tensor as a ``(3, 3)`` NumPy array (Å³).
        temperature: Temperature that this tensor corresponds to (K).

    Attributes:
        tensor: Susceptibility tensor as a ``(3, 3)`` NumPy array (Å³).
        iso: Isotropic susceptibility (Å³).
        dtensor: Deviatoric susceptibility tensor ``tensor - iso * I`` (Å³).
        eigvals: Eigenvalues of `tensor`.
        eigvecs: Eigenvectors corresponding to ``eigvals``.
        alpha: ZYZ Euler alpha angle between the input frame and the eigenframe
            (degrees).
        beta: ZYZ Euler beta angle between the input frame and the eigenframe
            (degrees).
        gamma: ZYZ Euler gamma angle between the input frame and the eigenframe
            (degrees).
        axiality: Axiality of the tensor (Å³).
        rhombicity: Rhombicity of the tensor (Å³).
        irred: Irreducible spherical components (length-5 complex array) ordered
            ``chi_-2, chi_-1, chi_0, chi_1, chi_2``.
        temperature: Temperature that this tensor corresponds to (K).
    """

    def __init__(
        self, tensor: NDArray = np.zeros([3, 3]), temperature: float = 0.0
    ) -> None:
        self._dtensor = None
        self._iso = None
        self._eigvals = None
        self._eigvecs = None
        self._axiality = None
        self._rhombicity = None
        self._irred = None
        self._alpha = None
        self._beta = None
        self._gamma = None

        self.temperature = temperature
        self.tensor = copy.deepcopy(tensor)
        pass

    @property
    def tensor(self) -> NDArray:
        """Susceptibility tensor as a ``(3, 3)`` array (Å³)."""
        return self._tensor

    @tensor.setter
    def tensor(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("Chi must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("Chi must be np.array (3x3) of floats")
        self._tensor = intensor

        # Recalculate isotropic susceptibility
        self.calc_iso()
        # and delta susceptibility
        self.calc_dtensor()

        # and reset eigenvalues and eigenvectors to None
        self._eigvals = None
        self._eigvecs = None
        self._axiality = None
        self._rhombicity = None
        self._irred = None
        self._alpha = None
        self._beta = None
        self._gamma = None
        return

    @property
    def iso(self) -> float:
        """Isotropic susceptibility (Å³)."""
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
        return

    def calc_iso(self):
        """Computes and stores the isotropic component from `self.tensor`."""
        self.iso = self._calc_iso(self.tensor)
        return

    @staticmethod
    def _calc_iso(tensor: NDArray) -> float:
        """Computes the isotropic component of a susceptibility tensor.

        Args:
            tensor: Susceptibility tensor as a ``(3, 3)`` array.

        Returns:
            The isotropic value (trace/3).
        """
        return np.trace(tensor) / 3.0

    @property
    def dtensor(self) -> NDArray:
        """Deviatoric susceptibility tensor as a ``(3, 3)`` array (Å³)."""
        if self._dtensor is None:
            self.calc_dtensor()
        return self._dtensor

    @dtensor.setter
    def dtensor(self, tensor: NDArray):
        self._dtensor = tensor
        return

    def calc_dtensor(self):
        """Computes and stores the delta susceptibility tensor from `self.tensor`."""
        self.dtensor = self._calc_dtensor(self.tensor)
        return

    @staticmethod
    def _calc_dtensor(tensor: NDArray) -> NDArray:
        """Computes the delta susceptibility tensor.

        Args:
            tensor: Susceptibility tensor as a ``(3, 3)`` array.

        Returns:
            The delta susceptibility tensor ``tensor - I * trace(tensor)/3``.
        """
        return tensor - (np.eye(3) * Susceptibility._calc_iso(tensor))

    @property
    def eigvals(self) -> NDArray:
        """Eigenvalues of the susceptibility tensor."""
        # Recalculate if not populated
        if self._eigvals is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: ArrayLike):
        value = np.asarray(value)
        if np.size(value) != 3 or np.shape(value) != (3,):
            raise TypeError("Values must be 3 element arraylike")
        self._eigvals = value
        return

    @property
    def eigvecs(self) -> NDArray:
        """Eigenvectors of the susceptibility tensor (dimensionless)."""
        # Recalculate if not populated
        if self._eigvecs is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        self._eigvecs = intensor
        return

    def calc_eig(self):
        """Computes and stores eigenvalues/eigenvectors of `self.tensor`.

        Returns:
            A tuple ``(vals, vecs)`` as returned by ``numpy.linalg.eigh``.
        """
        vals, vecs = la.eigh(self.tensor)

        self._eigvals = vals[np.argsort(np.abs(vals))]
        self._eigvecs = vecs[:, np.argsort(np.abs(vals))]

        return vals, vecs

    @property
    def axiality(self) -> float:
        if self._axiality is None:
            self.calc_axiality()
        return self._axiality

    @axiality.setter
    def axiality(self, value: float):
        if not isinstance(value, np.floating):
            raise ValueError("Axiality must be a float")
        else:
            self._axiality = value
        return

    def calc_axiality(self):
        devals = la.eigvalsh(self.dtensor)
        self.axiality = 1.5 * devals[np.argmax(np.abs(devals))]
        return

    @property
    def rhombicity(self) -> float:
        if self._rhombicity is None:
            self.calc_rhombicity()
        return self._rhombicity

    @rhombicity.setter
    def rhombicity(self, value: float):
        if not isinstance(value, np.floating):
            raise ValueError("Rhombicity must be a float")
        else:
            self._rhombicity = value
        return

    def calc_rhombicity(self):
        devals = la.eigvalsh(self.dtensor)
        order = np.argsort(np.abs(devals))
        devals = devals[order]
        self.rhombicity = 0.5 * (devals[0] - devals[1])
        return

    @property
    def alpha(self) -> float:
        """ZYZ Euler alpha angle between the input frame and the eigenframe.

        The value is stored in degrees.
        """
        # Calculate if unpopulated
        if self._alpha is None:
            self.calc_euler()
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, np.floating):
            raise ValueError("Alpha must be a float")
        else:
            self._alpha = value
        return

    @property
    def beta(self) -> float:
        """ZYZ Euler beta angle between the input frame and the eigenframe.

        The value is stored in degrees.
        """
        # Calculate if unpopulated
        if self._beta is None:
            self.calc_euler()
        return self._beta

    @beta.setter
    def beta(self, value):
        if not isinstance(value, np.floating):
            raise ValueError("Beta must be a float")
        else:
            self._beta = value
        return

    @property
    def gamma(self) -> float:
        """ZYZ Euler gamma angle between the input frame and the eigenframe.

        The value is stored in degrees.
        """
        # Calculate if unpopulated
        if self._gamma is None:
            self.calc_euler()
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not isinstance(value, np.floating):
            raise ValueError("Gamma must be a float")
        else:
            self._gamma = value
        return

    def calc_euler(self):
        """Computes and stores ZYZ Euler angles mapping input frame to eigenframe.

        Angles are stored in degrees.
        """
        _ev = np.abs(self.eigvals - self.iso)
        order = np.argsort(_ev)
        _vecs = self.eigvecs[:, order]

        self.alpha = np.rad2deg(np.arctan2(_vecs[2, 1], -_vecs[0, 1]))
        self.beta = np.rad2deg(np.arccos(_vecs[1, 1]))
        self.gamma = np.rad2deg(np.arctan2(-_vecs[1, 2], _vecs[1, 0]))
        return

    @property
    def irred(self) -> NDArray:
        """Irreducible spherical components of the susceptibility tensor.

        Returns a length-5 complex array ordered
        ``chi_-2, chi_-1, chi_0, chi_1, chi_2``.
        """
        # Calculate if unpopulated
        if self._irred is None:
            self.calc_irred()
        return self._irred

    @irred.setter
    def irred(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "Irreducible Spherical Components must be 5 element "
                "arraylike of complex numbers"
            )
        elif value.shape != (5,):
            raise TypeError(
                "Irreducible Spherical Components must be 5 element "
                "arraylike of complex numbers"
            )
        elif not np.iscomplexobj(value):
            raise TypeError(
                "Irreducible Spherical Components must be 5 element "
                "arraylike of complex numbers"
            )
        self._irred = value
        return

    def calc_irred(self):
        """Computes and stores irreducible spherical components from `self.tensor`."""
        self.irred = self._calc_irred(self.tensor)
        return

    @staticmethod
    def _calc_irred(tensor: NDArray) -> NDArray:
        """Computes irreducible spherical components of a susceptibility tensor.

        Note:
            This does not include any isotropic contribution.

        Args:
            tensor: Real susceptibility tensor as a ``(3, 3)`` array.

        Returns:
            A length-5 complex128 array ordered ``chi_-2, chi_-1, chi_0, chi_1, chi_2``.
        """
        irred = np.zeros(5, dtype=np.complex128)
        # chi_-2
        irred[0] = +np.sqrt(2 * np.pi / 15) * (
            tensor[0, 0] - tensor[1, 1] + 1j * (tensor[0, 1] + tensor[1, 0])
        )
        # chi_-1
        irred[1] = -np.sqrt(2 * np.pi / 15) * (
            tensor[0, 2] - tensor[2, 0] + 1j * (tensor[1, 2] + tensor[2, 1])
        )
        # chi_0
        irred[2] = +np.sqrt(4 * np.pi / 45) * (
            2 * tensor[2, 2] - tensor[0, 0] - tensor[1, 1]
        )
        # chi_+1
        irred[3] = -np.sqrt(2 * np.pi / 15) * (
            tensor[0, 2] - tensor[2, 0] - 1j * (tensor[1, 2] + tensor[2, 1])
        )
        # chi_2
        irred[4] = +np.sqrt(2 * np.pi / 15) * (
            tensor[0, 0] - tensor[1, 1] - 1j * (tensor[0, 1] + tensor[1, 0])
        )

        return irred

    @classmethod
    def from_csv(cls, file_name: str) -> list["Susceptibility"]:
        """Loads susceptibility tensors from a CSV file.

        The CSV header is expected to match the format written by
        `Susceptibility.save_susc`.

        Args:
            file_name: Path to the CSV file.

        Returns:
            A list of susceptibility tensors.
        """

        data = pd.read_csv(
            file_name, skipinitialspace=True, index_col=False, comment="#"
        )

        # Forward conversion, A^3 --> Key
        convs = {
            "(A^3)": 1.0,
            "(Å^3 mol^-1)": constants.Avogadro,
            "(A^3 mol^-1)": constants.Avogadro,
            "(cm^3)": 1e-24,
            "(cm^3 mol^-1)": 1e-24 * constants.Avogadro / (4 * np.pi),
        }

        # Check names of columns and convert units to angstrom cubed
        renamer = {}
        for name in data.keys():
            for unit in convs.keys():
                # Apply conversion backwards
                if unit in name:
                    data[name] /= convs[unit]
                    renamer[name] = name.replace(unit, "(Å^3)")

        # Rename column headers to angstrom cubed
        data.rename(renamer, inplace=True, axis=1)

        # Read susceptibility tensor
        suscs = [
            cls(
                np.array(
                    [
                        [row["chi_xx (Å^3)"], row["chi_xy (Å^3)"], row["chi_xz (Å^3)"]],
                        [row["chi_xy (Å^3)"], row["chi_yy (Å^3)"], row["chi_yz (Å^3)"]],
                        [row["chi_xz (Å^3)"], row["chi_yz (Å^3)"], row["chi_zz (Å^3)"]],
                    ]
                ),
                temperature=row["Temperature (K)"],
            )
            for _, row in data.iterrows()
        ]

        return suscs

    @classmethod
    def from_orca(cls, file_name: str, section: str) -> list["Susceptibility"]:
        """Loads susceptibility tensors from an ORCA output file.

        Args:
            file_name: Path to the ORCA output file.
            section: Section identifier to extract (e.g., ``"cas"`` or ``"nev"``).

        Returns:
            A list of susceptibility tensors (one per temperature).
        """

        # Extract all possible susceptibility tensors from ORCA output file
        tensors = rdrs.read_orca_susceptibility(file_name, section)

        # Orca units of XT are cm3 mol-1 K, so convert to Angstrom^3 K
        conv = 1e-24 * constants.Avogadro / (4 * np.pi)
        conv = 1 / conv

        suscs = [
            cls(tensor / temperature * conv, temperature=temperature)
            for temperature, tensor in tensors.items()
        ]

        return suscs

    def save_pcs_isosurface(
        self,
        labels: ArrayLike,
        coords: ArrayLike,
        center_atom: str,
        file_name: str = "pcs_isosurf.cube",
        comment="",
        verbose: bool = True,
    ) -> None:
        """Writes a PCS isosurface (cube file) for the current susceptibility tensor.

        PCS values are computed using the point-dipole tensor from `Hyperfine.calc_pdip`
        combined with the delta susceptibility tensor.

        Args:
            labels: Atomic labels (including indices).
            coords: Atomic coordinates in Å.
            center_atom: Label of the paramagnetic center atom (including index).
            file_name: Output cube file name.
            comment: Comment line written to the cube file.
            verbose: If ``True``, prints the output file path.

        Raises:
            ValueError: If `center_atom` is not present in `labels`.
            ValueError: If the delta susceptibility tensor is not initialized.
        """

        labels = np.asarray(labels)
        coords = np.asarray(coords)

        upper = 15
        step = 0.5
        lower = -15

        # Create a 3D grid of points where PCS values will be evaluated
        x, y, z = np.meshgrid(
            np.arange(lower, upper + step, step),
            np.arange(lower, upper + step, step),
            np.arange(lower, upper + step, step),
        )

        # Convert coordinates to atomic units (Bohr)
        coords *= 1.88973

        # Find the central atom's index
        center_idx = np.where(labels == center_atom)[0]
        if len(center_idx) == 0:
            raise ValueError(f"Center atom {center_atom} not found in labels")

        # Shift all coordinates so that the central atom is at the origin
        coords -= coords[center_idx[0]]

        # Ensure that the susceptibility tensor is initialized
        if self.dtensor is None:
            raise ValueError("Susceptibility tensor (dtensor) is not initialized!")

        # Load the deviatoric susceptibility tensor (in Å³)
        chi_tensor = self.dtensor

        # Initialize the isosurface array
        isosurf = np.zeros_like(x, dtype=float)

        # Compute PCS values for each grid point
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    r_point = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])

                    # Avoid division by zero by skipping points at the origin
                    if np.allclose(r_point, np.zeros(3)):
                        continue  # Skip calculation if r is zero

                    # Compute the dipolar hyperfine tensor
                    pdip = Hyperfine.calc_pdip(r_point)

                    # Compute the paramagnetic chemical shift (PCS)
                    pcs_value = (1 / 3) * np.trace(chi_tensor @ pdip)
                    isosurf[i, j, k] = pcs_value

        # Scale the PCS values (convert to ppb)
        isosurf *= 1e7

        # Write the computed PCS isosurface to a cube file
        with open(file_name, "w") as f:
            f.write(f"{comment}\n")
            f.write("Comment line\n")
            f.write(
                "{:d}   {:.6f} {:.6f} {:.6f}\n".format(len(labels), lower, lower, lower)
            )
            f.write("{:d}   {:.6f}    0.000000    0.000000\n".format(x.shape[0], step))
            f.write("{:d}   0.000000    {:.6f}    0.000000\n".format(y.shape[1], step))
            f.write("{:d}   0.000000    0.000000    {:.6f}\n".format(z.shape[2], step))

            # Write atomic labels and coordinates
            for lbl, c in zip(labels, coords):
                f.write(
                    "{:d}   0.000000  {:.6f} {:.6f} {:.6f}\n".format(
                        xyzf.lab_to_num(lbl), *c
                    )
                )

            # Write PCS values into the cube file
            a = 0
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        a += 1
                        f.write("{:.5e} ".format(isosurf[i, j, k]))
                        if a == 6:
                            f.write("\n")
                            a = 0
                    f.write("\n")
                    a = 0

        if verbose:
            logger.info("PCS isosurface written to %s", file_name)

        return


class Shift:
    """Chemical shift components for a nucleus.

    This container tracks diamagnetic and hyperfine contributions to the total
    paramagnetic chemical shift.

    Attributes:
        dia: Diamagnetic chemical shift (ppm).
        fc: Fermi contact chemical shift (ppm).
        pc: Pseudocontact chemical shift (ppm).
        hf: Hyperfine chemical shift (ppm), equal to ``fc + pc``.
        total: Total chemical shift (ppm), equal to ``dia + hf``.
        avg: Averaged total shift (ppm). Defaults to ``total`` and is reset to
            ``total`` whenever any component is modified.
        lw: Linewidth of the signal.
    """

    def __init__(
        self, dia: float = 0.0, pc: float = 0.0, fc: float = 0.0, lw: float = 1.0
    ) -> None:
        self._pc = pc  # Pseudocontact
        self._fc = fc  # Fermi Contact
        self._dia = dia  # Diamagnetic
        self._lw = lw
        self._avg = copy.copy(self.total)
        pass

    @property
    def total(self) -> float:
        return self.dia + self.hf

    @property
    def hf(self) -> float:
        return self.pc + self.fc

    @property
    def avg(self) -> float:
        return self._avg

    @avg.setter
    def avg(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._avg = float(val)
        return

    @property
    def pc(self) -> float:
        return self._pc

    @pc.setter
    def pc(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._pc = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def fc(self) -> float:
        return self._fc

    @fc.setter
    def fc(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._fc = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def dia(self) -> float:
        return self._dia

    @dia.setter
    def dia(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Diamagnetic chemical shift must be a float")
        self._dia = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def lw(self) -> float:
        return self._lw

    @lw.setter
    def lw(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Linewidth must be a float")
        self._lw = float(val)
        return

    @staticmethod
    def calc_pcs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Compute the pseudocontact shift contribution.

        Args:
            A: Hyperfine coupling tensor.
            chi: Magnetic susceptibility tensor.

        Returns:
            The pseudocontact shift (PCS).
        """
        shift = 1.0 / 3.0 * np.trace(chi.dtensor @ A.dip)
        return shift

    @staticmethod
    def calc_fcs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Computes the Fermi contact contribution to the chemical shift."""
        shift = chi.iso * A.iso
        return shift

    @staticmethod
    def calc_hfs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Computes the total hyperfine shift (Fermi contact + PCS)."""
        return Shift.calc_fcs(A, chi) + Shift.calc_pcs(A, chi)


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
            self.isotope = ut.DEFAULT_ISOTOPES[self.label_nn]

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
            raise ValueError(ut.cstr("Nucleus coordinates must be (1x3) array", "red"))

        elif incoord.shape[0] != 3:
            raise ValueError(ut.cstr("Nucleus coordinates must be (1x3) array", "red"))
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
        elif value not in ut.SUPPORTED_ISOTOPES:
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

    def __init__(self):
        self.spin_S = None
        self.orbit_L = None
        self.total_J = None
        self.model = None

    def load_from_config(self, config) -> None:
        """Load all supported electronic-state fields from a config object."""
        self.load_spin(config)
        self.load_orbit(config)
        self.load_total_J(config)

    def load_spin(self, config):
        """Initializes spin quantum number from a config object.

        Priority:
            1) Explicit ``spin_S`` value in the config.
            2) Inference from the QC hyperfine file when possible.

        Args:
            config: A configuration object (typically `FitSuscConfig`).

        Returns:
            The resolved spin quantum number ``S`` or ``None``.
        """

        # 1) Explicit spin from config
        spin = getattr(config, "spin_S", None)

        # 2) Infer from QC file
        if spin is None:
            hf_file = getattr(config, "hyperfine_file", None)
            hf_method = getattr(config, "hyperfine_method", None)

            if hf_file is not None:
                ext = os.path.splitext(hf_file)[1].lower()
                try:
                    if hf_method == "dft" or ext in (".log", ".out"):
                        spin_obj = rdrs.QCSpin.guess_from_file(hf_file)
                        spin = spin_obj.S
                except SystemExit:
                    spin = None

        self.spin_S = spin

        if spin is not None:
            self.model = "spin_only"

        return spin

    def load_orbit(self, config):
        """Initializes orbital angular momentum quantum number from a config.

        Args:
            config: A configuration object.

        Returns:
            The resolved orbital angular momentum ``L`` or ``None``.
        """
        orbit = getattr(config, "orbit", None)

        self.orbit_L = orbit

        if orbit is not None and self.model is None:
            self.model = "orbital"

        return orbit

    def load_total_J(self, config):
        """Initializes total angular momentum quantum number from a config.

        Args:
            config: A configuration object.

        Returns:
            The resolved total angular momentum ``J`` or ``None``.
        """
        total_J = getattr(config, "total_momentum_J", None)

        self.total_J = total_J

        if total_J is not None and self.model is None:
            self.model = "total_J"

        return total_J


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
        cls, labels: list[str], coords: ArrayLike, elements: list[str] | str = "all"
    ) -> "Molecule":
        """Create a `Molecule` from labels and coordinates.

        Args:
            labels: Atomic labels.
            coords: Atomic coordinates as an ``(n_atoms, 3)`` array-like in Å.
            elements: Elements/labels to include. Use ``"all"`` to include all.

        Returns:
            A `Molecule` instance.
        """

        if isinstance(elements, str):
            elements = [elements]

        coords = np.asarray(coords)

        elements_to_include = []
        for ele in elements:
            if ele == "all":
                elements_to_include = labels
                break
            elif "all_" in ele or ele in atoms.elements:
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
        nuclei = [
            Nucleus(label, coord, Hyperfine())
            for label, coord in zip(labels, coords)
            if label in elements_to_include
        ]

        # Generate Molecule using ALL labels and coords
        base = cls(labels, coords, nuclei)

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

        data = pd.read_csv(
            file_name,
            skipinitialspace=True,
            index_col=False,
            engine="python",
            comment="#",
        )

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
            elif "all_" in ele or ele in atoms.elements:
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
            elif "all_" in ele or ele in atoms.elements:
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
            a_isos = ut.a_tensor_mhz_to_angstrom(ab_initio.a_iso)
            a_dips = ut.a_tensor_mhz_to_angstrom(ab_initio.a_dip)
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
            dia = pd.read_csv(file_name, skipinitialspace=True, index_col=False)

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
                ref = pd.read_csv(ref_file_name, skipinitialspace=True, index_col=False)

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
        _fl_av_chemlabels = ut.flatten(av_chemlabels)
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
                ut.cstr(
                    "Error: No paramagnetic centres specified for point dipole",
                    "red",
                )
            )

        # Find user specified centre(s)
        for centre in centre_labels:
            it = [i for i, x in enumerate(self.labels) if x == centre]

            if len(it) > 1:
                raise ValueError(
                    ut.cstr("Error: More than one of specified label found", "red")
                )
            elif not len(it):
                raise ValueError(ut.cstr(f"Cant find {centre} in labels", "red"))

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
            raise ValueError(ut.cstr("Unknown shift specified", "red"))

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

        _tmp = pd.read_csv(file_name, skipinitialspace=True, comment="#")

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
                        ut.cstr(
                            f"Coordinates of {nuc.label} in chem_labels file\n"
                            " do not match those of molecule.",
                            "red",
                        )
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

        data = {
            "atom_label ()": [nuc.label for nuc in self.nuclei],
            "chem_label ()": [nuc.chem_label for nuc in self.nuclei],
            "Aiso (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_xx (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_xy (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_xz (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_yy (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_yz (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_zz (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
        }

        df = pd.DataFrame(data=data)

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

        data = {
            "atom_label ()": [nuc.label for nuc in self.nuclei],
            "chem_label ()": [nuc.chem_label for nuc in self.nuclei],
            "x (Å)": [nuc.coord[0] for nuc in self.nuclei],
            "y (Å)": [nuc.coord[1] for nuc in self.nuclei],
            "z (Å)": [nuc.coord[2] for nuc in self.nuclei],
            "Aiso (ppm Å^-3)": [nuc.A.iso for nuc in self.nuclei],
            "Adip_xx (ppm Å^-3)": [nuc.A.dip[0, 0] for nuc in self.nuclei],
            "Adip_xy (ppm Å^-3)": [nuc.A.dip[0, 1] for nuc in self.nuclei],
            "Adip_xz (ppm Å^-3)": [nuc.A.dip[0, 2] for nuc in self.nuclei],
            "Adip_yy (ppm Å^-3)": [nuc.A.dip[1, 1] for nuc in self.nuclei],
            "Adip_yz (ppm Å^-3)": [nuc.A.dip[1, 2] for nuc in self.nuclei],
            "Adip_zz (ppm Å^-3)": [nuc.A.dip[2, 2] for nuc in self.nuclei],
            "δ_total_avg (ppm)": [nuc.shift.avg for nuc in self.nuclei],
            "δ_total (ppm)": [nuc.shift.total for nuc in self.nuclei],
            "δ_dia (ppm)": [nuc.shift.dia for nuc in self.nuclei],
            "δ_fc (ppm)": [nuc.shift.fc for nuc in self.nuclei],
            "δ_pc (ppm)": [nuc.shift.pc for nuc in self.nuclei],
            "linewidth (Hz)": [1 for _ in self.nuclei],
        }

        df = pd.DataFrame(data=data)

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

    def save_xyz(self, file_name: str, verbose: bool = True, comment: str = ""):
        """Save the current structure to an XYZ file.

        This is a thin wrapper around `xyzf.save_xyz` that also injects a provenance
        comment including the SimpNMR version and timestamp.

        Args:
            file_name: Output XYZ file path.
            verbose: If True, prints the output file path.
            comment: Optional additional comment appended to the provenance line.

        Returns:
            None.
        """

        _comment = (
            f"This file was generated with SimpNMR v{__version__} at {{}}. ".format(
                datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            )
        )
        _comment += comment

        xyzf.save_xyz(
            file_name,
            labels=self.labels,
            coords=self.coords,
            verbose=False,
            comment=_comment,
        )

        if verbose:
            logger.info("Molecule.xyz file written to %s", file_name)
        return
