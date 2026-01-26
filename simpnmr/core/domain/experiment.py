# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO"""

import datetime
import logging
from itertools import chain, permutations, product

import numpy as np
from numpy.typing import ArrayLike

from simpnmr import utils as ut
from simpnmr.__version__ import __version__
from simpnmr.io.csv import readers
from simpnmr.mappers import dataframes as ser

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
            raise ValueError("No files provided")

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
            _data = readers.read_csv_safe(file_name)
            _data.rename(columns=name_convertor, inplace=True)
            _temperature, _magnetic_field, _isotope = readers.read_exp_metadata(
                file_name
            )
            _data["temperature"] = _temperature
            _data["magnetic_field"] = _magnetic_field
            _data["isotope"] = _isotope
            final.append(_data)

        # Assemble and normalize experiment table
        data = readers.assemble_experiments_table(final)

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

        df = ser.build_experiment_signals_df(self)

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
