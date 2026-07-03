# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define domain models for paramagnetic NMR experiments.

Provides Experiment and Signal containers used across the library.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


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

    def __init__(
        self,
        shift: float,
        width: float,
        area: float,
        assignment: str = "UNK",
        l_to_g: float = 1.0,
        r1: Optional[float] = None,
        isotope: Optional[str] = None,
    ) -> None:
        try:
            self.shift = float(shift)
        except (TypeError, ValueError) as exc:
            raise TypeError("shift must be floatable") from exc

        try:
            self.width = float(width)
        except (TypeError, ValueError) as exc:
            raise TypeError("width must be floatable") from exc

        try:
            self.area = float(area)
        except (TypeError, ValueError) as exc:
            raise TypeError("area must be floatable") from exc

        if self.width < 0.0:
            raise ValueError("width must be non-negative")
        if self.area < 0.0:
            raise ValueError("area must be non-negative")

        if assignment is None or (
            isinstance(assignment, str) and not assignment.strip()
        ):
            raise ValueError("assignment must be a non-empty string")
        if not isinstance(assignment, str):
            raise TypeError("assignment must be str")
        self.assignment = assignment.strip()

        try:
            self.l_to_g = float(l_to_g)
        except (TypeError, ValueError) as exc:
            raise TypeError("l_to_g must be floatable") from exc
        if self.l_to_g < 0.0:
            raise ValueError("l_to_g must be non-negative")

        if r1 is None:
            self.r1 = None
        else:
            try:
                self.r1 = float(r1)
            except (TypeError, ValueError) as exc:
                raise TypeError("r1 must be floatable") from exc
            if self.r1 < 0.0:
                raise ValueError("r1 must be non-negative")

        if isotope is None:
            self.isotope = None
        else:
            if not isinstance(isotope, str):
                raise TypeError("isotope must be str or None")
            self.isotope = isotope.strip() or None

        return


class Experiment:
    """Represents a paramagnetic NMR experiment at a single temperature.

    Args:
        temperature: Experiment temperature in Kelvin.
        magnetic_field: Spectrometer magnetic field in Tesla.
        signals: List of assigned signals.
        spectrum: Optional experimental spectrum as an ``(N, 2)`` array where
            the first column is ppm and the second column is intensity.
        exp_reference: Optional experimental reference position in ppm used for
            spectrum normalization/overlay.

    Attributes:
        temperature: Experiment temperature in Kelvin.
        signals: List of assigned signals.
        magnetic_field: Spectrometer magnetic field in Tesla.
        spectrum: Experimental spectrum as an ``(N, 2)`` array or ``None``.
        exp_reference: Experimental reference position in ppm or ``None``.
    """

    def __init__(
        self,
        temperature: float,
        magnetic_field: float,
        signals: list[Signal],
        spectrum: ArrayLike = None,
        exp_reference: Optional[float] = None,
    ) -> None:

        self._signals = signals
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.exp_reference = exp_reference

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

    def __contains__(self, item):
        if isinstance(item, tuple):
            label, isotope = item
            # Exact match first.
            if any(s.assignment == label and s.isotope == isotope for s in self.signals):
                return True
            # Fall back to untagged only when the experiment has no isotope
            # column at all (all signals untagged).
            exp_has_isotopes = any(s.isotope is not None for s in self.signals)
            if not exp_has_isotopes:
                return any(s.assignment == label for s in self.signals)
            return False
        return any(signal.assignment == item for signal in self.signals)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            label, isotope = item
            # Exact isotope match.
            for signal in self.signals:
                if signal.assignment == label and signal.isotope == isotope:
                    return signal
            # Fall back to untagged only when the experiment has no isotope column.
            exp_has_isotopes = any(s.isotope is not None for s in self.signals)
            if not exp_has_isotopes:
                for signal in self.signals:
                    if signal.assignment == label:
                        return signal
            raise KeyError(item)
        lookup = {signal.assignment: signal for signal in self.signals}
        return lookup[item]

    def __iter__(self):
        return iter(self.signals)

    @property
    def exp_reference(self) -> Optional[float]:
        """Experimental reference position in ppm."""
        return self._exp_reference

    @exp_reference.setter
    def exp_reference(self, value: Optional[float]):
        if value is None or value == "":
            self._exp_reference = None
            return
        try:
            self._exp_reference = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("exp_reference must be floatable (ppm)") from exc
        return

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
                signal.assignment.ljust(width),
                signal.shift, signal.width, signal.area,
            )
        return out
