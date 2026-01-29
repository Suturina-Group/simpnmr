# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO
Domain models for paramagnetic NMR experiments."""

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
