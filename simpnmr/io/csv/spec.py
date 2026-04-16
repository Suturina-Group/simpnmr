# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write spectrum data from CSV files.

Provides helpers to parse and write two-column shift–intensity spectra.
"""

import csv

import pandas as pd

from simpnmr.io.csv.csv_util import read_csv_safe, write_csv_safe


def read_spectrum(file_name: str):
    """Loads spectrum data from a CSV file.

    The input file must contain two columns with no header: the first column
    is ppm (shift) and the second column is intensity.

    Args:
        file_name: Path to the CSV file.
    """

    df = read_csv_safe(
        file_name,
        header=None,
        quoting=csv.QUOTE_NONE,  # treat quotes as normal characters
        converters={
            0: lambda s: float(s.strip().strip("\"'")),
            1: lambda s: float(s.strip().strip("\"'")),
        },
    )

    if df.shape[1] != 2:
        raise ValueError(
            "Spectrum file must contain exactly two columns: ppm and intensity"
        )

    spectrum = df.to_numpy(dtype=float)

    return spectrum


def write_spectrum(
    file_name: str,
    shift_ppm,
    intensity,
    *,
    isotope: str | None = None,
    temperature: float | None = None,
    peak_labels: list[str] | None = None,
    peak_shifts: list[float] | None = None,
) -> None:
    """Write spectrum data to a CSV file.

    Writes a two-column CSV (shift, intensity) with optional metadata encoded
    as ``#``-prefixed comment lines that can be parsed by
    :func:`read_spectrum_with_peaks`:

    - ``# isotope: 1H``
    - ``# temperature: 302.15``
    - ``# peak: H1a,45.23``  (one line per annotated peak)

    Args:
        file_name: Output CSV path.
        shift_ppm: 1D array-like of chemical shifts in ppm.
        intensity: 1D array-like of intensities (arbitrary units).
        isotope: Isotope string, e.g. ``"1H"``.
        temperature: Temperature in K.
        peak_labels: Ordered list of peak label strings.
        peak_shifts: Corresponding peak positions in ppm.
    """
    comments: list[str] = []
    if isotope is not None:
        comments.append(f"isotope: {isotope}")
    if temperature is not None:
        comments.append(f"temperature: {temperature:.4f}")
    if peak_labels and peak_shifts:
        for lbl, s in zip(peak_labels, peak_shifts):
            comments.append(f"peak: {lbl},{s:.6f}")

    df = pd.DataFrame(
        {
            "shift (ppm)": shift_ppm,
            "intensity (a.u.)": intensity,
        }
    )

    write_csv_safe(df, file_name, comment=comments or None)


def read_spectrum_with_peaks(
    file_name: str,
) -> dict:
    """Read a spectrum CSV written by :func:`write_spectrum`.

    Returns a dict with keys:

    - ``"shift"`` – numpy array of ppm values
    - ``"intensity"`` – numpy array of normalised intensities
    - ``"isotope"`` – str or ``None``
    - ``"temperature"`` – float or ``None``
    - ``"peak_labels"`` – list[str] (may be empty)
    - ``"peak_shifts"`` – list[float] (same length as peak_labels)
    - ``"peak_shifts"`` – list[float] (same length as peak_labels)
    """
    import re

    isotope = None
    temperature = None
    peak_labels: list[str] = []
    peak_shifts: list[float] = []

    with open(file_name, encoding="utf-8-sig") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            body = stripped.lstrip("#").strip()
            if body.startswith("isotope:"):
                isotope = body.split(":", 1)[1].strip()
            elif body.startswith("temperature:"):
                try:
                    temperature = float(body.split(":", 1)[1])
                except ValueError:
                    pass
            elif body.startswith("peak:"):
                _pat = r"peak:\s*(.+),\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
                m = re.match(_pat, body)
                if m:
                    peak_labels.append(m.group(1).strip())
                    peak_shifts.append(float(m.group(2)))

    df = read_csv_safe(file_name)
    shift = df.iloc[:, 0].to_numpy(dtype=float)
    intensity = df.iloc[:, 1].to_numpy(dtype=float)

    return {
        "shift": shift,
        "intensity": intensity,
        "isotope": isotope,
        "temperature": temperature,
        "peak_labels": peak_labels,
        "peak_shifts": peak_shifts,
    }
