# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read spectrum data from CSV files.

Provides helpers to parse two-column shift–intensity spectra into numeric arrays.
"""

import csv

from simpnmr.io.csv.csv_util import read_csv_safe


def read_spectrum(file_name: str):
    """Loads spectrum data from a CSV file.

    The input file must contain two columns with no header: the first column
    is ppm (shift) and the second column is intensity.

    Args:
        file_name: Path to the CSV file.
    """

    # Strict contract: comma-separated, two columns (ppm, intensity). CSV syntax
    # validation (e.g., trailing comma / extra delimiters) is handled by read_csv_safe.
    df = read_csv_safe(
        file_name,
        sep=",",
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
