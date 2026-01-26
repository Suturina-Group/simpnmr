# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

import csv
import os
import re

import pandas as pd

from ..text import find_first_group
from .validation import validate_csv_delimiters


def read_csv_safe(
    file_name: str | os.PathLike,
    **kwargs,
) -> pd.DataFrame:
    try:
        validate_csv_delimiters(file_name)
        df = pd.read_csv(
            file_name,
            skipinitialspace=True,
            comment="#",
            engine="python",
            **kwargs,
        )

        return df
    except FileNotFoundError:
        raise ValueError(f"CSV file not found: {os.path.basename(file_name)}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {os.path.basename(file_name)}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {os.path.basename(file_name)}: {e}")


def read_exp_metadata(file_name: str) -> tuple[float, float, str]:
    """Reads metadata from an experiment CSV file.

    Metadata is stored as single comment lines beginning with ``#`` and formatted as
    ``name value``. Supported keys are ``temperature``, ``magnetic_field``, and
    ``isotope``.

    Args:
        file_name: Path to the experiment file.`

    Returns:
        A tuple ``(temperature, magnetic_field, isotope)`` where temperature is in K,
        magnetic field is in T, and isotope is formatted like ``"1H"`` or ``"13C"``.

    Raises:
        IndexError: If a required metadata line is missing.
        ValueError: If a numeric metadata value cannot be parsed.
    """

    temperature, magnetic_field, isotope = None, None, None

    temperature = float(
        find_first_group(file_name, r"# *temperature (\d*\.*\d*)", re.IGNORECASE)
    )

    magnetic_field = float(
        find_first_group(file_name, r"# *magnetic_field (\d*\.*\d*)", re.IGNORECASE)
    )

    isotope = str(
        find_first_group(file_name, r"# *isotope (\d{0,3}[A-Za-z]{0,2})", re.IGNORECASE)
    )

    return temperature, magnetic_field, isotope


def read_spectrum(file_name: str):
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

    return spectrum


def assemble_experiments_table(
    frames: "pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...]",
) -> pd.DataFrame:
    """Combine one or more experiment DataFrames into a single normalized table.

    This is a low-level helper used by higher-level IO adapters. It should not
    construct domain objects; it only concatenates and normalizes tabular data.

    Args:
        frames: A single DataFrame or a sequence of DataFrames representing
            experiment signal tables (optionally with metadata columns already
            attached).

    Returns:
        A single DataFrame with concatenated rows, sorted by temperature (then
        by shift if present) and with a clean, consecutive index.

    Raises:
        ValueError: If no frames are provided or if required columns are missing.
    """
    if frames is None:
        raise ValueError("No experiment data frames provided to assemble")

    # Accept a single DataFrame or a sequence.
    if isinstance(frames, pd.DataFrame):
        frames_list: list[pd.DataFrame] = [frames]
    else:
        frames_list = list(frames)

    if not frames_list:
        raise ValueError("No experiment data frames provided to assemble")

    data = pd.concat(frames_list, ignore_index=True)

    # Validate minimal required columns for downstream processing.
    required = {"temperature"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Experiment table is missing required column(s): {sorted(missing)}"
        )

    # Coerce temperature to numeric for stable sorting (errors become NaN).
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")

    sort_cols = ["temperature"]
    if "shift" in data.columns:
        sort_cols.append("shift")
        data["shift"] = pd.to_numeric(data["shift"], errors="coerce")

    data = data.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return data
