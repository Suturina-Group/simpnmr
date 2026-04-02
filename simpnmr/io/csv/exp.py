# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write Experiment objects as CSV.

Provides CSV parsing and serialization helpers for Experiment and Signal data.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List

import pandas as pd

from simpnmr.core.domain.exp import Experiment, Signal
from simpnmr.io.csv.csv_util import read_csv_safe, write_csv_safe
from simpnmr.io.text.parse import find_first_group

logger = logging.getLogger(__name__)


def read_exp_metadata(file_name: str) -> tuple[float, float]:
    """Reads metadata from an experiment CSV file.

    Metadata is stored as single comment lines beginning with ``#`` and
    formatted as ``name value``. Supported keys are ``temperature`` and
    ``magnetic_field``.

    Args:
        file_name: Path to the experiment file.

    Returns:
        A tuple ``(temperature, magnetic_field)`` where temperature is in K
        and magnetic field is in T.

    Raises:
        IndexError: If a required metadata line is missing.
        ValueError: If a numeric metadata value cannot be parsed.
    """

    temperature = float(
        find_first_group(file_name, r"# *temperature (\d*\.*\d*)", re.IGNORECASE)
    )

    magnetic_field = float(
        find_first_group(file_name, r"# *magnetic_field (\d*\.*\d*)", re.IGNORECASE)
    )

    return temperature, magnetic_field


def assemble_experiments_table(frames):
    """Combine multiple experiment DataFrames into a single normalized table."""

    if not frames:
        raise ValueError("No experiment data frames provided to assemble")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("temperature")
    return data


def load_experiments_from_csv(
    file_names: str | Iterable[str],
) -> List[Experiment]:
    """
    Load one or more Experiment objects from CSV file(s).

    Args:
        file_names: Path or iterable of paths to CSV experiment files.

    Returns:
        List of Experiment objects.
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    experiments: list[Experiment] = []
    frames = []

    for file_name in file_names:
        df = read_csv_safe(file_name)
        df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")

        temperature, magnetic_field = read_exp_metadata(file_name)
        df["temperature"] = temperature
        df["magnetic_field"] = magnetic_field

        frames.append(df)

    table = assemble_experiments_table(frames)

    # Build one Experiment per (temperature, magnetic_field) block.
    def _pick_col(df_, candidates: list[str]) -> str:
        """Pick a required column from a dataframe using tolerant header matching."""
        # Map normalized column names -> actual column names.
        norm_to_actual = {str(c).strip().lower(): c for c in df_.columns}

        for candidate in candidates:
            norm = str(candidate).strip().lower()
            if norm in norm_to_actual:
                return norm_to_actual[norm]

        raise KeyError(
            f"Missing required column. Expected one of: {candidates}. "
            f"Found columns: {list(df_.columns)}"
        )

    shift_col = _pick_col(
        table, ["shift", "shifts", "ppm", "shift (ppm)", "shifts (ppm)"]
    )
    width_col = _pick_col(table, ["width", "widths", "width (Hz)", "width(Hz)"])
    area_col = _pick_col(
        table,
        ["area", "areas", "area ()", "area()", "integral", "integrals"],
    )
    assignment_col = _pick_col(table, ["assignment", "assignments"])

    l_to_g_col = next((c for c in ["L/G", "L/G ()"] if c in table.columns), None)
    r1_col = next(
        (
            c
            for c in [
                "R1", "r1",
                "R1 (s^-1)", "r1 (s^-1)",
                "R1 (Hz)", "r1 (Hz)",
                "1/T1", "1/T1 (s^-1)", "1/T1 (Hz)",
            ]
            if c in table.columns
        ),
        None,
    )

    for (temperature, magnetic_field), group in table.groupby(
        ["temperature", "magnetic_field"]
    ):
        signals: list[Signal] = []
        for _, row in group.iterrows():
            l_to_g = float(row[l_to_g_col]) if l_to_g_col is not None else 1.0
            r1 = (
                float(row[r1_col])
                if r1_col is not None and row[r1_col] == row[r1_col]
                else None
            )

            signals.append(
                Signal(
                    float(row[shift_col]),
                    float(row[width_col]),
                    float(row[area_col]),
                    str(row[assignment_col]),
                    l_to_g=l_to_g,
                    r1=r1,
                )
            )

        experiments.append(
            Experiment(
                float(temperature),
                float(magnetic_field),
                signals,
            )
        )

    return experiments


def write_experiment_to_csv(
    experiment: Experiment,
    file_name: str,
    *,
    delimiter: str = ",",
    comment: str = "",
    verbose: bool = True,
) -> None:
    """
    Write an Experiment object to a CSV file.

    Args:
        experiment: Experiment instance to serialize.
        file_name: Output CSV file path.
        delimiter: CSV delimiter.
        comment: Optional comment to prepend to the file.
        verbose: Whether to print status messages.
    """
    df = _build_experiment_signals_df(experiment)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Experiment written to %s", file_name)


def _build_experiment_signals_df(experiment: "Experiment") -> pd.DataFrame:
    columns = ["assignment ()", "shift (ppm)", "width (Hz)", "area ()", "L/G ()"]

    data = {
        "assignment ()": [s.assignment for s in experiment.signals],
        "shift (ppm)": [s.shift for s in experiment.signals],
        "width (Hz)": [s.width for s in experiment.signals],
        "area ()": [s.area for s in experiment.signals],
        "L/G ()": [s.l_to_g for s in experiment.signals],
    }

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.sort_values("shift (ppm)").reset_index(drop=True)
