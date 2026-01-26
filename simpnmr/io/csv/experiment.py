"""
CSV IO adapters for the Experiment domain model.

This module is responsible for translating CSV files into Experiment
objects and serializing Experiment objects back to CSV. It intentionally
contains IO logic and depends on low-level CSV readers.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from simpnmr.core.domain.experiment import Experiment, Signal
from simpnmr.io.csv.readers import (
    assemble_experiments_table,
    read_csv_safe,
    read_exp_metadata,
)
from simpnmr.mappers import dataframes as ser

logger = logging.getLogger(__name__)


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

        temperature, magnetic_field, isotope = read_exp_metadata(file_name)
        df["temperature"] = temperature
        df["magnetic_field"] = magnetic_field
        df["isotope"] = isotope

        frames.append(df)

    table = assemble_experiments_table(frames)

    # Build one Experiment per (temperature, magnetic_field, isotope) block.
    def _pick_col(df_, candidates: list[str]) -> str:
        for c in candidates:
            if c in df_.columns:
                return c
        raise KeyError(f"Missing required column. Expected one of: {candidates}")

    shift_col = _pick_col(
        table, ["shift", "shifts", "ppm", "shift (ppm)", "shifts (ppm)"]
    )
    width_col = _pick_col(table, ["width", "widths", "width (Hz)", "width(Hz)"])
    area_col = _pick_col(table, ["area", "areas", "integral", "integrals"])
    assignment_col = _pick_col(table, ["assignment", "assignments"])

    l_to_g_col = next((c for c in ["L/G", "L/G ()"] if c in table.columns), None)
    r1_col = next(
        (
            c
            for c in ["R1", "r1", "R1 (s^-1)", "r1 (s^-1)", "1/T1", "1/T1 (s^-1)"]
            if c in table.columns
        ),
        None,
    )

    for (temperature, magnetic_field, isotope), group in table.groupby(
        ["temperature", "magnetic_field", "isotope"]
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
                str(isotope),
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
    df = ser.build_experiment_signals_df(experiment)

    with open(file_name, "w", encoding="utf-8") as fh:
        if comment:
            fh.write(f"# {comment}\n")

        df.to_csv(fh, sep=delimiter, index=False)

    if verbose:
        logger.info("Experiment written to %s", file_name)
