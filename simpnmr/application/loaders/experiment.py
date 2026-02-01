# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load and save Experiment objects via CSV.

Reads external data and constructs domain Experiment instances, or serializes
them back to CSV.
"""

from __future__ import annotations

from typing import Iterable

from simpnmr.core.domain.experiment import Experiment


def load_experiments(file_names: str | Iterable[str]) -> list[Experiment]:
    """
    Load Experiment objects from one or more CSV files.

    This function is the application-level entry point for experiment IO.
    It delegates CSV parsing to the IO layer and returns domain objects.

    Args:
        file_names: Path or iterable of paths to CSV files.

    Returns:
        List of Experiment objects.
    """
    from simpnmr.io.csv.experiment import load_experiments_from_csv

    return load_experiments_from_csv(file_names)


def save_experiment(
    experiment: Experiment,
    file_name: str,
    *,
    delimiter: str = ",",
    comment: str | None = None,
) -> None:
    """
    Save a single Experiment object to CSV.

    Args:
        experiment: Experiment instance to save.
        file_name: Output CSV path.
        delimiter: CSV delimiter.
        comment: Optional comment written to the file header.
    """
    from simpnmr.io.csv.experiment import write_experiment_to_csv

    write_experiment_to_csv(
        experiment,
        file_name=file_name,
        delimiter=delimiter,
        comment=comment,
    )
