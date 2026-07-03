# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load and save Experiment objects via CSV.

Reads external data and constructs domain Experiment instances, or serializes
them back to CSV.
"""

from __future__ import annotations

from typing import Iterable

from simpnmr.core.domain.exp import Experiment


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
    from simpnmr.io.csv.exp import load_experiments_from_csv

    return load_experiments_from_csv(file_names)


def save_experiment(
    experiment: Experiment,
    file_name: str,
    *,
    delimiter: str = ",",
    comment: str | None = None,
) -> None:
    """Save a single Experiment object to a wide-format CSV file."""
    from simpnmr.io.csv.exp import write_experiment_to_csv

    write_experiment_to_csv(
        experiment,
        file_name=file_name,
        delimiter=delimiter,
    )


def save_experiments(
    experiments: list[Experiment],
    file_name: str,
    *,
    delimiter: str = ",",
) -> None:
    """Save multiple Experiment objects to a single wide-format CSV file."""
    from simpnmr.io.csv.exp import write_experiments_to_csv

    write_experiments_to_csv(experiments, file_name, delimiter=delimiter)
