# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Provide safe CSV reading utilities.

Defines helpers to read CSV files with basic validation and normalization.
"""

import os

import pandas as pd

from simpnmr.io.csv.csv_valid import validate_csv_delimiters


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
            encoding="utf-8-sig",
            **kwargs,
        )

        # Normalize column names (strip BOM and surrounding whitespace)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

        return df
    except FileNotFoundError:
        raise ValueError(f"CSV file not found: {os.path.basename(file_name)}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {os.path.basename(file_name)}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {os.path.basename(file_name)}: {e}")
