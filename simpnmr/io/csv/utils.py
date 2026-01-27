# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

import os

import pandas as pd

from simpnmr.io.csv.validation import validate_csv_delimiters


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
