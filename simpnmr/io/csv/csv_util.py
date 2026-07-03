# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Provide safe CSV I/O utilities.

Defines helpers to read and write CSV files with explicit UTF-8 encoding,
basic validation, and normalization suitable for cross-platform use
(including Windows and Excel).
"""

import datetime
import math
import os

import pandas as pd

from simpnmr.__version__ import __version__
from simpnmr.io.csv.csv_valid import validate_csv_delimiters


def format_full_precision(series: "pd.Series") -> "pd.Series":
    """Format a numeric column with full precision *and* tidy output.

    Each value is written as the **shortest** decimal string that still
    round-trips to the exact same float64 (Python's ``repr``). This keeps full
    precision — unlike a fixed ``%.Ng`` width, which either adds noise digits
    (``0.00089300000000000002``) or silently loses precision — while staying
    readable (``0.000893``, ``2e-07``).

    Use this on individual columns whose magnitude spans many orders (e.g.
    r⁻⁶ values) so they are unaffected by the table-wide ``float_format``
    passed to :func:`write_csv_safe`. Non-finite values and ``None`` become
    empty fields (matching pandas' default NaN handling).
    """
    def _fmt(v):
        if v is None:
            return ""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return v
        if not math.isfinite(fv):
            return ""
        return repr(fv)

    return series.map(_fmt)


def read_csv_safe(
    file_name: str | os.PathLike,
    **kwargs,
) -> pd.DataFrame:
    try:
        # Detect delimiter from first non-comment, non-empty line
        delimiter_is_comma = False
        with open(file_name, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "," in stripped:
                    delimiter_is_comma = True
                break

        if delimiter_is_comma:
            # Validate only comma-separated CSV files
            validate_csv_delimiters(file_name)
            df = pd.read_csv(
                file_name,
                sep=",",
                skipinitialspace=True,
                comment="#",
                encoding="utf-8-sig",
                **kwargs,
            )
        else:
            # Whitespace-separated file
            df = pd.read_csv(
                file_name,
                sep=r"\s+",
                engine="python",
                comment="#",
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


def write_csv_safe(
    df: pd.DataFrame,
    file_name: str | os.PathLike,
    comment: str | list[str] | None = None,
    *,
    sep: str = ",",
    # %.10g keeps full precision and prints readable decimals for normal-scale
    # columns (x/y/z, A_sd_*, shifts), switching to exponential only for very
    # small/large magnitudes (e.g. r⁻⁶ values) that %.6f would round to 0.
    float_format: str = "%.10g",
    index: bool = False,
    encoding: str = "utf-8-sig",
    newline: str = "",
    **kwargs,
) -> None:
    """Write DataFrame to CSV with explicit UTF-8 encoding (Excel-friendly by default).

    Symmetric to read_csv_safe. No delimiter sniffing or validation.

    Optionally, write comment lines (prefixed with '#') before the CSV header.
    """
    with open(file_name, "w", encoding=encoding, newline=newline) as fh:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        fh.write(
            f"# This file was generated with SimpNMR v{__version__} at {timestamp}\n"
        )
        if comment is not None:
            if isinstance(comment, str):
                fh.write(f"# {comment}\n")
            else:
                for line in comment:
                    fh.write(f"# {line}\n")
        df.to_csv(fh, sep=sep, index=index, float_format=float_format, **kwargs)
