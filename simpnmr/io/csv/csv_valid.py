# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Validate basic CSV formatting and delimiter consistency.

Provides lightweight checks to detect malformed CSV rows before parsing.
"""


def can_float(s: str) -> bool:
    """Returns whether a string can be parsed as a float."""
    out = True
    try:
        s = float(s.strip())
    except ValueError:
        out = False

    return out


def validate_csv_delimiters(file_name: str) -> None:
    """Scan a CSV file for likely missing commas.

    The function reads the file line by line (skipping comment lines starting with '#')
    and raises a ValueError if it finds a line that contains more than one numeric
    token but no commas, which is a common symptom of a missing delimiter.

    This does NOT try to fully parse the CSV; it only performs a light sanity check.
    """

    header_commas: int | None = None

    with open(file_name, "r", encoding="utf-8-sig", newline=None) as f:
        for i, line in enumerate(f, start=1):
            raw = line.rstrip("\r\n")
            text = raw.strip().lstrip("\ufeff")

            # Skip empty lines and comments
            if not text or text.startswith("#"):
                continue

            # First non-comment line is treated as header
            if header_commas is None:
                header_commas = text.count(",")
                continue

            parts = [p.strip() for p in text.split(",")]
            for part in parts:
                if not part:
                    continue

                tokens = part.split()
                numeric_tokens = [t for t in tokens if can_float(t)]

                # More than one numeric token in a single field is suspicious
                if len(numeric_tokens) > 1:
                    raise ValueError(
                        f"Possible missing comma on line {i}: '{raw}'. "
                        f"Detected multiple values within a single field ('{part}')."
                    )

            # If we have a header with commas, enforce delimiter consistency
            if header_commas > 0:
                # Count commas outside of quotes only (very light heuristic)
                in_quotes = False
                comma_count = 0
                for ch in text:
                    if ch == '"':
                        in_quotes = not in_quotes
                    elif ch == "," and not in_quotes:
                        comma_count += 1

                if comma_count != header_commas:
                    expected_cols = header_commas + 1
                    actual_cols = comma_count + 1

                    if comma_count < header_commas:
                        reason = "possible missing comma"
                    else:
                        reason = "possible extra comma"

                    raise ValueError(
                        f"CSV delimiter issue on line {i}: '{raw}'. "
                        f"Expected {expected_cols} columns, "
                        f"but found {actual_cols} columns ({reason})."
                    )
