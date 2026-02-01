# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse simple patterns from text files.

Provides helpers to extract regex capture groups from line-oriented text files.
"""

import re
from os import PathLike


def find_first_group(
    file_name: str | PathLike[str],
    pattern: str,
    flags: int = 0,
) -> str:
    """Return the first captured group for the first regex match in a text file

    The function scans the file line by line and applies `pattern` using `re.search`
    It returns group 1 from the first match, so `pattern` must contain at least one
    capturing group in parentheses

    Args:
        file_name: Path to the text file to scan
        pattern: Regular expression pattern with at least one capturing group
        flags: Regex flags passed to `re.compile`, e.g. `re.IGNORECASE`

    Returns:
        The first captured group from the first matching line

    Raises:
        ValueError: If no matching line is found
        IndexError: If the pattern matches but has no capturing group 1
    """
    rx = re.compile(pattern, flags)

    with open(file_name, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                return m.group(1)

    raise ValueError(f"No relevant data found in {file_name} for pattern: {pattern}")
