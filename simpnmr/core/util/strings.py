# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define simple string manipulation helpers.

Provides utilities to remove numeric or alphabetic characters from strings.
"""


def remove_numbers(string: str) -> str:
    """
    Remove all numeric characters from a string.

    Args:
        string (str): Input string from which digits will be removed.

    Returns:
        str: The input string with all numeric characters removed.
    """

    no_digits = []
    for i in string:
        if not i.isdigit():
            no_digits.append(i)
        elif i.isdigit():
            continue
    result = "".join(no_digits)

    return result


def remove_letters(string: str) -> str:
    """
    Remove all non-numeric characters from a string.

    Args:
        string (str): Input string from which non-digit characters will be removed.

    Returns:
        str: A string containing only the numeric characters from the input.
    """

    no_letters = []
    for i in string:
        if i.isdigit():
            no_letters.append(i)
        elif not i.isdigit():
            continue
    result = "".join(no_letters)

    return result
