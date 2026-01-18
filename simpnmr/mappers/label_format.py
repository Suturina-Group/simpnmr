# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""
Utility functions for manipulating atom labels and formatted strings.

This submodule provides helpers for formatting text output and for
parsing, modifying, and expanding atom labels used in coordinate and
spectroscopic workflows.
"""

import re

from simpnmr.core.chemistry import periodic_table


def title(string: str) -> str:
    """
    Wrap a string with a title-style header consisting of horizontal
    separator lines above and below the string.

    Args:
        string (str): Input string to be wrapped with title separators.

    Returns:
        str: The formatted title string with surrounding separator lines.
    """

    titled = "\n"
    titled += "-" * (len(string) + 4)
    titled += "\n"
    titled += "- {} -\n".format(string)
    titled += "-" * (len(string) + 4)
    titled += "\n"

    return titled


def subtitle(string: str) -> str:
    """
    Wrap a string with a subtitle-style footer consisting of a horizontal
    separator line below the string.

    Args:
        string (str): Input string to be wrapped with a subtitle separator.

    Returns:
        str: The formatted subtitle string with a separator line beneath.
    """

    subtitled = "\n{}\n".format(string)
    subtitled += "-" * len(string)
    subtitled += "\n"

    return subtitled


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


def lab_adjust(label: str, shift: int) -> str:
    """
    Shift the numeric index of an atom label by a given integer offset.

    The label is expected to be in the format ATOMNUMBER (e.g. "Ca33").

    Args:
        label (str): Atom label to be modified.
        shift (int): Integer offset to apply to the numeric part of the label.

    Returns:
        str: New atom label with the shifted numeric index.
    """

    letters = remove_numbers(label)
    number = remove_letters(label)
    new_number = int(number) + shift

    new_lab = "{}{}".format(letters, new_number)

    return new_lab


def atom_range_to_list(atom_range: str) -> list[str]:
    """
    Expand an atom label range into a list of individual atom labels.

    For example, the range "H1-H10" is expanded into
    ["H1", "H2", ..., "H10"].

    Args:
        atom_range (str): Atom range string in the form "Xn-Xm".

    Returns:
        list[str]: List of expanded atom labels.

    Raises:
        ValueError: If the element symbol extracted from the range is not
            recognized.
    """

    # Remove hyphen, get number and generate all labels in range
    start = int(re.search(r"\d+", atom_range.split("-")[0]).group(0))
    end = int(re.search(r"\d+", atom_range.split("-")[1]).group(0))
    ele = re.search(r"[a-zA-Z\s]+", atom_range.split("-")[1]).group(0).capitalize()
    if ele not in periodic_table.elements:
        raise ValueError(f"Unknown nucleus type {ele}")
    else:
        atom_list = [f"{ele}{it}" for it in range(start, end + 1)]
    return atom_list
