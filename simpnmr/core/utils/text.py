# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define simple text formatting helpers.

Provides utilities to generate titled and subtitled text blocks.
"""


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
