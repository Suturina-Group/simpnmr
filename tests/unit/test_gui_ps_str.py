# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test for the GUI picosecond formatter.

PyYAML parses ``140e-12`` (no decimal point) as the *string* ``"140e-12"``,
not a float. The relaxation-form loader divides such values by 1e-12, so a raw
``str / float`` raised "unsupported operand type(s) for /: 'str' and 'float'".
``_ps_str`` must coerce with ``float()`` first.
"""

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from simpnmr.gui.app import _ps_str


def test_ps_str_handles_yaml_string_scientific_notation():
    # the exact value that broke: str, no decimal point
    assert _ps_str("140e-12") == "140"
    assert _ps_str("0.2e-12") == "0.2"


def test_ps_str_handles_floats_and_blanks():
    assert _ps_str(140e-12) == "140"
    assert _ps_str(2e-13) == "0.2"
    assert _ps_str(None) == ""
    assert _ps_str("") == ""


def test_ps_str_non_numeric_is_empty():
    assert _ps_str("methanol") == ""
