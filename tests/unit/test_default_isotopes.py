# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression tests for default isotope choices.

Boron defaults to 11B: it is the standard NMR nucleus (~80% abundant, spin 3/2,
far more receptive than 10B), and the element gyromagnetic ratio in the table is
11B's value, so the default label and gamma must agree.
"""

import pytest

from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.isotopes import DEFAULT_ISOTOPES


@pytest.mark.unit
def test_default_boron_is_11B():
    assert DEFAULT_ISOTOPES["B"] == "11B"


@pytest.mark.unit
def test_default_boron_gamma_matches_11B_not_10B():
    gamma = get_nuclear_gamma(DEFAULT_ISOTOPES["B"])
    # 11B ~ 13.66 MHz/T; 10B ~ 4.58 MHz/T
    assert gamma == pytest.approx(13.66, abs=0.1)
    # default label resolves to the same gamma as the element fallback
    assert get_nuclear_gamma("11B") == get_nuclear_gamma("B")
