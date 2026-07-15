# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test: warn when the Curie prefactor collapses to zero.

Setting ``total_momentum_J: 0`` makes J(J+1) = 0, so the susceptibility and all
paramagnetic shifts are silently zero. The user must be warned rather than left
to wonder why every predicted shift is 0.
"""

import logging

import pytest

from simpnmr.core.fitting.vt import compute_chi_prefactor


@pytest.mark.unit
def test_zero_total_j_warns_and_returns_zero(caplog):
    with caplog.at_level(logging.WARNING):
        c = compute_chi_prefactor(1.0, total_J=0.0)
    assert c == 0.0
    assert any(
        "Curie prefactor is zero" in r.message and "total_momentum_J" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_none_total_j_uses_spin_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        c = compute_chi_prefactor(1.0, total_J=None)
    assert c > 0.0
    assert not caplog.records


@pytest.mark.unit
def test_spin_only_j_equals_s_is_nonzero(caplog):
    with caplog.at_level(logging.WARNING):
        c = compute_chi_prefactor(1.0, total_J=1.0)  # J = S for L = 0
    assert c > 0.0
    assert not caplog.records
