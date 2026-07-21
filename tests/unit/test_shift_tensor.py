# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression tests for the full 3x3 paramagnetic shift tensors.

Each contribution's tensor is stored raw (non-symmetric); one third of its
trace must recover the corresponding scalar shift, and the Shift.paramag_tensor
must equal fc + pc + orb.
"""

import numpy as np
import pytest

from simpnmr.core.domain.tensor import Hyperfine, Shift, Susceptibility


@pytest.fixture
def A_chi():
    A = Hyperfine(
        fc=np.eye(3) * 5.0,
        sd=np.array([[1.0, 0.2, 0.1], [0.3, -0.5, 0.4], [0.0, 0.1, -0.5]]),
    )
    chi = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    return A, chi


def test_tensor_trace_matches_scalar(A_chi):
    A, chi = A_chi
    assert np.isclose(
        np.trace(Shift.calc_pcs_tensor(A, chi)) / 3.0, Shift.calc_pcs(A, chi)
    )
    assert np.isclose(
        np.trace(Shift.calc_fcs_tensor(A, chi)) / 3.0, Shift.calc_fcs(A, chi)
    )


def test_pcs_tensor_is_raw_non_symmetric(A_chi):
    A, chi = A_chi
    t = Shift.calc_pcs_tensor(A, chi)
    assert t.shape == (3, 3)
    assert not np.allclose(t, t.T)  # stored raw, never symmetrised


def test_paramag_tensor_is_sum_of_components():
    s = Shift()
    s.fc_tensor = np.full((3, 3), 1.0)
    s.pc_tensor = np.full((3, 3), 2.0)
    s.orb_tensor = np.full((3, 3), 3.0)
    assert np.allclose(s.paramag_tensor, np.full((3, 3), 6.0))


def test_default_shift_tensors_are_zero():
    s = Shift()
    assert np.allclose(s.paramag_tensor, np.zeros((3, 3)))
