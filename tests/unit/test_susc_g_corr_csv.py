# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Round-trip tests for the explicit g-corrected isotropic susceptibility.

A fit cannot recover the true spin-only chi_iso, so its fitted value is written
as chi_iso_g_corr with chi_iso left blank. Prediction reads chi_iso_g_corr back
(with precedence over chi_iso) as the canonical susc.iso, giving a g-corrected
Fermi contact.
"""

import os
import tempfile
import types

import numpy as np
import pytest

from simpnmr.app.loaders.susc_load import load_susceptibility_csv
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.io.csv.susc import read_susceptibilities_csv, save_susc


def _write(susc):
    path = os.path.join(tempfile.mkdtemp(), "susceptibility_tensor.csv")
    save_susc([types.SimpleNamespace(susc=susc)], path, verbose=False)
    return path


@pytest.mark.unit
def test_fit_writes_g_corr_and_blank_chi_iso():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_g_corr = 42.0  # fit records its fitted iso here
    _, _, chi_iso, chi_iso_g_corr = read_susceptibilities_csv(_write(s))[0]
    assert chi_iso is None
    assert chi_iso_g_corr == pytest.approx(42.0)


@pytest.mark.unit
def test_predict_reads_g_corr_as_canonical_iso():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_g_corr = 42.0
    path = _write(s)
    loaded = load_susceptibility_csv(path, electronic=None, g_tensor=None)
    # g-corrected value adopted as canonical iso -> drives the Fermi contact
    assert loaded[0].iso == pytest.approx(42.0)
    assert loaded[0].iso_g_corr == pytest.approx(42.0)


@pytest.mark.unit
def test_dft_predict_writes_both_columns():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_spin_only = 10.0
    s.iso_g_corr = 8.0
    _, _, chi_iso, chi_iso_g_corr = read_susceptibilities_csv(_write(s))[0]
    assert chi_iso == pytest.approx(10.0)
    assert chi_iso_g_corr == pytest.approx(8.0)


@pytest.mark.unit
def test_g_corr_takes_precedence_over_chi_iso():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_spin_only = 10.0
    s.iso_g_corr = 8.0
    loaded = load_susceptibility_csv(_write(s), electronic=None, g_tensor=None)
    assert loaded[0].iso == pytest.approx(8.0)  # g_corr wins
