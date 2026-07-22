# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Round-trip tests for the explicit isotropic susceptibility channels.

The susceptibility carries three isotropic channels: the true Tr(chi)/3
(``iso``, read-only, derived from the tensor), the spin-only reference
(``iso_spin_only``), and the g-corrected contact value (``iso_g_corr``). Each is
written to its own CSV column when known, and the spin-only / g-corrected
channels are read back directly; the g-corrected channel drives the Fermi
contact.
"""

import os
import tempfile
import types

import numpy as np
import pytest

from simpnmr.app.loaders.susc_load import load_susceptibility_csv
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.io.csv.susc import read_susceptibilities_csv, save_susc

# Tr(diag([2, 3, 5]))/3
_TRUE_ISO = 10.0 / 3.0


def _write(susc):
    path = os.path.join(tempfile.mkdtemp(), "susceptibility_tensor.csv")
    save_susc([types.SimpleNamespace(susc=susc)], path, verbose=False)
    return path


@pytest.mark.unit
def test_fit_writes_g_corr_and_true_chi_iso():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_g_corr = 42.0  # fit records its fitted iso here
    _, _, chi_iso, chi_iso_spin_only, chi_iso_g_corr = read_susceptibilities_csv(
        _write(s)
    )[0]
    # chi_iso is the true Tr(chi)/3 from the tensor; spin-only is unknown here
    assert chi_iso == pytest.approx(_TRUE_ISO)
    assert chi_iso_spin_only is None
    assert chi_iso_g_corr == pytest.approx(42.0)


@pytest.mark.unit
def test_predict_reads_g_corr_channel():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_g_corr = 42.0
    path = _write(s)
    loaded = load_susceptibility_csv(path, electronic=None, g_tensor=None)
    # g-corrected value drives the Fermi contact; the true iso stays Tr(chi)/3
    assert loaded[0].iso_g_corr == pytest.approx(42.0)
    assert loaded[0].iso == pytest.approx(_TRUE_ISO)


@pytest.mark.unit
def test_writes_all_three_iso_columns():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_spin_only = 10.0
    s.iso_g_corr = 8.0
    _, _, chi_iso, chi_iso_spin_only, chi_iso_g_corr = read_susceptibilities_csv(
        _write(s)
    )[0]
    assert chi_iso == pytest.approx(_TRUE_ISO)  # true
    assert chi_iso_spin_only == pytest.approx(10.0)
    assert chi_iso_g_corr == pytest.approx(8.0)


@pytest.mark.unit
def test_load_populates_spin_only_and_g_corr_channels():
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_spin_only = 10.0
    s.iso_g_corr = 8.0
    loaded = load_susceptibility_csv(_write(s), electronic=None, g_tensor=None)
    assert loaded[0].iso_g_corr == pytest.approx(8.0)  # drives Fermi contact
    assert loaded[0].iso_spin_only == pytest.approx(10.0)
    assert loaded[0].iso == pytest.approx(_TRUE_ISO)  # true, from tensor
