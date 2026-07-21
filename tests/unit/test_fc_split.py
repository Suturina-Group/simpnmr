# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""The Fermi contact is split into spin-only and g-correction contributions.

The spin-only reference is computed directly from S and the temperature, so the
split is available even for fit-derived g-corrected susceptibilities that carry
no stored spin-only channel. The two contributions sum to the total FC.
"""

import numpy as np
import pytest

from simpnmr.core.domain.mol import ElectronicState, Molecule, Nucleus
from simpnmr.core.domain.tensor import Hyperfine, Shift, Susceptibility
from simpnmr.core.phys.susc import get_spin_only_susc


def _molecule_with_g_corr_only():
    # susc as if loaded from a fit CSV: only the g-corrected iso is known
    s = Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)
    s.iso_g_corr = s.iso
    assert s.iso_spin_only is None
    nuc = Nucleus(
        label="C1",
        coord=[0.0, 0.0, 0.0],
        A=Hyperfine(fc=np.eye(3) * 100.0),
        shift=Shift(),
    )
    m = Molecule(labels=["C"], coords=np.zeros((1, 3)), nuclei=[nuc])
    m.susc = s
    m.electronic = ElectronicState(spin_S=1.0)
    return m, nuc


@pytest.mark.unit
def test_fc_spin_only_computed_from_spin_and_temperature():
    m, nuc = _molecule_with_g_corr_only()
    m.calculate_shifts()
    expected = get_spin_only_susc(
        spin=1.0, orbit=0.0, total_momentum_J=None, temperature=298.0
    ) * 100.0  # A_iso = 100
    assert nuc.shift.fc_spin_only == pytest.approx(expected)


@pytest.mark.unit
def test_fc_contributions_sum_to_total():
    m, nuc = _molecule_with_g_corr_only()
    m.calculate_shifts()
    assert nuc.shift.fc_spin_only + nuc.shift.fc_delta_g_corr == pytest.approx(
        nuc.shift.fc
    )


@pytest.mark.unit
def test_split_available_without_stored_spin_only():
    # the pre-change code skipped when susc.iso_spin_only was None
    m, nuc = _molecule_with_g_corr_only()
    m.calculate_shifts()
    assert m.susc.iso_spin_only is None
    assert nuc.shift.fc_spin_only is not None
