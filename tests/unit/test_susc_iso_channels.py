# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Semantics of the three isotropic susceptibility channels.

``iso`` is the read-only true Tr(chi)/3. The Fermi contact is built from the
explicit spin-only and g-corrected channels (contact = spin-only + delta),
while the orbital contribution uses the true ``iso``.
"""

import numpy as np
import pytest

from simpnmr.core.domain.tensor import Hyperfine, Shift, Susceptibility


def _chi():
    # true iso = Tr(diag([2, 3, 5]))/3 = 10/3
    return Susceptibility(tensor=np.diag([2.0, 3.0, 5.0]), temperature=298.0)


@pytest.mark.unit
def test_iso_is_read_only_true_trace():
    chi = _chi()
    assert chi.iso == pytest.approx(10.0 / 3.0)
    with pytest.raises(AttributeError):
        chi.iso = 42.0


@pytest.mark.unit
def test_fermi_contact_is_spin_only_plus_delta():
    chi = _chi()
    chi.iso_spin_only = 6.0
    chi.iso_g_corr = 9.0
    A = Hyperfine(fc=np.eye(3) * 3.0)  # A_iso = 3
    assert Shift.calc_fc_spin_only(A, chi) == pytest.approx(18.0)  # 6 * 3
    assert Shift.calc_fc_delta_gcorr(A, chi) == pytest.approx(9.0)  # (9-6) * 3
    # total equals the sum, i.e. chi_iso_g_corr * A_iso
    assert Shift.calc_fcs(A, chi) == pytest.approx(27.0)
    assert Shift.calc_fcs_tensor(A, chi) == pytest.approx(9.0 * A.fc)


@pytest.mark.unit
def test_fermi_contact_falls_back_to_spin_only_when_no_gcorr():
    chi = _chi()
    chi.iso_spin_only = 6.0  # no iso_g_corr
    A = Hyperfine(fc=np.eye(3) * 3.0)
    assert Shift.calc_fc_delta_gcorr(A, chi) == pytest.approx(0.0)
    assert Shift.calc_fcs(A, chi) == pytest.approx(18.0)  # spin-only only


@pytest.mark.unit
def test_orbital_uses_true_iso_not_the_contact_channels():
    A = Hyperfine(fc=np.zeros((3, 3)), sd=np.eye(3) * 0.5, orb=np.eye(3) * 0.2)
    g = np.eye(3)

    base = _chi()
    base.iso_spin_only = 6.0
    base.iso_g_corr = 9.0
    orb_base = Shift.calc_orb_iso(A, base, g)

    # Change only the contact channels: the orbital iso must not move, since it
    # uses the true Tr(chi)/3 from the (unchanged) tensor.
    other = _chi()
    other.iso_spin_only = 100.0
    other.iso_g_corr = 250.0
    orb_other = Shift.calc_orb_iso(A, other, g)

    assert orb_base == pytest.approx(orb_other)
    assert orb_base != pytest.approx(0.0)
