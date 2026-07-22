# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Tests for g-corrected isotropic susceptibility on the spin-Hamiltonian path.

The SH susceptibility is built from the real g-tensor, so its g-corrected
isotropic value (used for the Fermi contact) can be computed directly from chi
and g. These tests check that it is populated and is a rotation-invariant
scalar, and that it reduces to GE/g * iso in the isotropic-g limit.
"""

from __future__ import annotations

import pytest

from simpnmr.core.const.physics import GE
from simpnmr.core.phys.susc import build_susceptibility_from_sh


def _sh(alpha, beta, gamma, **kw):
    params = dict(
        gx=2.0, gy=2.1, gz=2.5, D_cmm1=5.0, E_cmm1=0.5, spin=2.0,
        temperatures=[298.0],
    )
    params.update(kw)
    return build_susceptibility_from_sh(
        alpha_deg=alpha, beta_deg=beta, gamma_deg=gamma, **params
    )[0]


def test_sh_populates_iso_g_corr():
    susc = _sh(0.0, 0.0, 0.0)
    assert susc.iso_g_corr is not None
    # g > GE (2.0023) here on average, so the g-corrected iso is below the raw iso
    assert susc.iso_g_corr < susc.iso


def test_sh_iso_g_corr_is_rotation_invariant():
    """iso_g_corr is a physical scalar and must not depend on the SH frame."""
    ref = _sh(0.0, 0.0, 0.0).iso_g_corr
    for angles in [(37.0, 58.0, 124.0), (90.0, 45.0, 10.0), (12.0, 170.0, 300.0)]:
        rotated = _sh(*angles).iso_g_corr
        assert rotated == pytest.approx(ref, rel=1e-9)


def test_sh_isotropic_g_limit():
    """For isotropic g, iso_g_corr == GE / g * iso exactly."""
    g = 2.1
    susc = _sh(31.0, 47.0, 63.0, gx=g, gy=g, gz=g)
    assert susc.iso_g_corr == pytest.approx(GE / g * susc.iso, rel=1e-12)


def test_sh_iso_is_rotation_invariant_baseline():
    """Sanity: the plain isotropic value is already frame-invariant."""
    ref = _sh(0.0, 0.0, 0.0).iso
    for angles in [(37.0, 58.0, 124.0), (90.0, 45.0, 10.0)]:
        assert _sh(*angles).iso == pytest.approx(ref, rel=1e-9)
