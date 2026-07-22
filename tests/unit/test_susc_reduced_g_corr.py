# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Tests for iso_g_corr on the reduced-chiT predict path.

The reduced ΔχT components fed to predict come from the fit_susc isoaxrh plot,
whose isotropic value is the g-corrected chi_iso. The builder must therefore
record the reconstructed iso as iso_g_corr so predict can split the Fermi contact
into spin-only and g-correction contributions.
"""

from __future__ import annotations

import pytest

from simpnmr.core.phys.susc import build_susceptibility_from_reduced_chi


def test_reduced_chi_populates_iso_g_corr_as_iso():
    suscs = build_susceptibility_from_reduced_chi(
        chi_iso_T=0.5,
        chi_ax_T=0.2,
        chi_rh_T=0.02,
        alpha_deg=17.0,
        beta_deg=42.0,
        gamma_deg=88.0,
        spin=2.0,
        temperatures=[250.0, 300.0],
    )
    assert len(suscs) == 2
    for susc in suscs:
        assert susc.iso_g_corr is not None
        # The reduced iso parameter IS the g-corrected chi_iso, so iso_g_corr
        # equals the reconstructed isotropic value exactly.
        assert susc.iso_g_corr == pytest.approx(susc.iso, rel=1e-12)
