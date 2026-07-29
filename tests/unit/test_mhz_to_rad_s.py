# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Hyperfine coupling MHz -> angular frequency (rad/s) conversion.

ORCA prints the HFC as A/h in MHz (linear frequency); the SBM/Abragam contact
relaxation rates use A/hbar = 2*pi*nu (angular frequency). The shared helper
must apply exactly 2*pi*1e6.
"""

import numpy as np
import pytest

from simpnmr.core.conv.ang_to_freq import mhz_to_rad_s


@pytest.mark.unit
def test_mhz_to_rad_s_scalar():
    # 1 MHz -> 2*pi*1e6 rad/s
    assert float(mhz_to_rad_s(1.0)) == pytest.approx(2 * np.pi * 1e6, rel=1e-12)
    assert float(mhz_to_rad_s(12.5)) == pytest.approx(
        12.5 * 1e6 * 2 * np.pi, rel=1e-12
    )


@pytest.mark.unit
def test_mhz_to_rad_s_is_two_pi_times_linear_hz():
    # rad/s must be 2*pi larger than the linear-frequency (Hz) form
    nu_mhz = 7.3
    hz = nu_mhz * 1e6
    assert float(mhz_to_rad_s(nu_mhz)) == pytest.approx(2 * np.pi * hz, rel=1e-12)


@pytest.mark.unit
def test_mhz_to_rad_s_array():
    vals = np.array([0.0, 1.0, -3.0])
    out = mhz_to_rad_s(vals)
    assert np.allclose(out, vals * 1e6 * 2 * np.pi)
