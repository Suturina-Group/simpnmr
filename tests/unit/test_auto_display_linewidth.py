# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test for the cosmetic display-linewidth fallback.

A spin-only prediction with a single peak has a zero-width shift range. The
auto display linewidth must not be zero: a zero FWHM makes the Lorentzian kernel
divide by zero, producing a NaN spectrum ("Axis limits cannot be NaN or Inf").
"""

import pytest

from simpnmr.app.policies.linewidth import (
    AUTO_LINEWIDTH_FRACTION,
    _auto_display_linewidth_ppm,
)


@pytest.mark.unit
def test_single_peak_range_gives_nonzero_width():
    # degenerate range (one peak): must be strictly positive
    assert _auto_display_linewidth_ppm([-147.7, -147.7]) > 0.0
    assert _auto_display_linewidth_ppm([0.0, 0.0]) > 0.0
    # a paramagnetic single peak scales off its own magnitude
    assert _auto_display_linewidth_ppm([-147.7, -147.7]) == pytest.approx(
        AUTO_LINEWIDTH_FRACTION * 147.7
    )


@pytest.mark.unit
def test_normal_range_unchanged():
    assert _auto_display_linewidth_ppm([-150.0, 150.0]) == pytest.approx(
        AUTO_LINEWIDTH_FRACTION * 300.0
    )
