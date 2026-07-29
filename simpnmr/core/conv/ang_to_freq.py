# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert hyperfine couplings from ppm Å^-3 to MHz.

Provides helpers to convert hyperfine values using nuclear gyromagnetic ratios.
"""

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.physics import EGAMMA, MU0, H


def angstrom_to_mhz(
    val_ang: NDArray | float,
    nuclear_gamma: float,
) -> NDArray | float:
    """Converts a hyperfine coupling value from ``ppm Å^-3`` to MHz.

    Args:
        val_ang: Hyperfine tensor as a ``(3, 3)`` array or an isotropic value
            in ``ppm Å^-3``.
        nuclear_gamma: Nuclear gyromagnetic ratio for the nucleus (MHz/T).

    Returns:
        The converted value in MHz with the same shape as `val_ang`.
    """
    val_ang = np.asarray(val_ang)

    # Inverse conversion factor: ppm Å^-3 → MHz
    inv = (H * EGAMMA * nuclear_gamma * 1e12 * MU0) / 1e-18

    val_mhz = val_ang * inv
    return val_mhz


def mhz_to_rad_s(val_mhz: NDArray | float) -> NDArray | float:
    """Converts a frequency from MHz to angular frequency (rad/s).

    ORCA prints the hyperfine coupling as ``A/h`` in MHz (a linear frequency),
    while the SBM/Abragam relaxation rates use ``A/hbar = 2*pi*nu`` (angular
    frequency). This applies ``omega = 2*pi*nu`` with ``nu`` in MHz, i.e.
    ``rad/s = 2*pi * 1e6 * MHz``.

    Args:
        val_mhz: Value in MHz (scalar or array).

    Returns:
        The value in rad/s, with the same shape as `val_mhz`.
    """
    return np.asarray(val_mhz) * 1e6 * 2 * np.pi
