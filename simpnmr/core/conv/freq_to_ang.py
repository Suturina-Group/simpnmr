# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert hyperfine couplings from MHz to ppm Å⁻³.

Provides helpers to convert isotropic and tensor hyperfine values using
nuclear gyromagnetic ratios.
"""

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.gammas import NUCLEAR_GAMMAS, get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA, MU0, H


def a_tensor_mhz_to_ang(a_tensors: dict[str, NDArray]) -> dict[str, NDArray]:
    """Converts hyperfine A tensors from MHz to ``ppm Å^-3``.

    Uses the gyromagnetic ratio of each nucleus (looked up from `NUCLEAR_GAMMAS`)

    Args:
        a_tensors: Mapping from atom label (with global index, e.g. ``"H34"``) to a
            ``(3, 3)`` hyperfine tensor in MHz.

    Returns:
        Mapping from atom label to hyperfine tensor in ``ppm Å^-3``. Labels whose
        element has no gamma defined (gamma=0) are omitted.
    """

    def _gamma_or_zero(k):
        try:
            return get_nuclear_gamma(k)
        except KeyError:
            return 0.0

    a_tensors_ang = {
        key: _mhz_to_angstrom(val, _gamma_or_zero(key))
        for key, val in a_tensors.items()
        if _gamma_or_zero(key) != 0
    }

    return a_tensors_ang


def _mhz_to_angstrom(val_mhz: NDArray | float, nuclear_gamma: float) -> NDArray | float:
    """Converts a hyperfine coupling value from MHz to ``ppm Å^-3``.

    Args:
        val_mhz: Hyperfine tensor as a ``(3, 3)`` array or an isotropic value in MHz.
        nuclear_gamma: Nuclear gyromagnetic ratio for the nucleus (MHz/T).

    Returns:
        The converted value in ``ppm Å^-3`` with the same shape as `val_mhz`.
    """

    val_mhz = np.asarray(val_mhz)

    # Conversion factor for MHz to ppm Angstrom^-3
    val = 1e-18 / (H * EGAMMA * nuclear_gamma * 1e12 * MU0)

    val_ang = val_mhz * val

    return val_ang
