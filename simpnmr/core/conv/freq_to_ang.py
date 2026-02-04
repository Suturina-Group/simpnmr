# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert hyperfine couplings from MHz to ppm Å⁻³.

Provides helpers to convert isotropic and tensor hyperfine values using
nuclear gyromagnetic ratios.
"""

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr.core.const.physics import EGAMMA, MU0, H
from simpnmr.core.util.strings import remove_numbers


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

    a_tensors_ang = {
        key: _mhz_to_angstrom(val, NUCLEAR_GAMMAS[remove_numbers(key)])
        for key, val in a_tensors.items()
        if remove_numbers(key) in NUCLEAR_GAMMAS.keys()
        and NUCLEAR_GAMMAS[remove_numbers(key)]
    }

    return a_tensors_ang


def a_iso_mhz_to_ang(a_iso: dict[str, float]) -> dict[str, float]:
    """Convert isotropic hyperfine A values from MHz to ppm Å^-3.

    Uses the nuclear gyromagnetic ratio for each nucleus (from NUCLEAR_GAMMAS).

    Args:
        a_iso: Mapping from atom label (with global index, e.g. "H34")
            to isotropic hyperfine value in MHz.

    Returns:
        Mapping from atom label to isotropic hyperfine value in ppm Å^-3.
        Labels whose element has no gamma defined (gamma=0) are omitted.
    """
    a_iso_ang: dict[str, float] = {}

    for key, val in a_iso.items():
        elem = remove_numbers(key)

        if elem not in NUCLEAR_GAMMAS:
            continue

        gamma = NUCLEAR_GAMMAS[elem]
        if not gamma:
            continue

        # _mhz_to_angstrom returns ndarray or float → force float
        a_iso_ang[key] = float(_mhz_to_angstrom(val, gamma))

    return a_iso_ang


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
