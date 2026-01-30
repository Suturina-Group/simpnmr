# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""HFC Convertors for SimpNMR.
TODO
This module ...
"""

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray

from simpnmr.core.constants.gammas import NUCLEAR_GAMMAS
from simpnmr.mappers import label_format as lf

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
HBAR = consts.hbar  # [J s radian-1]
H = consts.h  # [J s radian-1]
KB = consts.physical_constants["Boltzmann constant"][0]  # Boltzmann constant k [J·K⁻¹]
GE = abs(consts.physical_constants["electron g factor"][0])  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]


def a_tensor_mhz_to_angst(a_tensors: dict[str, NDArray]) -> dict[str, NDArray]:
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
        key: _mhz_to_angstrom(val, NUCLEAR_GAMMAS[lf.remove_numbers(key)])
        for key, val in a_tensors.items()
        if lf.remove_numbers(key) in NUCLEAR_GAMMAS.keys()
        and NUCLEAR_GAMMAS[lf.remove_numbers(key)]
    }

    return a_tensors_ang


def a_iso_mhz_to_angst(a_iso: dict[str, float]) -> dict[str, float]:
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
        elem = lf.remove_numbers(key)

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
