# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Utility helpers for SimpNMR.

This module provides constants, parsing helpers, formatting utilities, and
relaxation-rate helper functions used across the package.
"""

import math

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
HBAR = consts.hbar  # [J s radian-1]
H = consts.h  # [J s radian-1]
KB = consts.physical_constants["Boltzmann constant"][0]  # Boltzmann constant k [J·K⁻¹]
GE = abs(consts.physical_constants["electron g factor"][0])  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]


def flatten(biglist: list) -> list:
    """Flattens a nested list by one level.

    Args:
        biglist: A list of lists.

    Returns:
        A single list containing the concatenated elements.
    """
    return [item for sublist in biglist for item in sublist]


def find_mean_values(values: list[float], thresh: float = 0.1) -> list[int]:
    """Finds indices where a 1D sequence changes by at least a threshold.

    Args:
        values: Values to analyze.
        thresh: Threshold for identifying a step change.

    Returns:
        Indices at which the step size ``abs(diff(values))`` is greater than or equal
        to `thresh`.
    """

    # Find values for which step size is >= thresh
    mask = np.abs(np.diff(values)) >= thresh
    # and mark indices at which to split
    split_indices = np.where(mask)[0] + 1

    return split_indices.tolist()


def comp2ind(comp_str: str) -> list[int]:
    """Converts a tensor component label into matrix indices.

    Args:
        comp_str: Component string, e.g. ``"xy"``.

    Returns:
        A tuple ``(row, col)`` for the corresponding element of a ``(3, 3)`` tensor.
    """

    _c2i = {
        "xx": [0, 0],
        "xy": [0, 1],
        "xz": [0, 2],
        "yx": [1, 0],
        "yy": [1, 1],
        "yz": [1, 2],
        "zx": [2, 0],
        "zy": [2, 1],
        "zz": [2, 2],
    }

    return _c2i[comp_str][0], _c2i[comp_str][1]


def find_index_of_nearest(array, value):
    """Returns the index of the nearest value in a sorted array."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def isotope_format(isotope_string: str) -> str:
    r"""Formats an isotope label as Matplotlib mathtext.

    Args:
        isotope_string: Isotope label, e.g. ``"1H"`` or ``"13C"``.

    Returns:
        A mathtext string, e.g. ``$^\mathregular{13} \mathregular{C}$``.
    """

    # Split at number letter boundary
    for it, char in enumerate(isotope_string):
        if char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            split_at = it
            break
    nums = isotope_string[:split_at]
    lets = isotope_string[split_at:]

    return r"$^\mathregular{{{}}} \mathregular{{{}}}$".format(nums, lets)


def calc_g_eff(spin: float, orbit: float, total_momentum_J: float | None):
    """Computes an effective electron g-factor.

    For spin-only systems (transition metals, organic radicals) where no total J is
    defined, this returns the free-electron g value `GE`.

    For systems with well-defined ``L``, ``S``, and ``J`` (e.g. lanthanides), this
    returns the Landé ``g_J`` factor.

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J``. If ``None`` or
            ``0``, the function falls back to `GE`.

    Returns:
        Effective g-factor (either `GE` or ``g_J``).
    """

    # Spin-only case: no total J provided or explicitly zero
    if total_momentum_J is None or total_momentum_J == 0.0:
        return GE

    # Landé g_J expression using S, L and J
    J = float(total_momentum_J)

    return 1.5 + (spin * (spin + 1) - orbit * (orbit + 1)) / (2.0 * J * (J + 1))


def choose_S_eff(spin: float, total_momentum_J: float | None):
    """Returns the effective angular momentum quantum number used in prefactors.

    Args:
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J``. If ``None``, `spin` is used.

    Returns:
        ``S`` for spin-only systems, or ``J`` when `total_momentum_J` is provided.
    """

    return spin if total_momentum_J is None else total_momentum_J


def get_spin_only_susceptibility(
    spin: float, orbit: float, total_momentum_J: float | None, temperature: float
) -> float:
    """Computes the spin-only isotropic molar susceptibility in Å³.

    Uses the Curie law with an effective g-factor and an effective angular momentum
    quantum number (``S`` for spin-only systems, ``J`` when defined).

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None`` for spin-only.
        temperature: Temperature in Kelvin.

    Returns:
        Spin-only isotropic molar susceptibility in Å³.
    """

    # Landé g-factor uses S, L, J
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Effective moment quantum number for Curie law:
    # S for transition metals, J for lanthanides
    S_eff = choose_S_eff(spin, total_momentum_J)

    # Chi (SI, m^3 mol^-1)
    chi_only_iso_SI = (
        MU0 * MUB**2 * g_eff**2 * S_eff * (S_eff + 1) / (3 * KB * temperature)
    )

    # Convert m^3 to Å^3: 1 Å^3 = 1e-30 m^3
    chi_only_iso = chi_only_iso_SI * 1e30

    return chi_only_iso


def get_true_iso_susceptibility(
    spin: float,
    orbit: float,
    g_tensor: NDArray,
    chi_tensors: dict[float, NDArray],
    total_momentum_J: float | None,
) -> float:
    """Computes a g-tensor-corrected isotropic susceptibility in Å³.

    Uses susceptibility principal components (from `chi_tensors`) and the supplied
    g-tensor to compute an effective isotropic value.

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        g_tensor: g-tensor as a ``(3, 3)`` array.
        chi_tensors: chi tensors in A^3.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Corrected isotropic susceptibility in A^3.

    """

    # Use Landé g_J (or GE) to get an effective g-factor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Trace-based expression with g correction (cm^3 mol^-1)
    chi_true_iso = g_eff / 3.0 * np.trace(chi_tensors * np.linalg.inv(g_tensor.T))

    return chi_true_iso
