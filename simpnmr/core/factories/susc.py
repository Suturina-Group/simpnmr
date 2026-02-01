# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct magnetic susceptibility values and tensors.

Provides helpers to build susceptibility tensors and isotropic values from
quantum-chemical outputs and spin parameters.
"""

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.constants.physics import KB, MU0, MUB, NA
from simpnmr.core.factories.eff_factors import calc_g_eff, choose_S_eff


def susc_from_orca_xt(
    temperature: float,
    tensor_xt: NDArray,
) -> NDArray:
    """Convert an ORCA XT tensor to a susceptibility tensor in Å³.

    ORCA reports XT in units of cm^3 mol^-1 K. This factory converts XT to
    a molar susceptibility tensor X (Å^3) by applying the appropriate
    physical conversion and dividing by temperature.

    Args:
        temperature: Temperature in Kelvin.
        tensor_xt: ORCA XT tensor as a (3, 3) array in cm^3 mol^-1 K.

    Returns:
        Susceptibility tensor in Å^3.
    """

    # Conversion factor:
    # 1 cm^3 mol^-1 = 1e-6 m^3 / N_A
    # then convert m^3 -> Å^3 (1 m^3 = 1e30 Å^3)
    conv = 1e-24 * NA / (4.0 * np.pi)
    conv = 1.0 / conv

    chi_tensor = tensor_xt / temperature * conv
    return chi_tensor


def get_spin_only_susc(
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


def get_g_corr_iso_susc(
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
