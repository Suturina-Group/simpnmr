# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute effective electronic g-factors and angular momenta.

Provides helpers to evaluate effective g-values and angular momentum parameters
from spin, orbital, and total angular momentum quantum numbers.
"""

from simpnmr.core.constants.physics import GE


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
