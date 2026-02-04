# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct ElectronicState domain objects.

Provides helpers to build ElectronicState instances from resolved quantum numbers.
"""

from __future__ import annotations

from typing import Optional

from simpnmr.core.domain.mol import ElectronicState


def build_electronic_state(
    *,
    spin_S: Optional[float] = None,
    orbit_L: Optional[float] = None,
    total_J: Optional[float] = None,
    model: Optional[str] = None,
) -> ElectronicState:
    """Construct a domain `ElectronicState` from resolved quantum numbers.

    Args:
        spin_S: Spin quantum number S.
        orbit_L: Orbital angular momentum quantum number L.
        total_J: Total angular momentum quantum number J.
        model: Optional explicit model selector.

    Returns:
        A domain `ElectronicState` instance.

    Raises:
        ValueError: If the electronic model cannot be resolved.
    """

    resolved_model = resolve_model(
        model=model,
        spin_S=spin_S,
        orbit_L=orbit_L,
        total_J=total_J,
    )

    return ElectronicState(
        spin_S=spin_S,
        orbit_L=orbit_L,
        total_J=total_J,
        model=resolved_model,
    )


def resolve_model(
    *,
    model: Optional[str],
    spin_S: Optional[float],
    orbit_L: Optional[float],
    total_J: Optional[float],
) -> str:
    """Resolve the electronic structure model.

    Priority (if `model` is not explicitly provided):
        1) total_J -> "total_J"
        2) orbit_L -> "orbital"
        3) spin_S  -> "spin_only"

    Returns:
        Resolved model string.

    Raises:
        ValueError: If no model can be resolved.
    """

    if model is not None:
        return model

    if total_J is not None:
        return "total_J"

    if orbit_L is not None:
        return "orbital"

    if spin_S is not None:
        return "spin_only"

    # Policy default: allow an empty ElectronicState and treat it as spin-only.
    return "spin_only"
