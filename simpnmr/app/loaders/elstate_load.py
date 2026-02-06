# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load an ElectronicState from explicit inputs and optional QC inference.

Constructs a domain ElectronicState and may infer spin S from a QC hyperfine file.
"""

from __future__ import annotations

import os
from typing import Optional

from simpnmr.core.build.elstate import build_electronic_state
from simpnmr.io.qc import gateway as rdrs


def load_electronic_state(
    *,
    spin_S: Optional[float] = None,
    orbit_L: Optional[float] = None,
    total_J: Optional[float] = None,
    model: Optional[str] = None,
    hyperfine_file: Optional[str] = None,
    hyperfine_method: Optional[str] = None,
):
    """Load/construct a domain `ElectronicState`.

    Args:
        spin_S: Spin quantum number S. If provided, no inference is attempted.
        orbit_L: Orbital angular momentum quantum number L.
        total_J: Total angular momentum quantum number J.
        model: Explicit model selector. If provided, it is used as-is.
        hyperfine_file: Optional path to a QC hyperfine output used to infer S.
        hyperfine_method: Optional method identifier (e.g. "dft"). Used to decide
            whether S inference should be attempted.

    Returns:
        A domain `ElectronicState` instance.
    """

    spin_guess = _infer_spin_from_hyperfine_file(
        spin_S=spin_S,
        hyperfine_file=hyperfine_file,
        hyperfine_method=hyperfine_method,
    )

    resolved_spin = spin_S if spin_S is not None else spin_guess

    return build_electronic_state(
        spin_S=resolved_spin,
        orbit_L=orbit_L,
        total_J=total_J,
        model=model,
    )


def _infer_spin_from_hyperfine_file(
    *,
    spin_S: Optional[float],
    hyperfine_file: Optional[str],
    hyperfine_method: Optional[str],
) -> Optional[float]:
    """Infer S from a QC hyperfine output file when possible.

    This performs IO and should only be called from the application layer.
    """

    if spin_S is not None:
        return None

    if not hyperfine_file:
        return None

    ext = os.path.splitext(hyperfine_file)[1].lower()

    should_try = (hyperfine_method == "dft") or (ext in (".log", ".out"))
    if not should_try:
        return None

    try:
        spin_obj = rdrs.QCSpin.guess_from_file(hyperfine_file)
    except SystemExit:
        return None

    return spin_obj.S
