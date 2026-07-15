# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Resolve linewidth values for prediction outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from simpnmr.core.domain.mol import Molecule

AUTO_LINEWIDTH_FRACTION = 0.005
LinewidthMode = Literal["auto", "relax"]


@dataclass(frozen=True)
class LinewidthOutput:
    """Resolved linewidth values for plots and peak CSV output.

    Args:
        mode: Linewidth resolution mode.
        column_name: CSV column name for the resolved linewidth values.
        values_by_label: Per-nucleus linewidth values in ppm.
    """

    mode: LinewidthMode
    column_name: str
    values_by_label: dict[str, float]


def resolve_output_linewidths(
    molecule: Molecule,
    shift_range: Sequence[float],
) -> LinewidthOutput:
    """Resolve linewidths for prediction plots and peak CSV output.

    Args:
        molecule: Molecule with shifts and optional relaxation-derived linewidths.
        shift_range: Two-value ppm range used for the predicted spectrum window.

    Returns:
        Resolved linewidth values and their CSV column contract.
    """
    relaxation = getattr(molecule, "relaxation", None)
    nuclei_with_lw = [nuc for nuc in molecule.nuclei if nuc.shift.lw is not None]

    if relaxation is not None:
        missing = [nuc.label for nuc in molecule.nuclei if nuc.shift.lw is None]
        if missing:
            raise ValueError("Relaxation linewidths are incomplete")
        return LinewidthOutput(
            mode="relax",
            column_name="linewidth_avg_relax (ppm)",
            values_by_label={nuc.label: nuc.shift.lw for nuc in molecule.nuclei},
        )

    # r6-fit linewidths written directly to nuc.shift.lw (no relaxation object).
    if nuclei_with_lw:
        auto_lw = _auto_display_linewidth_ppm(shift_range)
        return LinewidthOutput(
            mode="relax",
            column_name="linewidth_avg_relax (ppm)",
            values_by_label={
                nuc.label: nuc.shift.lw if nuc.shift.lw is not None else auto_lw
                for nuc in molecule.nuclei
            },
        )

    linewidth = _auto_display_linewidth_ppm(shift_range)
    return LinewidthOutput(
        mode="auto",
        column_name="linewidth_avg_auto (ppm)",
        values_by_label={nuc.label: linewidth for nuc in molecule.nuclei},
    )


def _auto_display_linewidth_ppm(shift_range: Sequence[float]) -> float:
    lo, hi = min(shift_range), max(shift_range)
    span = abs(hi - lo)
    if span == 0.0:
        # A single peak (or coincident peaks) has no range to scale from. Fall
        # back to the peak's own shift magnitude, with a 1 ppm floor, so the
        # cosmetic width is never zero — a zero FWHM makes the Lorentzian divide
        # by zero and the spectrum become NaN.
        span = max(abs(lo), abs(hi), 1.0)
    return AUTO_LINEWIDTH_FRACTION * span
