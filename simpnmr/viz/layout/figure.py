# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Canonical figure-size tokens for visualization profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from simpnmr.app.params.plot_cfg import PlotProfile

FigureVariant = Literal["standard", "narrow", "vertical", "vertical_extended"]


@dataclass(frozen=True, slots=True)
class FigureSize:
    """Canonical figure size in inches."""

    width: float
    height: float

    def as_tuple(self) -> tuple[float, float]:
        """Return the figure size as a Matplotlib ``figsize`` tuple."""
        return (self.width, self.height)


_FIGURE_SIZES: dict[PlotProfile, dict[FigureVariant, FigureSize]] = {
    "paper": {
        # ACS single-column: 3.25 in (8.26 cm) wide
        "standard": FigureSize(width=3.25, height=2.20),
        "narrow": FigureSize(width=1.77, height=1.48),
        "vertical": FigureSize(width=3.25, height=3.10),
        "vertical_extended": FigureSize(width=3.25, height=3.50),
    },
    "poster": {
        "standard": FigureSize(width=3.54, height=2.40),
        "narrow": FigureSize(width=2.48, height=2.40),
        "vertical": FigureSize(width=3.54, height=4.05),
        "vertical_extended": FigureSize(width=3.54, height=4.33),
    },
}


def get_figsize(
    profile: PlotProfile,
    variant: FigureVariant = "standard",
) -> tuple[float, float]:
    """Return the canonical ``figsize`` tuple for a profile and variant.

    Args:
        profile: Plotting profile that selects the publication context.
        variant: Figure geometry variant within the selected profile.

    Returns:
        Canonical Matplotlib ``figsize`` tuple in inches.

    Raises:
        ValueError: If the profile/variant combination is not supported.
    """
    try:
        return _FIGURE_SIZES[profile][variant].as_tuple()
    except KeyError as exc:
        raise ValueError(
            "Unsupported figsize combination: "
            f"profile={profile!r}, variant={variant!r}."
        ) from exc
