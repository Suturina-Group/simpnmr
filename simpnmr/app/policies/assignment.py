# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Policy helpers for assignment search configuration.

This module resolves user-facing assignment search modes into concrete numeric
settings used by the Hungarian assignment workflow. It is intentionally kept at
the application-policy layer so that configuration parsing can remain focused
on schema validation and type coercion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssignmentSearchSettings:
    """Resolved search settings for assignment optimisation.

    Attributes:
        mode: Resolved search mode.
        n_attempts: Maximum number of restart attempts.
        max_iter: Maximum number of alternating-fit iterations per attempt.
        rmse_threshold: Early-stop threshold for RMSE (ppm). The search stops
            early if a converged attempt achieves RMSE below this value.
            Set to 0.0 to disable early stopping and always run all attempts.
    """

    mode: str
    n_attempts: int
    max_iter: int
    rmse_threshold: float


ASSIGNMENT_SEARCH_PRESETS: dict[str, AssignmentSearchSettings] = {
    "fast": AssignmentSearchSettings(
        mode="fast",
        n_attempts=1,
        max_iter=20,
        rmse_threshold=0.0,
    ),
    "balanced": AssignmentSearchSettings(
        mode="balanced",
        n_attempts=10,
        max_iter=100,
        rmse_threshold=0.0,
    ),
    "robust": AssignmentSearchSettings(
        mode="robust",
        n_attempts=25,
        max_iter=250,
        rmse_threshold=0.0,
    ),
}


DEFAULT_ASSIGNMENT_SEARCH_MODE = "balanced"
_ALLOWED_ASSIGNMENT_SEARCH_MODES = {
    "fast",
    "balanced",
    "robust",
    "custom",
}


def resolve_assignment_search_settings(
    mode: str | None,
    n_attempts: int | None,
    max_iter: int | None,
    rmse_threshold: float | None,
) -> AssignmentSearchSettings:
    """Resolve assignment search settings from raw policy inputs.

    Preset modes are expanded into concrete numeric parameters. The ``custom``
    mode preserves explicit caller-provided values and fills any missing values
    with the balanced defaults.

    Args:
        mode: Requested search mode.
        n_attempts: Optional explicit restart budget.
        max_iter: Optional explicit iteration budget per attempt.
        rmse_threshold: Optional explicit early-stop RMSE threshold (ppm).

    Returns:
        AssignmentSearchSettings: Resolved numeric search settings.

    Raises:
        ValueError: If the configured search mode is invalid.
    """
    raw_mode = mode
    mode = (mode or DEFAULT_ASSIGNMENT_SEARCH_MODE).strip().lower()

    if mode not in _ALLOWED_ASSIGNMENT_SEARCH_MODES:
        raise ValueError(
            "Invalid assignment search mode '"
            + str(raw_mode)
            + "'. Allowed values are: 'fast', 'balanced', 'robust', 'custom'."
        )

    if mode != "custom":
        return ASSIGNMENT_SEARCH_PRESETS[mode]

    balanced = ASSIGNMENT_SEARCH_PRESETS[DEFAULT_ASSIGNMENT_SEARCH_MODE]

    return AssignmentSearchSettings(
        mode="custom",
        n_attempts=balanced.n_attempts if n_attempts is None else n_attempts,
        max_iter=balanced.max_iter if max_iter is None else max_iter,
        rmse_threshold=(
            balanced.rmse_threshold
            if rmse_threshold is None
            else rmse_threshold
        ),
    )
