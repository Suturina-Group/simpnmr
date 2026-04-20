# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Susceptibility policy helpers.

This module centralizes application-level decisions for loading susceptibility
sources. In particular, it provides a single source of truth for:

- Determining the susceptibility backend/source (CSV vs ORCA).
- Resolving the ORCA QDPT section to read, honoring legacy overrides when
  provided and otherwise selecting the best available method by priority.

The functions here intentionally return simple primitives (strings/tuples)
so pipelines and loaders do not need to embed backend-specific heuristics.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import numpy as np

from simpnmr.core.const.physics import KB, MU0, MUB, NA
from simpnmr.io.qc.backends.orca.detect import detect_susc_methods
from simpnmr.io.qc.detect import detect_backend

logger = logging.getLogger(__name__)

SusceptibilityBackend = Literal["csv", "orca"]

_ORCA_PREFIX: Final[str] = "orca_"

# Default ORCA susceptibility method priority (best -> fallback).
# Extend this tuple as new ORCA QDPT methods are supported.
ORCA_SUSC_PRIORITY: Final[tuple[str, ...]] = (
    "nevpt2",
    "casscf",
)

SuscFitInputUnits = Literal["A3", "cm3 mol-1", "reduced"]

_DEFAULT_SUSC_FIT_INPUT_UNITS: Final[SuscFitInputUnits] = "A3"
_SUSC_FIT_INPUT_UNIT_ALIASES: Final[dict[str, SuscFitInputUnits]] = {
    "a3": "A3",
    "a^3": "A3",
    "angstrom3": "A3",
    "angstrom^3": "A3",
    "ang3": "A3",
    "ang^3": "A3",
    "å^3": "A3",
    "å3": "A3",
    "Å^3": "A3",
    "Å3": "A3",
    "cm3 mol-1": "cm3 mol-1",
    "cm^3 mol^-1": "cm3 mol-1",
    "cm3/mol": "cm3 mol-1",
    "reduced": "reduced",
}
_DIMENSIONLESS_SUSC_FIT_VARIABLES: Final[frozenset[str]] = frozenset(
    {"rh_over_ax", "alpha", "beta", "gamma"}
)


def resolve_susceptibility_backend(susceptibility_file: str) -> SusceptibilityBackend:
    """Resolve which backend should be used to read susceptibility.

    Args:
        susceptibility_file: Path to the susceptibility source.

    Returns:
        Backend identifier ("csv" or "orca").

    Raises:
        ValueError: If the backend is unsupported.
    """

    path = Path(susceptibility_file)
    if path.suffix.lower() == ".csv":
        return "csv"

    backend = detect_backend(susceptibility_file)

    if backend == "orca":
        return "orca"

    if backend == "molcas":
        raise ValueError("Susceptibility Molcas support is in active development")

    raise ValueError(
        "Unsupported QC backend for "
        f"susceptibility_file='{susceptibility_file}': {backend}"
    )


def normalize_susc_fit_input_units(value: str | None) -> SuscFitInputUnits:
    """Normalize user-facing susceptibility-fit input units.

    Args:
        value: Optional YAML value from ``susc_fit:input_units``.

    Returns:
        Canonical input-unit label used by the application layer.

    Raises:
        ValueError: If the unit label is unsupported.
    """

    if value is None or value == "":
        return _DEFAULT_SUSC_FIT_INPUT_UNITS

    if isinstance(value, (list, tuple)):
        value = value[0] if value else _DEFAULT_SUSC_FIT_INPUT_UNITS

    if not isinstance(value, str):
        raise ValueError("susc_fit:input_units must be a string")

    normalized = _SUSC_FIT_INPUT_UNIT_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise ValueError(
            "Unknown susc_fit:input_units "
            f"{value!r}. Supported values are: 'A3', 'cm3 mol-1', 'reduced'."
        )

    return normalized


def resolve_susc_fit_variables(
    *,
    raw_variables: dict[str, list[object] | tuple[object, object]],
    input_units: str | None,
    temperature: float,
    spin: float,
    total_J: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert YAML susceptibility-fit variables into canonical internal units.

    The fitting models operate in internal ``Å^3`` units. This helper converts
    user-facing YAML values into those canonical units while preserving
    dimensionless parameters such as ``rh_over_ax``.

    Args:
        raw_variables: Mapping from variable name to ``[mode, value]`` pair.
        input_units: Optional unit label from ``susc_fit:input_units``.
        temperature: Experiment temperature in kelvin. Used for ``reduced`` input.
        spin: Electronic spin quantum number. Used for ``reduced`` input.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the Curie prefactor used for ``reduced`` units.

    Returns:
        Tuple ``(fit_vars, fix_vars)`` with canonical values in ``Å^3``.

    Raises:
        ValueError: If the variable schema or input units are invalid.
    """

    units = normalize_susc_fit_input_units(input_units)
    scale_to_a3 = _get_susc_fit_input_scale_to_a3(
        input_units=units,
        temperature=temperature,
        spin=spin,
        total_J=total_J,
    )

    fit_vars: dict[str, float] = {}
    fix_vars: dict[str, float] = {}

    for key, raw_value in raw_variables.items():
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
            raise ValueError(
                "Each susc_fit:variables entry must be a [mode, value] pair. "
                f"Got {key}={raw_value!r}."
            )

        mode, value = raw_value
        if not isinstance(mode, str):
            raise ValueError(
                f"susc_fit:variables:{key} mode must be 'fit' or 'fix', got {mode!r}."
            )

        mode_normalized = mode.strip().lower()
        if mode_normalized not in {"fit", "fix"}:
            raise ValueError(
                f"susc_fit:variables:{key} mode must be 'fit' or 'fix', got {mode!r}."
            )

        try:
            numeric_value = float(value)
        except Exception as exc:
            raise ValueError(
                f"susc_fit:variables:{key} value must be numeric, got {value!r}."
            ) from exc

        if key not in _DIMENSIONLESS_SUSC_FIT_VARIABLES:
            numeric_value *= scale_to_a3

        target = fit_vars if mode_normalized == "fit" else fix_vars
        target[key] = numeric_value

    return fit_vars, fix_vars


def resolve_orca_section(
    susceptibility_file: str,
    susceptibility_format: str | None,
    *,
    priority: tuple[str, ...] = ORCA_SUSC_PRIORITY,
) -> str:
    """Resolve the ORCA QDPT section label to read.

    Resolution order:
      1) Legacy explicit override via `susceptibility_format` (e.g. "orca_nevpt2").
      2) Autodetect available methods in the ORCA output and select by priority.

    Args:
        susceptibility_file: Path to the ORCA output file.
        susceptibility_format: Optional legacy override (e.g. "orca_nevpt2"). If
            None or "orca", autodetection is used.
        priority: Ordered preference list (best -> fallback).

    Returns:
        Selected section label (lowercase, e.g. "nevpt2", "casscf").

    Raises:
        ValueError: If no supported methods are present in the output.
    """

    fmt = _normalize_format(susceptibility_format)

    # Honor explicit legacy override if present.
    explicit = _extract_orca_section_from_format(fmt)
    if explicit is not None:
        return explicit

    # Otherwise autodetect and select by priority.
    return _resolve_orca_section_autodetect(
        susceptibility_file,
        priority=priority,
    )


def resolve_susceptibility_source(
    susceptibility_file: str,
    susceptibility_format: str | None,
) -> tuple[SusceptibilityBackend, str | None]:
    """Resolve backend and (if ORCA) the section to read.

    Args:
        susceptibility_file: Path to the susceptibility source.
        susceptibility_format: Optional legacy override.

    Returns:
        Tuple (backend, section). For CSV, section is None. For ORCA, section is
        a lowercase method label.
    """

    backend = resolve_susceptibility_backend(susceptibility_file)
    if backend == "csv":
        return backend, None

    section = resolve_orca_section(susceptibility_file, susceptibility_format)
    return backend, section


def _normalize_format(fmt: str | None) -> str | None:
    """Normalize a user-provided susceptibility format string."""

    if fmt is None:
        return None

    cleaned = fmt.strip().lower()
    return cleaned or None


def _extract_orca_section_from_format(fmt: str | None) -> str | None:
    """Extract an explicit ORCA section from a legacy format string."""

    if fmt is None:
        return None

    # Accept "orca" as 'auto'.
    if fmt == "orca":
        return None

    if fmt.startswith(_ORCA_PREFIX) and len(fmt) > len(_ORCA_PREFIX):
        return fmt.split(_ORCA_PREFIX, 1)[1]

    return None


@lru_cache(maxsize=32)
def _resolve_orca_section_autodetect(
    susceptibility_file: str,
    *,
    priority: tuple[str, ...],
) -> str:
    """Autodetect ORCA susceptibility methods and pick by priority."""

    methods = detect_susc_methods(susceptibility_file)
    method_set = set(methods)

    for method in priority:
        if method in method_set:
            return method

    raise ValueError(
        "No supported ORCA susceptibility methods found. "
        f"Detected methods: {sorted(method_set)}"
    )


def _get_susc_fit_input_scale_to_a3(
    *,
    input_units: SuscFitInputUnits,
    temperature: float,
    spin: float,
    total_J: float | None = None,
) -> float:
    """Return the multiplicative factor that converts input values to ``Å^3``.

    Args:
        input_units: Canonical unit label (``"A3"``, ``"cm3 mol-1"``,
            ``"reduced"``).
        temperature: Experiment temperature in Kelvin. Required for
            ``"reduced"`` units.
        spin: Spin quantum number S.  Required for ``"reduced"`` units.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the Curie prefactor used for ``"reduced"`` units.

    Returns:
        Scale factor (float) such that ``input_value * scale == value_in_A3``.
    """

    if input_units == "A3":
        return 1.0

    if input_units == "cm3 mol-1":
        return 1.0 / (1e-24 * NA / (4 * np.pi))

    if spin is None:
        raise ValueError(
            "susc_fit:input_units='reduced' requires hyperfine:spin or an "
            "inferable spin from the hyperfine QC file"
        )

    if temperature <= 0:
        raise ValueError(
            "susc_fit:input_units='reduced' requires a positive experiment temperature"
        )

    return _compute_curie_prefactor(spin, total_J) / float(temperature)


def _compute_curie_prefactor(
    spin: float, total_J: float | None = None
) -> float:
    """Return the Curie prefactor in ``Å^3 K`` for reduced susc. units."""
    J_eff = total_J if total_J is not None else spin
    return (MU0 * MUB**2 * J_eff * (J_eff + 1.0)) / (3.0 * KB) * 1e30
