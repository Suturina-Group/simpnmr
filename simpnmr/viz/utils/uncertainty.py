# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Utilities for formatting uncertainties in compact scientific notation."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal


def _decimal_places_for_uncertainty(uncertainty: float, *, sig_digits: int) -> int:
    """Return decimal places needed to keep ``sig_digits`` in an uncertainty.

    Args:
        uncertainty: Positive uncertainty value.
        sig_digits: Number of significant digits to retain.

    Returns:
        Number of decimal places required for fixed-point rounding.
    """
    exponent = math.floor(math.log10(abs(uncertainty)))
    return max(sig_digits - 1 - exponent, 0)


def _quantize_decimal(value: float, *, decimal_places: int) -> Decimal:
    """Round a float to a fixed number of decimal places using half-up rules.

    Args:
        value: Numeric value to round.
        decimal_places: Decimal places to retain.

    Returns:
        Rounded decimal value.
    """
    quant = Decimal("1").scaleb(-decimal_places)
    return Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)


def format_compact_uncertainty(
    value: float,
    uncertainty: float | None,
    *,
    sig_digits: int = 1,
) -> str:
    """Format ``value ± uncertainty`` as compact uncertainty notation.

    This helper converts numeric input into the standard compact form used in
    scientific tables, where the uncertainty is written in parentheses and
    aligned with the final digits of the rounded value.

    Examples:
        ``0.062, 0.083 -> "0.062(83)"``
        ``12.3, 0.4 -> "12.3(4)"``
        ``1.234, 0.056 -> "1.234(56)"``

    Args:
        value: Central value.
        uncertainty: Absolute uncertainty associated with ``value``.
        sig_digits: Significant digits retained in the uncertainty. Must be at
            least 1. The default of 2 matches common scientific table style.

    Returns:
        Compact uncertainty string. If ``uncertainty`` is ``None`` the function
        returns the value formatted with general numeric representation.

    Raises:
        ValueError: If ``sig_digits < 1`` or if ``uncertainty`` is negative or
            non-finite.
    """
    if sig_digits < 1:
        raise ValueError("sig_digits must be at least 1.")

    if uncertainty is None:
        return f"{value:g}"

    if not math.isfinite(value):
        raise ValueError("value must be finite.")
    if not math.isfinite(uncertainty):
        raise ValueError("uncertainty must be finite.")
    if uncertainty < 0:
        raise ValueError("uncertainty must be non-negative.")

    if uncertainty == 0:
        return f"{value:g}(0)"

    decimal_places = _decimal_places_for_uncertainty(
        uncertainty,
        sig_digits=sig_digits,
    )

    # Coarsen until the parenthesised digit is ≤ 5.
    while True:
        rounded_uncertainty = _quantize_decimal(
            uncertainty,
            decimal_places=decimal_places,
        )
        scaled_uncertainty = int(
            rounded_uncertainty.scaleb(decimal_places).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
        if scaled_uncertainty <= 5:
            break
        decimal_places -= 1

    rounded_value = _quantize_decimal(
        value,
        decimal_places=decimal_places,
    )

    if decimal_places >= 0:
        value_str = f"{rounded_value:.{decimal_places}f}"
    else:
        value_str = str(int(rounded_value))

    return f"{value_str}({scaled_uncertainty})"
