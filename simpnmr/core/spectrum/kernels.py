# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define spectral line-shape kernels.

Provides Gaussian and Lorentzian peak functions for spectrum construction.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian(x: ArrayLike, fwhm: float, b: float, area: float) -> NDArray:
    """Evaluates a Gaussian peak with a given position, width, and area.

    The functional form is:

        ``g(x) = area/(c*sqrt(2*pi)) * exp(-(x-b)^2/(2*c^2))``

    where ``c = fwhm/(2*sqrt(2*ln(2)))``.

    Args:
        x: Coordinate grid.
        fwhm: Full width at half maximum.
        b: Peak position.
        area: Peak area.

    Returns:
        Array of ``g(x)`` values.
    """

    c = fwhm / (2 * np.sqrt(2 * np.log(2)))

    a = 1.0 / (c * np.sqrt(2 * np.pi))

    gaus = a * np.exp(-((x - b) ** 2) / (2 * c**2))

    gaus *= area

    return gaus


def lorentzian(x: ArrayLike, fwhm, x0, area) -> NDArray:
    """Evaluates a Lorentzian peak with a given position, width, and area.

    The functional form is:

        ``L(x) = (0.5*area*fwhm/pi) / ((x-x0)^2 + (0.5*fwhm)^2)``

    Args:
        x: Coordinate grid.
        fwhm: Full width at half maximum.
        x0: Peak position.
        area: Peak area.

    Returns:
        Array of ``L(x)`` values.
    """

    lor = 0.5 * fwhm / np.pi
    lor *= 1.0 / ((x - x0) ** 2 + (0.5 * fwhm) ** 2)

    lor *= area

    return lor
