# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct magnetic susceptibility tensors and isotropic channels.

Provides builders that assemble tensor-backed susceptibility domain objects
from source data and attach isotropic susceptibility channels from explicit
physical models or direct source values.
"""

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.physics import NA
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.core.phys.susc import get_g_corr_iso_susc, get_spin_only_susc


def build_chi_d_tensor_from_orca(
    temperature: float,
    tensor_xt: NDArray,
) -> Susceptibility:
    """Build a tensor-backed susceptibility object from an ORCA XT tensor.

    ORCA reports XT in units of cm^3 mol^-1 K. This builder converts XT to a
    molar susceptibility tensor chi (Å^3) by applying the appropriate physical
    conversion and dividing by temperature. It constructs only the tensor-backed
    susceptibility domain object and its irreducible tensor representation. It
    does not attach isotropic susceptibility channels.

    Args:
        temperature: Temperature in Kelvin.
        tensor_xt: ORCA XT tensor as a (3, 3) array in cm^3 mol^-1 K.

    Returns:
        Tensor-backed susceptibility domain object.
    """

    # Conversion factor:
    # 1 cm^3 mol^-1 = 1e-6 m^3 / N_A
    # then convert m^3 -> Å^3 (1 m^3 = 1e30 Å^3)
    conv = 1e-24 * NA / (4.0 * np.pi)
    conv = 1.0 / conv

    chi_tensor = tensor_xt / temperature * conv

    susc = Susceptibility(chi_tensor, temperature=float(temperature))
    susc.calc_irred()

    return susc


def build_chi_d_tensor_from_csv(
    temperature: float,
    tensor: NDArray,
) -> Susceptibility:
    """Build a tensor-backed susceptibility object from CSV tensor data.

    The anisotropic susceptibility tensor is taken directly from the CSV input.
    This builder constructs only the tensor-backed susceptibility domain object
    and its irreducible tensor representation. It does not attach isotropic
    susceptibility channels.

    Args:
        temperature: Temperature in Kelvin.
        tensor: Susceptibility tensor from CSV as a (3, 3) array.

    Returns:
        Tensor-backed susceptibility domain object.
    """
    susc = Susceptibility(
        np.asarray(tensor, dtype=float), temperature=float(temperature)
    )
    susc.calc_irred()
    return susc


def build_chi_iso_from_csv(
    susc: Susceptibility,
    *,
    chi_iso: float,
) -> Susceptibility:
    """Attach an isotropic susceptibility read directly from CSV data.

    This builder performs no physical transformation. It records the provided
    CSV isotropic susceptibility as the g-corrected contact channel
    (``susc.iso_g_corr``): a stored ``chi_iso``/``chi_iso_g_corr`` column is the
    isotropic value that drives the (g-corrected) Fermi contact. The true
    Tr(chi)/3 is read-only and derived from the tensor.

    Args:
        susc: Existing susceptibility domain object to enrich.
        chi_iso: Isotropic susceptibility value read directly from CSV.

    Returns:
        The same susceptibility domain object enriched with CSV isotropic
        susceptibility.
    """
    susc.iso_g_corr = float(chi_iso)
    return susc


def build_chi_iso_spin_only(
    susc: Susceptibility,
    *,
    spin: float,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
) -> Susceptibility:
    """Attach spin-only isotropic susceptibility to a domain object.

    This builder computes the spin-only isotropic susceptibility channel and
    stores it in ``susc.iso_spin_only`` on an existing susceptibility domain
    object.

    Args:
        susc: Existing susceptibility domain object to enrich.
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        The same susceptibility domain object enriched with spin-only isotropic
        susceptibility.
    """
    susc.iso_spin_only = float(
        get_spin_only_susc(
            spin=float(spin),
            orbit=0.0 if orbit is None else float(orbit),
            total_momentum_J=total_momentum_J,
            temperature=float(susc.temperature),
        )
    )
    return susc


def build_chi_iso_g_corr(
    susc: Susceptibility,
    *,
    spin: float,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
    g_tensor: NDArray,
) -> Susceptibility:
    """Attach g-corrected isotropic susceptibility to a domain object.

    This builder computes the g-tensor-corrected isotropic susceptibility
    channel and stores it in ``susc.iso_g_corr`` on an existing susceptibility
    domain object.

    Args:
        susc: Existing susceptibility domain object to enrich.
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.
        g_tensor: g-tensor as a (3, 3) array.

    Returns:
        The same susceptibility domain object enriched with g-corrected
        isotropic susceptibility.
    """
    susc.iso_g_corr = float(
        get_g_corr_iso_susc(
            spin=float(spin),
            orbit=0.0 if orbit is None else float(orbit),
            g_tensor=np.asarray(g_tensor, dtype=float),
            chi_tensors=susc.tensor,
            total_momentum_J=total_momentum_J,
        )
    )
    return susc
