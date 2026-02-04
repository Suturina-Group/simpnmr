# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct magnetic susceptibility values and tensors.

Provides helpers to build susceptibility tensors and isotropic values from
quantum-chemical outputs and spin parameters.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.build.eff_factors import calc_g_eff, choose_S_eff
from simpnmr.core.const.physics import KB, MU0, MUB, NA
from simpnmr.core.domain.tensor import Susceptibility


def susc_from_orca_xt(
    temperature: float,
    tensor_xt: NDArray,
    *,
    iso_mode: str = "raw",
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> Susceptibility:
    """Convert an ORCA XT tensor to a Susceptibility domain object.

    ORCA reports XT in units of cm^3 mol^-1 K. This factory converts XT to a
    molar susceptibility tensor chi (Å^3) by applying the appropriate physical
    conversion and dividing by temperature.

    The isotropic component (``susc.iso``) is set according to ``iso_mode``:
    - "g_corr": g-tensor-corr iso (requires g_tensor and spin_S; orbit_L optional)
    - "spin_only": spin-only Curie iso (requires spin_S; orbit_L optional)
    - "raw": trace-based iso from the susceptibility tensor

    Args:
        temperature: Temperature in Kelvin.
        tensor_xt: ORCA XT tensor as a (3, 3) array in cm^3 mol^-1 K.
        iso_mode: Iso handling mode ("raw", "spin_only", "g_corr").
        electronic: Optional electronic-state context used for iso computation.

    Returns:
        Susceptibility domain object.
    """

    # Conversion factor:
    # 1 cm^3 mol^-1 = 1e-6 m^3 / N_A
    # then convert m^3 -> Å^3 (1 m^3 = 1e30 Å^3)
    conv = 1e-24 * NA / (4.0 * np.pi)
    conv = 1.0 / conv

    chi_tensor = tensor_xt / temperature * conv

    susc = Susceptibility(chi_tensor, temperature=float(temperature))
    susc.calc_irred()

    if iso_mode == "g_corr":
        if electronic is None:
            raise ValueError("iso_mode='g_corr' requires an electronic-state context.")
        spin = getattr(electronic, "spin_S", None)
        orbit = getattr(electronic, "orbit_L", None)
        total_J = getattr(electronic, "total_J", None)

        if g_tensor is None or spin is None:
            raise ValueError("iso_mode='g_corr' requires g_tensor and quantum number")

        orbit_val = 0.0 if orbit is None else float(orbit)

        susc.iso = float(
            get_g_corr_iso_susc(
                spin=float(spin),
                orbit=orbit_val,
                g_tensor=np.asarray(g_tensor, dtype=float),
                chi_tensors=chi_tensor,
                total_momentum_J=total_J,
            )
        )

    elif iso_mode == "spin_only":
        if electronic is None:
            raise ValueError(
                "iso_mode='spin_only' requires an electronic-state context."
            )
        spin = getattr(electronic, "spin_S", None)
        orbit = getattr(electronic, "orbit_L", None)
        total_J = getattr(electronic, "total_J", None)

        if spin is None:
            raise ValueError("iso_mode='spin_only' requires electronic.spin_S.")

        # For strict domain naming, orbit_L may legitimately be None for spin-only.
        # Treat missing orbit_L as 0.0 for the Landé-factor helper.
        orbit_val = 0.0 if orbit is None else float(orbit)

        susc.iso = float(
            get_spin_only_susc(
                spin=float(spin),
                orbit=orbit_val,
                total_momentum_J=total_J,
                temperature=float(temperature),
            )
        )

    elif iso_mode == "raw":
        susc.iso = float(np.trace(chi_tensor) / 3.0)

    else:
        raise ValueError(f"Unknown iso_mode: {iso_mode!r}")

    return susc


def get_spin_only_susc(
    spin: float, orbit: float, total_momentum_J: float | None, temperature: float
) -> float:
    """Computes the spin-only isotropic molar susceptibility in Å³.

    Uses the Curie law with an effective g-factor and an effective angular momentum
    quantum number (``S`` for spin-only systems, ``J`` when defined).

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None`` for spin-only.
        temperature: Temperature in Kelvin.

    Returns:
        Spin-only isotropic molar susceptibility in Å³.
    """

    # Landé g-factor uses S, L, J
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Effective moment quantum number for Curie law:
    # S for transition metals, J for lanthanides
    S_eff = choose_S_eff(spin, total_momentum_J)

    # Chi (SI, m^3 mol^-1)
    chi_only_iso_SI = (
        MU0 * MUB**2 * g_eff**2 * S_eff * (S_eff + 1) / (3 * KB * temperature)
    )

    # Convert m^3 to Å^3: 1 Å^3 = 1e-30 m^3
    chi_only_iso = chi_only_iso_SI * 1e30

    return chi_only_iso


def get_g_corr_iso_susc(
    spin: float,
    orbit: float,
    g_tensor: NDArray,
    chi_tensors: NDArray,
    total_momentum_J: float | None,
) -> float:
    """Computes a g-tensor-corrected isotropic susceptibility in Å³.

    Uses susceptibility principal components (from `chi_tensors`) and the supplied
    g-tensor to compute an effective isotropic value.

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        g_tensor: g-tensor as a ``(3, 3)`` array.
        chi_tensors: Susceptibility tensor in Å^3.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Corrected isotropic susceptibility in A^3.

    """

    # Use Landé g_J (or GE) to get an effective g-factor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Trace-based expression with g correction (cm^3 mol^-1)
    chi_true_iso = g_eff / 3.0 * np.trace(chi_tensors * np.linalg.inv(g_tensor.T))

    return chi_true_iso


def build_ab_initio_chit_series(
    suscs_ab_initio: list,
    *,
    g_corr_iso: bool = False,
    spin: float | None = None,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
    g_tensor: NDArray | None = None,
) -> dict[str, np.ndarray]:
    """Build an ab initio chiT series from susceptibility objects.

    This factory:
    - computes irreducible representations for all ab initio tensors
    - sorts data by temperature
    - returns chiT components on the ab initio temperature grid (optionally
    using a g-corrected iso)

    Args:
        suscs_ab_initio: List of susceptibility objects.
        g_corr_iso: Whether to compute g-tensor-corrected iso values.
        spin: Spin quantum number S (required if g_corr_iso).
        orbit: Orbital angular momentum L (required if g_corr_iso).
        total_momentum_J: Total angular momentum J (required if
        g_corr_iso can be inferred).
        g_tensor: g-tensor as a (3, 3) array (required if g_corr_iso).

    Returns:
        Dictionary with:
            temps: Temperatures (K), sorted.
            inv_t: Inverse temperatures (K^-1).
            iso: chiT_iso values (A^3 K).
            ax: chiT_ax values (A^3 K).
            rho: chiT_rho values (A^3 K).
    """
    if not suscs_ab_initio:
        raise ValueError("suscs_ab_initio must be a non-empty list.")

    if g_corr_iso:
        if spin is None or g_tensor is None:
            raise ValueError(
                "g_corr_iso=True requires spin and g_tensor to be provided."
            )

    # Ensure irreducible components / eigenframes are available
    for s in suscs_ab_initio:
        s.calc_irred()

    temps = np.array([float(s.temperature) for s in suscs_ab_initio], dtype=float)

    order = np.argsort(temps)
    temps = temps[order]
    suscs_sorted = [suscs_ab_initio[i] for i in order]

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_t = 1.0 / temps

    if g_corr_iso:
        iso_vals = []
        for s in suscs_sorted:
            iso_vals.append(
                float(
                    get_g_corr_iso_susc(
                        spin=float(spin),
                        orbit=0.0 if orbit is None else float(orbit),
                        g_tensor=np.asarray(g_tensor, dtype=float),
                        chi_tensors=s.tensor,
                        total_momentum_J=total_momentum_J,
                    )
                )
            )
        iso_base = np.asarray(iso_vals, dtype=float)
    else:
        iso_base = np.array([float(s.iso) for s in suscs_sorted], dtype=float)

    iso = iso_base * temps
    ax = np.array([float(s.axiality) for s in suscs_sorted], dtype=float) * temps
    rho = np.array([float(s.rhombicity) for s in suscs_sorted], dtype=float) * temps

    return {
        "temps": temps,
        "inv_t": inv_t,
        "iso": iso,
        "ax": ax,
        "rho": rho,
    }
