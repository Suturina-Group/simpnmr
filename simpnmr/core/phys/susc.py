# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct magnetic susceptibility values and tensors.

Provides helpers to build susceptibility tensors and isotropic values from
quantum-chemical outputs and spin parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.build.eff_factors import calc_g_eff, choose_S_eff
from simpnmr.core.const.physics import C, H, KB, MU0, MUB


def get_spin_only_susc(
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    temperature: float,
) -> float:
    """Computes the spin-only isotropic molar susceptibility in Å³.

    Uses the Curie law with an effective g-factor and an effective angular
    momentum quantum number (``S`` for spin-only systems, ``J`` when defined).

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None`` for
            spin-only.
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


def _zyz_rotation_matrix(
    alpha_deg: float, beta_deg: float, gamma_deg: float
) -> NDArray:
    """Return ZYZ active rotation matrix R = Rz(α) @ Ry(β) @ Rz(γ)."""
    a, b, g = (
        np.deg2rad(alpha_deg),
        np.deg2rad(beta_deg),
        np.deg2rad(gamma_deg),
    )
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    Rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    Ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=float)
    Rz_g = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=float)
    return Rz_a @ Ry_b @ Rz_g


def build_susceptibility_from_sh(
    gx: float,
    gy: float,
    gz: float,
    D_cmm1: float,
    E_cmm1: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
    spin: float,
    temperatures: list[float],
) -> list:
    """Build a list of Susceptibility objects from spin-Hamiltonian parameters.

    Evaluates the analytic high-temperature expansion for the three irreducible
    susceptibility components (iso, ax, rh) using the given g-tensor principal
    values and ZFS parameters, then rotates the diagonal tensor from the SH
    eigenframe into the molecular frame using the supplied ZYZ Euler angles.

    Args:
        gx, gy, gz: Principal g-tensor components.
        D_cmm1: Axial zero-field splitting in cm⁻¹.
        E_cmm1: Rhombic zero-field splitting in cm⁻¹.
        alpha_deg: ZYZ Euler angle α (molecular → eigenframe) in degrees.
        beta_deg: ZYZ Euler angle β in degrees.
        gamma_deg: ZYZ Euler angle γ in degrees.
        spin: Total spin quantum number S.
        temperatures: Sequence of temperatures in Kelvin.

    Returns:
        List of Susceptibility objects, one per temperature.
    """
    from simpnmr.core.domain.tensor import Susceptibility
    from simpnmr.core.fitting.vt import (
        compute_analytic_component,
        compute_curie_prefactor,
        compute_g_sq_components,
    )

    g_diag = np.diag([gx, gy, gz])
    g_sq = compute_g_sq_components(g_diag)

    # Convert ZFS from cm⁻¹ to Joules
    D_J = D_cmm1 * H * C * 100.0
    E_J = E_cmm1 * H * C * 100.0

    prefactor = compute_curie_prefactor(spin)  # Å³·K
    R = _zyz_rotation_matrix(alpha_deg, beta_deg, gamma_deg)

    def _analytic(comp, t):
        # compute_analytic_component returns chi/prefactor [1/K];
        # chi [Å³] = analytic * prefactor
        return (
            float(compute_analytic_component(comp, t, g_sq, D_J, E_J, spin)[0])
            * prefactor
        )

    suscs = []
    for T in temperatures:
        t = np.asarray([float(T)], dtype=float)
        chi_iso = _analytic("iso", t)
        chi_ax = _analytic("ax", t)
        chi_rh = _analytic("rh", t)

        # Reconstruct diagonal tensor eigenvalues from irreducible components:
        # chi_ax = 3/2 * (chi_zz - chi_iso)  →  chi_zz = chi_iso + 2/3 * chi_ax
        # chi_rh = (chi_xx - chi_yy) / 2
        # chi_iso = (chi_xx + chi_yy + chi_zz) / 3
        chi_zz = chi_iso + 2.0 / 3.0 * chi_ax
        chi_xx = chi_iso - 1.0 / 3.0 * chi_ax + chi_rh
        chi_yy = chi_iso - 1.0 / 3.0 * chi_ax - chi_rh

        chi_eigen = np.diag([chi_xx, chi_yy, chi_zz])
        # Rotate eigenframe → molecular frame: chi_mol = R @ chi_eigen @ R.T
        chi_mol = R @ chi_eigen @ R.T

        suscs.append(Susceptibility(tensor=chi_mol, temperature=float(T)))

    return suscs


def build_susceptibility_from_reduced_chi(
    chi_iso_T: float | list[float],
    chi_ax_T: float | list[float],
    chi_rh_T: float | list[float],
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
    spin: float,
    temperatures: list[float],
) -> list:
    """Build Susceptibility objects from reduced ΔχT values and Euler angles.

    The reduced ΔχT components (``chi_comp * T / prefactor``, dimensionless)
    are the quantities plotted on the isoaxrh susceptibility plot after
    ``fit_susc``.  Supplying them directly bypasses any SH parameterisation.

    Each component may be a single scalar (used at all temperatures, i.e. pure
    Curie / temperature-independent limit) or a list with one value per
    temperature.

    Args:
        chi_iso_T: Reduced iso component Δχ_iso·T/C (dimensionless).
        chi_ax_T:  Reduced axial component Δχ_ax·T/C (dimensionless).
        chi_rh_T:  Reduced rhombic component Δχ_rh·T/C (dimensionless).
        alpha_deg: ZYZ Euler angle α (molecular → eigenframe) in degrees.
        beta_deg:  ZYZ Euler angle β in degrees.
        gamma_deg: ZYZ Euler angle γ in degrees.
        spin:       Total spin quantum number S (used for Curie prefactor).
        temperatures: Sequence of temperatures in Kelvin.

    Returns:
        List of Susceptibility objects, one per temperature.
    """
    from simpnmr.core.domain.tensor import Susceptibility
    from simpnmr.core.fitting.vt import compute_curie_prefactor

    prefactor = compute_curie_prefactor(spin)  # Å³·K
    R = _zyz_rotation_matrix(alpha_deg, beta_deg, gamma_deg)

    def _broadcast(val, n):
        if isinstance(val, (int, float)):
            return [float(val)] * n
        vals = list(val)
        if len(vals) != n:
            raise ValueError(
                f"Expected {n} reduced-chi values (one per temperature), "
                f"got {len(vals)}"
            )
        return [float(v) for v in vals]

    n = len(temperatures)
    iso_vals = _broadcast(chi_iso_T, n)
    ax_vals = _broadcast(chi_ax_T, n)
    rh_vals = _broadcast(chi_rh_T, n)

    suscs = []
    for T, ciso, cax, crh in zip(temperatures, iso_vals, ax_vals, rh_vals):
        # chi [Å³] = chiT_reduced * prefactor / T
        chi_iso = ciso * prefactor / float(T)
        chi_ax = cax * prefactor / float(T)
        chi_rh = crh * prefactor / float(T)

        # Reconstruct diagonal eigenvalues:
        # chi_ax = 3/2*(chi_zz - chi_iso)  →  chi_zz = chi_iso + 2/3*chi_ax
        # chi_rh = (chi_xx - chi_yy)/2
        chi_zz = chi_iso + 2.0 / 3.0 * chi_ax
        chi_xx = chi_iso - 1.0 / 3.0 * chi_ax + chi_rh
        chi_yy = chi_iso - 1.0 / 3.0 * chi_ax - chi_rh

        chi_eigen = np.diag([chi_xx, chi_yy, chi_zz])
        chi_mol = R @ chi_eigen @ R.T

        suscs.append(Susceptibility(tensor=chi_mol, temperature=float(T)))

    return suscs


def get_g_corr_iso_susc(
    spin: float,
    orbit: float,
    g_tensor: NDArray,
    chi_tensors: NDArray,
    total_momentum_J: float | None,
) -> float:
    """Computes a g-tensor-corrected isotropic susceptibility in Å³.

    Uses susceptibility principal components (from `chi_tensors`) and the
    supplied g-tensor to compute an effective isotropic value.

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
    chi_true_iso = g_eff / 3.0 * np.trace(
        chi_tensors * np.linalg.inv(g_tensor.T)
    )

    return chi_true_iso
