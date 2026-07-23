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
        compute_chi_prefactor,
        compute_g_sq_components,
    )

    g_diag = np.diag([gx, gy, gz])
    g_sq = compute_g_sq_components(g_diag)

    # Convert ZFS from cm⁻¹ to Joules
    D_J = D_cmm1 * H * C * 100.0
    E_J = E_cmm1 * H * C * 100.0

    prefactor = compute_chi_prefactor(spin)  # Å³·K
    R = _zyz_rotation_matrix(alpha_deg, beta_deg, gamma_deg)

    # g-tensor rotated into the molecular frame (same eigenframe as chi), used
    # to compute the g-corrected isotropic susceptibility for the Fermi contact.
    g_mol = R @ g_diag @ R.T

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

        susc = Susceptibility(tensor=chi_mol, temperature=float(T))
        # The SH tensor is built from the real g-tensor, so its isotropic part is
        # the g-corrected susceptibility that drives the Fermi contact. Record it
        # explicitly (chi and g are both known) so predict splits FC into
        # spin-only and g-correction contributions, matching the fit workflow.
        susc.iso_g_corr = float(
            get_g_corr_iso_susc(
                spin=spin,
                orbit=0.0,
                g_tensor=g_mol,
                chi_tensors=chi_mol,
                total_momentum_J=None,
            )
        )
        suscs.append(susc)

    return suscs


def build_susceptibility_from_bleaney(
    B20_cmm1: float,
    B22_cmm1: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
    spin: float,
    orbit_L: float,
    total_J: float,
    temperatures: list[float],
) -> list:
    """Build Susceptibility objects from Bleaney crystal-field parameters.

    Uses the second-rank Stevens crystal-field parameters B²₀ and B²₂ (in
    cm⁻¹) to parameterise the ZFS via D = 3·B²₀ and E = B²₂.  The g-tensor
    is taken as isotropic with the Landé g_J value, which is appropriate for
    lanthanide/actinide J-multiplet states where the orbital contribution is
    quenched only by the total angular momentum.

    Args:
        B20_cmm1: Second-rank axial Stevens parameter B²₀ in cm⁻¹.
        B22_cmm1: Second-rank rhombic Stevens parameter B²₂ in cm⁻¹.
        alpha_deg: ZYZ Euler angle α (molecular → eigenframe) in degrees.
        beta_deg:  ZYZ Euler angle β in degrees.
        gamma_deg: ZYZ Euler angle γ in degrees.
        spin:    Spin quantum number S.
        orbit_L: Orbital angular momentum quantum number L.
        total_J: Total angular momentum quantum number J (must be set).
        temperatures: Sequence of temperatures in Kelvin.

    Returns:
        List of Susceptibility objects, one per temperature.

    Raises:
        ValueError: If ``total_J`` is zero or not provided
            (method requires L≠0).
    """
    if not total_J:
        raise ValueError(
            "susceptibility:method bleaney requires total_J to be set "
            "(hyperfine:total_momentum_J must be provided for L≠0 systems)"
        )

    from simpnmr.core.build.eff_factors import alpha_J_from_LSJ
    from simpnmr.core.domain.tensor import Susceptibility
    from simpnmr.core.fitting.vt import (
        compute_analytic_component,
        compute_chi_prefactor,
    )

    # Rank-2 Stevens operator-equivalent factor.
    alpha_J = alpha_J_from_LSJ(int(round(orbit_L)), spin, total_J)

    # Scale B²₀ and B²₂ by α_J before converting to ZFS parameters.
    # D = 3·B²₀·α_J,  E = B²₂·α_J  (Bleaney/Stevens convention).
    D_cmm1 = 3.0 * B20_cmm1 * alpha_J
    E_cmm1 = B22_cmm1 * alpha_J

    D_J_si = D_cmm1 * H * C * 100.0
    E_J_si = E_cmm1 * H * C * 100.0

    # g_J² enters explicitly as g_sq_iso; the Curie prefactor C₀ carries
    # only μ₀ μ_B² J(J+1)/(3 k_B) without a g² factor.
    g_J = calc_g_eff(spin, orbit_L, total_J)
    # compute_analytic_component needs both the g² invariants (ax/rh) and the
    # iso-branch keys (ge_g_*). Here we are building a susceptibility *tensor*,
    # whose isotropic element is the ordinary Curie susceptibility ∝ g_J² — the
    # g_e·g cross-product is a Fermi-contact-shift construct that does not enter
    # the susceptibility itself. So ge_g_iso = g_J² and the anisotropic
    # cross-products vanish for an isotropic g_J.
    g_sq = {
        "g_sq_iso": g_J ** 2, "g_sq_ax": 0.0, "g_sq_rh": 0.0,
        "ge_g_iso": g_J ** 2, "ge_g_ax": 0.0, "ge_g_rh": 0.0,
    }
    prefactor = compute_chi_prefactor(spin, total_J)   # C₀, no g_J²

    R = _zyz_rotation_matrix(alpha_deg, beta_deg, gamma_deg)

    def _analytic(comp, t):
        return (
            float(
                compute_analytic_component(
                    comp, t, g_sq, D_J_si, E_J_si, spin, total_J=total_J
                )[0]
            )
            * prefactor
        )

    suscs = []
    for T in temperatures:
        t = np.asarray([float(T)], dtype=float)
        chi_iso = _analytic("iso", t)
        chi_ax = _analytic("ax", t)
        chi_rh = _analytic("rh", t)

        chi_zz = chi_iso + 2.0 / 3.0 * chi_ax
        chi_xx = chi_iso - 1.0 / 3.0 * chi_ax + chi_rh
        chi_yy = chi_iso - 1.0 / 3.0 * chi_ax - chi_rh

        chi_eigen = np.diag([chi_xx, chi_yy, chi_zz])
        chi_mol = R @ chi_eigen @ R.T

        susc = Susceptibility(tensor=chi_mol, temperature=float(T))
        # Bleaney uses an isotropic g_J, so the g-corrected isotropic
        # susceptibility equals the true Tr(chi)/3. Record it as iso_g_corr so
        # the Fermi contact is driven by that value (and split reports a ~zero
        # g-correction delta), consistent with the other parametric paths.
        susc.iso_g_corr = susc.iso
        suscs.append(susc)

    return suscs


def compute_bleaney_params(
    chi_ax: float,
    chi_rh: float,
    temperature: float,
    spin: float,
    orbit_L: float,
    total_J: float,
) -> tuple[float, float]:
    """Invert fitted χ_ax / χ_rh to Bleaney parameters B²₀ and B²₂ (cm⁻¹).

    Follows the forward procedure of ``build_susceptibility_from_bleaney`` in
    reverse.  D = 3·B²₀ and E = B²₂.  The g_J² factor is carried entirely
    by the Curie prefactor C_J used to normalise the input susceptibilities;
    ``g_sq_iso`` is therefore 1 here (no double-counting).

    ``chi_ax`` and ``chi_rh`` must be the Curie-reduced components
    χ_ax / C₀ and χ_rh / C₀ (units K⁻¹), where C₀ =
    ``compute_chi_prefactor(spin, total_J)`` (without ``orbit_L``, so
    g_J² is NOT included in the prefactor).  g_J² enters the analytic
    formula explicitly through ``g_sq_iso = g_J²``.

    Args:
        chi_ax: Axial reduced susceptibility (K⁻¹, = χ_ax / C₀).
        chi_rh: Rhombic reduced susceptibility (K⁻¹, = χ_rh / C₀).
        temperature: Temperature in Kelvin.
        spin: Electron spin quantum number S.
        orbit_L: Orbital angular momentum L, used to compute g_J and α_J.
        total_J: Total angular momentum quantum number J (must be > 0).

    Returns:
        Tuple ``(B20_cmm1, B22_cmm1, scale_ax, scale_rh)`` where the Stevens
        parameters are in cm⁻¹ and the scale factors satisfy
        ``sigma(B20) = sigma(chi_ax / C₀) * scale_ax`` and
        ``sigma(B22) = sigma(chi_rh / C₀) * scale_rh``.
    """
    from simpnmr.core.build.eff_factors import alpha_J_from_LSJ
    from simpnmr.core.fitting.vt import compute_analytic_component

    # g_J² is explicit in g_sq_iso; C₀ (no g_J²) is used for normalisation.
    g_J = calc_g_eff(spin, orbit_L, total_J)
    g_sq = {"g_sq_iso": g_J ** 2, "g_sq_ax": 0.0, "g_sq_rh": 0.0}
    T_arr = np.asarray([float(temperature)], dtype=float)

    # Probe with unit D_J = 1 cm⁻¹ in Joules to get sensitivity.
    unit_J = H * C * 100.0  # 1 cm⁻¹ in Joules

    # K⁻¹ per (cm⁻¹ of D)
    coeff_D_ax = float(
        compute_analytic_component(
            "ax", T_arr, g_sq, unit_J, 0.0, spin, total_J=total_J
        )[0]
    )

    # K⁻¹ per (cm⁻¹ of E)
    coeff_E_rh = float(
        compute_analytic_component(
            "rh", T_arr, g_sq, 0.0, unit_J, spin, total_J=total_J
        )[0]
    )

    alpha_J = alpha_J_from_LSJ(int(round(orbit_L)), spin, total_J)

    D_cmm1 = float(chi_ax) / coeff_D_ax
    E_cmm1 = float(chi_rh) / coeff_E_rh

    # Invert D = 3·B²₀·α_J  and  E = B²₂·α_J.
    B20_cmm1 = D_cmm1 / (3.0 * alpha_J)
    B22_cmm1 = E_cmm1 / alpha_J

    # Linear scale factors for error propagation:
    # sigma(B20) = sigma(chi_ax / C₀) * scale_ax  (cm⁻¹ per K⁻¹)
    scale_ax = abs(1.0 / (3.0 * alpha_J * coeff_D_ax))
    scale_rh = abs(1.0 / (alpha_J * coeff_E_rh))

    return B20_cmm1, B22_cmm1, scale_ax, scale_rh


def build_susceptibility_from_reduced_chi(
    chi_iso_T: float | list[float],
    chi_ax_T: float | list[float],
    chi_rh_T: float | list[float],
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
    spin: float,
    temperatures: list[float],
    total_J: float | None = None,
) -> list:
    """Build Susceptibility objects from reduced ΔχT values and Euler angles.

    The reduced ΔχT components (``chi_comp * T / prefactor``, dimensionless)
    are the quantities plotted on the isoaxrh susceptibility plot after
    ``fit_susc``.  Supplying them directly bypasses any SH parameterisation.

    Each component may be a single scalar (used at all temperatures, i.e. pure
    Curie / temperature-independent limit) or a list with one value per
    temperature.

    Args:
        chi_iso_T: Reduced iso component Δχ_iso·T/C (dimensionless). This is the
            g-corrected isotropic susceptibility: fit_susc plots the fitted iso
            (which it records as the g-corrected chi_iso) on the isoaxrh plot, so
            the value is stored as ``iso_g_corr`` on each returned Susceptibility.
        chi_ax_T:  Reduced axial component Δχ_ax·T/C (dimensionless).
        chi_rh_T:  Reduced rhombic component Δχ_rh·T/C (dimensionless).
        alpha_deg: ZYZ Euler angle α (molecular → eigenframe) in degrees.
        beta_deg:  ZYZ Euler angle β in degrees.
        gamma_deg: ZYZ Euler angle γ in degrees.
        spin:       Spin quantum number S.
        temperatures: Sequence of temperatures in Kelvin.
        total_J:   Total angular momentum J. When provided, J(J+1) replaces
                   S(S+1) in the Curie prefactor.

    Returns:
        List of Susceptibility objects, one per temperature.
    """
    from simpnmr.core.domain.tensor import Susceptibility
    from simpnmr.core.fitting.vt import compute_chi_prefactor

    prefactor = compute_chi_prefactor(spin, total_J)  # Å³·K
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

        susc = Susceptibility(tensor=chi_mol, temperature=float(T))
        # The reduced iso component is taken from the fit_susc isoaxrh plot, whose
        # isotropic value is the g-corrected chi_iso (fit_susc records its fitted
        # iso as iso_g_corr). Record it as iso_g_corr so predict splits the Fermi
        # contact into spin-only and g-correction contributions, as in the fit.
        susc.iso_g_corr = susc.iso
        suscs.append(susc)

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

    # Trace-based expression with g correction (Å³). The matrix product (not an
    # element-wise product) is required so the result is a rotation-invariant
    # scalar: Tr(chi @ g^-T) is frame-independent, whereas summing only the
    # diagonal products depends on the molecular-frame orientation.
    chi_true_iso = g_eff / 3.0 * np.trace(
        chi_tensors @ np.linalg.inv(g_tensor.T)
    )

    return chi_true_iso
