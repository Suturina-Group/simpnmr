# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute Solomon–Bloembergen–Morgan relaxation rates.

Provides helpers to evaluate dipolar and contact R1 and R2 rates for labelled nuclei.
"""

import numpy as np

from simpnmr.core.build.eff_factors import calc_g_eff, choose_S_eff
from simpnmr.core.const.physics import MU0, MUB


def calc_r1_dipolar(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    gamma_I_dict,
    omega_I_dict,
    omega_S,
    tau_c1,
    tau_c2,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes SBM dipolar R1 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the longitudinal
    relaxation rate due to electron-nucleus dipolar interactions.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        gamma_I_dict: Nuclear gyromagnetic ratios (rad s^-1 T^-1) by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_c1: Correlation time for the dipolar interaction (s).
        tau_c2: Correlation time entering cross terms (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to dipolar R1 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the prefactor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        gamma_I = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 10)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (gamma_I * g_eff * MUB) ** 2
            * S_eff
            * (S_eff + 1)
        )
        spectral_density = (
            3 * J(omega_I, tau_c1)
            + 6 * J(omega_I + omega_S, tau_c2)
            + J(omega_I - omega_S, tau_c2)
        )
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def calc_r2_dipolar(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    gamma_I_dict,
    omega_I_dict,
    omega_S,
    tau_c1,
    tau_c2,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes SBM dipolar R2 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the transverse
    relaxation rate due to electron-nucleus dipolar interactions.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        gamma_I_dict: Nuclear gyromagnetic ratios (rad s^-1 T^-1) by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_c1: Correlation time for the dipolar interaction (s).
        tau_c2: Correlation time entering cross terms (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to dipolar R2 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the prefactor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        gamma_I = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 15)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (gamma_I * g_eff * MUB) ** 2
            * S_eff
            * (S_eff + 1)
        )
        spectral_density = (
            4 * J(0, tau_c1)
            + 3 * J(omega_I, tau_c1)
            + 6 * J(omega_S, tau_c2)
            + 6 * J(omega_I + omega_S, tau_c2)
            + J(omega_I - omega_S, tau_c2)
        )
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def calc_r1_contact(
    nuclei_labels, Aiso_dict, omega_I_dict, omega_S, tau_e2, spin, total_momentum_J
):
    """Computes SBM contact R1 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the longitudinal
    relaxation rate due to isotropic Fermi-contact hyperfine coupling.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        Aiso_dict: Isotropic hyperfine couplings ``A_iso`` (angular frequency units)
            by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_e2: Electronic correlation time entering the spectral density (s).
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to contact R1 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective angular momentum quantum number for the contact term
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 contact rates
    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (2 / 3) * Aiso**2 * S_eff * (S_eff + 1)
        spectral_density = J(omega_I - omega_S, tau_e2)
        rate = prefactor * spectral_density
        rates[label] = rate
    return rates


def calc_r2_contact(
    nuclei_labels,
    Aiso_dict,
    omega_I_dict,
    omega_S,
    tau_e1,
    tau_e2,
    spin,
    total_momentum_J,
):
    """Computes SBM contact R2 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the transverse
    relaxation rate due to isotropic Fermi-contact hyperfine coupling.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        Aiso_dict: Isotropic hyperfine couplings ``A_iso`` (angular frequency units)
            by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_e1: Electronic correlation time entering the zero-frequency term (s).
        tau_e2: Electronic correlation time entering the finite-frequency term (s).
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to contact R2 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective angular momentum quantum number for the contact term
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 contact rates
    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (1 / 3) * Aiso**2 * S_eff * (S_eff + 1)
        spectral_density = J(0, tau_e1) + J(omega_I - omega_S, tau_e2)
        rate = prefactor * spectral_density
        rates[label] = rate
    return rates
