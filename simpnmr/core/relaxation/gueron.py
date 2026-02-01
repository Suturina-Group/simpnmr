# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute Guéron Curie relaxation rates.

Provides helpers to evaluate point-dipole Curie R1 and R2 rates for labelled nuclei.
"""

import numpy as np

from simpnmr.core.constants.physics import KB, MU0, MUB
from simpnmr.core.factories.eff_factors import calc_g_eff, choose_S_eff


def calc_r1_curie(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    omega_I_dict,
    T,
    tau_R,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes Guéron Curie R1 relaxation rates for each nucleus.

    Implements the Guéron expression for longitudinal relaxation due to the Curie
    (static) dipolar interaction in the point-dipole approximation.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        T: Temperature in Kelvin.
        tau_R: Rotational correlation time (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to Curie R1 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the Curie term
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 Curie rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        omega_I = omega_I_dict[label]
        prefactor = (
            (2 / 5)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (omega_I / (3 * KB * T)) ** 2
            * (g_eff * MUB) ** 4
            * (S_eff * (S_eff + 1)) ** 2
        )
        spectral_density = 3 * J(omega_I, tau_R)
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def calc_r2_curie(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    omega_I_dict,
    T,
    tau_R,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes Guéron Curie R2 relaxation rates for each nucleus.

    Implements the Guéron expression for transverse relaxation due to the Curie
    (static) dipolar interaction in the point-dipole approximation.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        T: Temperature in Kelvin.
        tau_R: Rotational correlation time (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to Curie R2 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the Curie term
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 Curie rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 5)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (omega_I / (3 * KB * T)) ** 2
            * (g_eff * MUB) ** 4
            * (S_eff * (S_eff + 1)) ** 2
        )
        spectral_density = 4 * J(0, tau_R) + 3 * J(omega_I, tau_R)
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates
