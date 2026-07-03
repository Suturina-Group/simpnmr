# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Workflow-level policies for relaxation-rate post-processing.

This module contains policy helpers that transform canonical per-nucleus
relaxation outputs into workflow-specific representations. These helpers do not
change the underlying relaxation physics; they define how application-level
workflows choose to aggregate, project, or otherwise consume the evaluated
rates.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.conv.ang_to_freq import angstrom_to_mhz
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.relaxation.eval import evaluate_relaxation_rates


def apply_relaxation_decomposition(
    molecule: Molecule,
    *,
    relaxation_model: str,
    temperature: float,
    magnetic_field_tesla: float,
    tau_R: float,
    tau_e1: float,
    tau_e2: float,
    hyperfine_method: str | None = None,
    min_linewidth_hz: float = 0.0,
) -> None:
    """Evaluate the R1/R2 relaxation decomposition from explicit correlation
    times and store it on the molecule.

    Computes per-nucleus R1 and R2 channels (SBM dipolar/contact + Curie, per
    ``relaxation_model``) for every nucleus in ``molecule`` using the supplied
    rotational (``tau_R``) and electronic (``tau_e1`` = T1e, ``tau_e2`` = T2e)
    correlation times, stores the result on ``molecule.relaxation``, and writes
    the resulting R2-derived linewidths to ``nuc.shift.lw`` (ppm).

    This is the shared core used by workflows that already know their
    correlation times (e.g. the susceptibility fit, which derives ``tau_e``
    from its r⁻⁶ relaxation fits). It performs no τ estimation of its own.

    Args:
        molecule: Molecule with nuclei, coordinates, hyperfines, electronic
            state and ``paramagnetic_centre`` set. Updated in place.
        relaxation_model: ``"sbm"``, ``"curie"`` or ``"sbm curie"``.
        temperature: Temperature (K).
        magnetic_field_tesla: Static field B0 (T).
        tau_R: Rotational correlation time (s).
        tau_e1: Electronic T1e (s).
        tau_e2: Electronic T2e (s).
        hyperfine_method: When ``"pdip"``, contact A_iso is taken as zero.
        min_linewidth_hz: Floor added to each linewidth (Hz) before conversion.
    """
    B0 = float(magnetic_field_tesla)
    nuclei_coords = {nuc.label: nuc.coord for nuc in molecule.nuclei}

    if hyperfine_method == "pdip":
        A_iso_dict = {label: 0.0 for label in nuclei_coords}
    else:
        A_iso_dict = {
            nuc.label: float(
                angstrom_to_mhz(
                    1.0 / 3.0 * np.trace(nuc.A.fc),
                    nuclear_gamma=get_nuclear_gamma(nuc.isotope),
                )
            ) * 1e6
            for nuc in molecule.nuclei
        }

    gamma_I_dict = {
        nuc.label: get_nuclear_gamma(nuc.isotope) * 2 * np.pi * 1e6
        for nuc in molecule.nuclei
    }
    omega_I_dict = {label: gamma_I_dict[label] * B0 for label in nuclei_coords}
    omega_S = EGAMMA * B0 * 2 * np.pi * 1e6

    tau_c1 = 1.0 / (1.0 / tau_R + 1.0 / tau_e1)
    tau_c2 = 1.0 / (1.0 / tau_R + 1.0 / tau_e2)

    relaxation_eval = evaluate_relaxation_rates(
        relaxation_model=relaxation_model,
        nuclei_coords=nuclei_coords,
        electron_coords=molecule.paramagnetic_centre,
        gamma_I_dict=gamma_I_dict,
        omega_I_dict=omega_I_dict,
        omega_S=omega_S,
        spin=molecule.electronic.spin_S,
        orbit=molecule.electronic.orbit_L,
        total_momentum_J=molecule.electronic.total_J,
        A_iso_dict=A_iso_dict,
        temperature=temperature,
        tau_R=tau_R,
        tau_c1=tau_c1,
        tau_c2=tau_c2,
        tau_e1=tau_e1,
        tau_e2=tau_e2,
        compute_r1=True,
        compute_r2=True,
    )
    relaxation_eval.tau_R = tau_R
    relaxation_eval.tau_e1 = tau_e1
    relaxation_eval.tau_e2 = tau_e2

    # Without a usable R2 channel there are no linewidths to store. Leave
    # ``molecule.relaxation`` unset so callers keep their existing linewidth
    # source (e.g. the r⁻⁶ fit) rather than seeing a half-populated state.
    rates_r2 = relaxation_eval.r2.total if relaxation_eval.r2 else None
    if not rates_r2:
        return

    # R2-derived linewidths (ppm), averaged over equivalent nuclei.
    r2_by_chem_label: dict[str, list[float]] = defaultdict(list)
    for nuc in molecule.nuclei:
        if nuc.label in rates_r2:
            r2_by_chem_label[nuc.chem_label].append(rates_r2[nuc.label])
    avg_lw_hz_by_chem_label = {
        cl: float(np.mean([r / np.pi for r in rates]))
        for cl, rates in r2_by_chem_label.items()
    }
    for nuc in molecule.nuclei:
        if nuc.chem_label in avg_lw_hz_by_chem_label:
            larmor_hz = abs(omega_I_dict[nuc.label]) / (2 * np.pi)
            lw_hz = avg_lw_hz_by_chem_label[nuc.chem_label] + min_linewidth_hz
            nuc.shift.lw = lw_hz / larmor_hz * 1e6

    # Only commit the relaxation object once every nucleus carries a linewidth,
    # so downstream linewidth resolution never sees an incomplete state.
    if all(nuc.shift.lw is not None for nuc in molecule.nuclei):
        molecule.relaxation = relaxation_eval


def average_relaxation_rates_by_chem_label(
    molecule: Molecule,
    rates_by_label: dict[str, float] | None,
) -> dict[str, float] | None:
    """Average per-nucleus relaxation rates by chemical label.

    This helper projects atom-label-indexed relaxation rates onto the
    chemical-label representation used by higher-level workflows such as
    linewidth prediction and correlation-time fitting.

    Args:
        molecule: Molecule providing the ``label`` to ``chem_label`` mapping.
        rates_by_label: Optional mapping from atom label to relaxation rate.

    Returns:
        Mapping from chemical label to averaged relaxation rate, or ``None``
        when no channel data is available.
    """
    if rates_by_label is None:
        return None

    grouped_rates = defaultdict(list)
    for nuc in molecule.nuclei:
        if nuc.label in rates_by_label:
            grouped_rates[nuc.chem_label].append(rates_by_label[nuc.label])

    return {
        chem_label: np.mean(rate_list)
        for chem_label, rate_list in grouped_rates.items()
    }


def resolve_relaxation_conditions(
    config,
    experiment,
) -> tuple[float | None, float | None]:
    """Resolve workflow-level relaxation temperature and magnetic field.

    Application workflows may provide optional relaxation overrides in the
    config. When present, these values take precedence. Otherwise, the values
    are taken from the paired experiment object when available.

    Args:
        config: Workflow config object that may define
            ``relaxation_temperature`` and
            ``relaxation_magnetic_field_tesla`` overrides.
        experiment: Optional experiment object that may provide
            ``temperature`` and ``magnetic_field`` values.

    Returns:
        Tuple ``(temperature, magnetic_field_tesla)`` resolved using the
        precedence ``config override > experiment value > None``.
    """
    temperature = getattr(config, "relaxation_temperature", None)
    magnetic_field_tesla = getattr(config, "relaxation_magnetic_field_tesla", None)

    if temperature is None and experiment is not None:
        temperature = experiment.temperature
    if magnetic_field_tesla is None and experiment is not None:
        magnetic_field_tesla = experiment.magnetic_field

    return temperature, magnetic_field_tesla
