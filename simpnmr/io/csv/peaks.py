# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write relaxation analysis outputs as CSV.

Provides helpers to export relaxation decompositions and correlation-time fit data.
"""

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from simpnmr.io.csv.csv_util import write_csv_safe

logger = logging.getLogger(__name__)


def _average_by_chem_label(
    label_to_chem_label: dict[str, str],
    values_by_label: dict[str, float],
) -> dict[str, float]:
    """Average per-label values by chemical label."""
    grouped: dict[str, list[float]] = {}
    for label, value in values_by_label.items():
        chem_label = label_to_chem_label.get(label)
        if chem_label is None:
            continue
        if chem_label not in grouped:
            grouped[chem_label] = []
        grouped[chem_label].append(value)

    return {
        chem_label: float(np.mean(values)) for chem_label, values in grouped.items()
    }


def save_peak_data_to_csv(
    molecule,
    file_name: str,
    comment: str = "",
    verbose: bool = True,
    *,
    linewidth_by_label: Mapping[str, float] | None = None,
    linewidth_column_name: str = "linewidth_avg (ppm)",
) -> None:
    """Write peak linewidth and optional relaxation data to a CSV file.

    The function groups linewidths by chemical label and writes the resulting
    averages to CSV. Linewidths may be provided explicitly by the application
    pipeline or read from the molecule domain object.

    Args:
        molecule: Molecule domain object containing nuclei, linewidths, and
            optional relaxation data.
        file_name: Output CSV file path.
        comment: Optional comment line appended to the file header. If provided,
            it must begin with ``#`` (or will be prefixed automatically).
        verbose: If ``True``, prints the output file path.
        linewidth_by_label: Optional per-nucleus linewidth values in ppm to
            write instead of reading `nuc.shift.lw`.
        linewidth_column_name: Output column name for linewidth values.

    Returns:
        None.
    """
    label_to_chem_label = {nuc.label: nuc.chem_label for nuc in molecule.nuclei}
    # Map each chemical label to the isotope of the first nucleus that carries it
    chem_label_to_isotope: dict[str, str] = {}
    chem_label_count: dict[str, int] = {}
    for nuc in molecule.nuclei:
        if nuc.chem_label:
            if nuc.chem_label not in chem_label_to_isotope:
                chem_label_to_isotope[nuc.chem_label] = nuc.isotope
            chem_label_count[nuc.chem_label] = chem_label_count.get(nuc.chem_label, 0) + 1

    if linewidth_by_label is None:
        lw_by_label = {
            nuc.label: nuc.shift.lw
            for nuc in molecule.nuclei
            if nuc.shift.lw is not None
        }
    else:
        lw_by_label = dict(linewidth_by_label)

    relaxation = getattr(molecule, "relaxation", None)

    # Record the correlation times used (next to T / B0 in the header) since
    # the relaxation decomposition below is τ- and field-dependent.
    if relaxation is not None:
        _tau_R = getattr(relaxation, "tau_R", None)
        _tau_e1 = getattr(relaxation, "tau_e1", None)
        _tau_e2 = getattr(relaxation, "tau_e2", None)
        _tau_bits: list[str] = []
        if _tau_R is not None:
            _tau_bits.append(f"τ_R = {_tau_R * 1e12:.1f} ps")
        if _tau_e1 is not None and _tau_e2 is not None:
            if abs(_tau_e1 - _tau_e2) < 1e-18:
                _tau_bits.append(f"τ_e = {_tau_e1 * 1e12:.3f} ps")
            else:
                _tau_bits.append(
                    f"τ_e1 = {_tau_e1 * 1e12:.3f} ps, "
                    f"τ_e2 = {_tau_e2 * 1e12:.3f} ps"
                )
        elif _tau_e1 is not None:
            _tau_bits.append(f"τ_e = {_tau_e1 * 1e12:.3f} ps")
        if _tau_bits:
            _tau_str = ", ".join(_tau_bits)
            comment = f"{comment}, {_tau_str}" if comment else _tau_str

    r1_by_label = relaxation.r1.total if relaxation is not None else None
    dipolar_r1_by_label = relaxation.r1.dipolar if relaxation is not None else None
    contact_r1_by_label = relaxation.r1.contact if relaxation is not None else None
    curie_r1_by_label = relaxation.r1.curie if relaxation is not None else None
    # R2 is written as its decomposition (SBM dipolar/contact + Curie) rather
    # than the total, which is redundant with the linewidth column
    # (linewidth = R2 / (π |γ| B0)).
    _r2 = getattr(relaxation, "r2", None) if relaxation is not None else None
    dipolar_r2_by_label = _r2.dipolar if _r2 is not None else None
    contact_r2_by_label = _r2.contact if _r2 is not None else None
    curie_r2_by_label = _r2.curie if _r2 is not None else None

    # Split reported whenever the canonical susceptibility is g-corrected; the
    # spin-only reference is computed from S and T, not the susceptibility.
    has_fc_gcorr = getattr(molecule.susc, "iso_g_corr", None) is not None
    has_fc_spin_only = (
        getattr(molecule.susc, "iso_spin_only", None) is not None
        and getattr(molecule.susc, "iso_g_corr", None) is None
    )
    hyperfine_meta = molecule.metadata.get("hyperfine", {})
    has_orb = hyperfine_meta.get("orbital_contribution") == "available"

    if has_fc_gcorr:
        fc_column_name = "δ_fc_g_corr_avg (ppm)"
    elif has_fc_spin_only:
        fc_column_name = "δ_fc_spin_only_avg (ppm)"
    else:
        fc_column_name = "δ_fc_avg (ppm)"

    delta_total_avg_by_label = {}
    delta_dia_by_label = {}
    delta_pc_by_label = {}
    delta_fc_by_label = {}
    delta_fc_spin_only_by_label = {}
    delta_fc_g_corr_by_label = {}
    delta_orb_by_label = {}
    delta_orb_iso_by_label = {}
    delta_orb_aniso_by_label = {}

    for nuc in molecule.nuclei:
        if getattr(nuc.shift, "avg", None) is not None:
            delta_total_avg_by_label[nuc.label] = nuc.shift.avg
        if getattr(nuc.shift, "dia", None) is not None:
            delta_dia_by_label[nuc.label] = nuc.shift.dia
        if getattr(nuc.shift, "pc", None) is not None:
            delta_pc_by_label[nuc.label] = nuc.shift.pc
        if getattr(nuc.shift, "fc", None) is not None:
            delta_fc_by_label[nuc.label] = nuc.shift.fc
        if getattr(nuc.shift, "fc_spin_only", None) is not None:
            delta_fc_spin_only_by_label[nuc.label] = nuc.shift.fc_spin_only
        if getattr(nuc.shift, "fc_delta_g_corr", None) is not None:
            delta_fc_g_corr_by_label[nuc.label] = nuc.shift.fc_delta_g_corr
        if getattr(nuc.shift, "orb", None) is not None:
            delta_orb_by_label[nuc.label] = nuc.shift.orb
        if getattr(nuc.shift, "orb_iso", None) is not None:
            delta_orb_iso_by_label[nuc.label] = nuc.shift.orb_iso
        if getattr(nuc.shift, "orb_aniso", None) is not None:
            delta_orb_aniso_by_label[nuc.label] = nuc.shift.orb_aniso

    avg_r1_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, r1_by_label)
        if r1_by_label is not None
        else None
    )
    avg_dipolar_r2_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, dipolar_r2_by_label)
        if dipolar_r2_by_label is not None
        else None
    )
    avg_contact_r2_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, contact_r2_by_label)
        if contact_r2_by_label is not None
        else None
    )
    avg_curie_r2_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, curie_r2_by_label)
        if curie_r2_by_label is not None
        else None
    )

    avg_lw_by_chem_label = _average_by_chem_label(
        label_to_chem_label,
        lw_by_label,
    )

    avg_dipolar_by_chem_label = (
        _average_by_chem_label(
            label_to_chem_label,
            dipolar_r1_by_label,
        )
        if dipolar_r1_by_label is not None
        else None
    )
    avg_contact_by_chem_label = (
        _average_by_chem_label(
            label_to_chem_label,
            contact_r1_by_label,
        )
        if contact_r1_by_label is not None
        else None
    )
    avg_curie_by_chem_label = (
        _average_by_chem_label(
            label_to_chem_label,
            curie_r1_by_label,
        )
        if curie_r1_by_label is not None
        else None
    )
    avg_delta_total_avg_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_total_avg_by_label)
        if delta_total_avg_by_label
        else None
    )
    avg_delta_dia_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_dia_by_label)
        if delta_dia_by_label
        else None
    )
    avg_delta_pc_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_pc_by_label)
        if delta_pc_by_label
        else None
    )
    avg_delta_fc_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_fc_by_label)
        if delta_fc_by_label
        else None
    )
    avg_delta_fc_spin_only_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_fc_spin_only_by_label)
        if delta_fc_spin_only_by_label
        else None
    )
    avg_delta_fc_g_corr_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_fc_g_corr_by_label)
        if delta_fc_g_corr_by_label
        else None
    )
    avg_delta_orb_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_orb_by_label)
        if delta_orb_by_label
        else None
    )
    avg_delta_orb_iso_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_orb_iso_by_label)
        if delta_orb_iso_by_label
        else None
    )
    avg_delta_orb_aniso_by_chem_label = (
        _average_by_chem_label(label_to_chem_label, delta_orb_aniso_by_label)
        if delta_orb_aniso_by_label
        else None
    )

    # Collect the union of all chemical labels that appear in any available dict
    chem_labels: set[str] = set(avg_lw_by_chem_label.keys())

    if avg_r1_by_chem_label is not None:
        chem_labels |= set(avg_r1_by_chem_label.keys())
    if avg_dipolar_r2_by_chem_label is not None:
        chem_labels |= set(avg_dipolar_r2_by_chem_label.keys())
    if avg_contact_r2_by_chem_label is not None:
        chem_labels |= set(avg_contact_r2_by_chem_label.keys())
    if avg_curie_r2_by_chem_label is not None:
        chem_labels |= set(avg_curie_r2_by_chem_label.keys())
    if avg_dipolar_by_chem_label is not None:
        chem_labels |= set(avg_dipolar_by_chem_label.keys())
    if avg_contact_by_chem_label is not None:
        chem_labels |= set(avg_contact_by_chem_label.keys())
    if avg_curie_by_chem_label is not None:
        chem_labels |= set(avg_curie_by_chem_label.keys())
    if avg_delta_total_avg_by_chem_label is not None:
        chem_labels |= set(avg_delta_total_avg_by_chem_label.keys())
    if avg_delta_dia_by_chem_label is not None:
        chem_labels |= set(avg_delta_dia_by_chem_label.keys())
    if avg_delta_pc_by_chem_label is not None:
        chem_labels |= set(avg_delta_pc_by_chem_label.keys())
    if avg_delta_fc_by_chem_label is not None:
        chem_labels |= set(avg_delta_fc_by_chem_label.keys())
    if avg_delta_fc_spin_only_by_chem_label is not None:
        chem_labels |= set(avg_delta_fc_spin_only_by_chem_label.keys())
    if avg_delta_fc_g_corr_by_chem_label is not None:
        chem_labels |= set(avg_delta_fc_g_corr_by_chem_label.keys())
    if avg_delta_orb_by_chem_label is not None:
        chem_labels |= set(avg_delta_orb_by_chem_label.keys())
    if avg_delta_orb_iso_by_chem_label is not None:
        chem_labels |= set(avg_delta_orb_iso_by_chem_label.keys())
    if avg_delta_orb_aniso_by_chem_label is not None:
        chem_labels |= set(avg_delta_orb_aniso_by_chem_label.keys())

    chem_labels = sorted(chem_labels)

    # Base columns: chem_label, isotope, count, then shift columns, linewidth, R1/R2
    out: dict[str, list] = {
        "chem_label": chem_labels,
        "isotope": [chem_label_to_isotope.get(lbl, "") for lbl in chem_labels],
        "count": [chem_label_count.get(lbl, 0) for lbl in chem_labels],
    }

    if avg_delta_total_avg_by_chem_label is not None:
        out["δ_total_avg (ppm)"] = [
            avg_delta_total_avg_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_delta_dia_by_chem_label is not None:
        out["δ_dia_avg (ppm)"] = [
            avg_delta_dia_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_delta_pc_by_chem_label is not None:
        out["δ_pc_avg (ppm)"] = [
            avg_delta_pc_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_delta_fc_by_chem_label is not None:
        out[fc_column_name] = [
            avg_delta_fc_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    # Only write the dedicated spin-only column when the FC column above was
    # named something else (has_fc_gcorr) — otherwise we'd get two columns
    # with the same name "δ_fc_spin_only_avg (ppm)".
    if avg_delta_fc_spin_only_by_chem_label is not None and not has_fc_spin_only:
        out["δ_fc_spin_only_avg (ppm)"] = [
            avg_delta_fc_spin_only_by_chem_label.get(lbl, np.nan)
            for lbl in chem_labels
        ]
    if avg_delta_fc_g_corr_by_chem_label is not None:
        out["Δδ_fc_g_corr_avg (ppm)"] = [
            avg_delta_fc_g_corr_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if has_orb and avg_delta_orb_by_chem_label is not None:
        out["δ_orb_avg (ppm)"] = [
            avg_delta_orb_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if has_orb and avg_delta_orb_iso_by_chem_label is not None:
        out["δ_orb_iso_avg (ppm)"] = [
            avg_delta_orb_iso_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if has_orb and avg_delta_orb_aniso_by_chem_label is not None:
        out["δ_orb_aniso_avg (ppm)"] = [
            avg_delta_orb_aniso_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]

    out[linewidth_column_name] = [
        avg_lw_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
    ]

    if avg_r1_by_chem_label is not None:
        out["R1_total (s^-1)"] = [
            avg_r1_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]

    # Optional decompositions. R1 is written as total + components; R2 is
    # written as components only (its total is the linewidth column).
    if avg_dipolar_by_chem_label is not None:
        out["R1_sbm_dipolar (s^-1)"] = [
            avg_dipolar_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_contact_by_chem_label is not None:
        out["R1_sbm_contact (s^-1)"] = [
            avg_contact_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_curie_by_chem_label is not None:
        out["R1_curie (s^-1)"] = [
            avg_curie_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_dipolar_r2_by_chem_label is not None:
        out["R2_sbm_dipolar (s^-1)"] = [
            avg_dipolar_r2_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_contact_r2_by_chem_label is not None:
        out["R2_sbm_contact (s^-1)"] = [
            avg_contact_r2_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_curie_r2_by_chem_label is not None:
        out["R2_curie (s^-1)"] = [
            avg_curie_r2_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]

    df = pd.DataFrame(data=out)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("pNMR data written to %s", file_name)
