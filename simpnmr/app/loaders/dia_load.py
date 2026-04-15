# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load diamagnetic shifts from CSV or QC output.

Reads external data and returns plain mappings for application-level workflows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from simpnmr.core.const.isotopes import DEFAULT_ISOTOPES
from simpnmr.io.csv.csv_util import read_csv_safe
from simpnmr.io.qc import gateway as rdrs
from simpnmr.tools.coords import xyz_fmt as xyzf


def load_diamagnetic_shifts(
    file_name: str,
    file_type: str = "csv",
    ref_file_name: str | dict[str, str] = "",
    ref_file_type: str = "csv",
    ref_values: dict[str, float] | None = None,
) -> tuple[dict[str, float], str, Optional[dict[str, float]]]:
    """Load diamagnetic shifts and optional reference corrections.

    Args:
        file_name: Path to the diamagnetic shift file (molecule).
        file_type: Format of the main file — ``"csv"`` or ``"dft"``.
        ref_file_name: Reference file path(s). May be:
            - A single path string (all isotopes share one reference file).
            - A dict ``{isotope: path}`` (per-isotope reference files), e.g.
              ``{"1H": "tms_1h.log", "13C": "tms_13c.log"}``.
            Ignored when ``ref_values`` is provided.
        ref_file_type: Format of reference file(s) — ``"dft"`` or ``"csv"``.
            Ignored when ``ref_values`` is provided.
        ref_values: Explicit per-isotope reference shielding values, e.g.
            ``{"1H": 31.74, "13C": 188.07}``.  When provided, ``ref_file_name``
            and ``ref_file_type`` are ignored.

    Returns:
        dia_by_key: Mapping key -> diamagnetic shielding (or shift for CSV).
        key_kind: ``"atom_label"`` or ``"chem_label"``.
        ref_avg_by_isotope: Mapping isotope -> averaged reference shielding,
            or ``None`` when no reference is specified.  Used in
            ``Molecule.apply_diamagnetic_shifts`` to compute
            delta_dia = sigma_ref - sigma.
    """
    # --- main diamagnetic file ---
    if file_type == "csv":
        dia = read_csv_safe(file_name)
        if "shift" not in dia.columns:
            raise KeyError("Missing required column 'shift' in diamagnetic shift file")

        if "atom_label" in dia.columns:
            key_kind = "atom_label"
            dia_by_key = {
                str(k): float(v) for k, v in zip(dia["atom_label"], dia["shift"])
            }
        elif "chem_label" in dia.columns:
            key_kind = "chem_label"
            dia_by_key = {
                str(k): float(v) for k, v in zip(dia["chem_label"], dia["shift"])
            }
        else:
            raise KeyError(
                "atom_label or chem_label not present in diamagnetic shift file"
            )

    elif file_type == "dft":
        data = rdrs.QCCS.guess_from_file(file_name)

        labels = list(data.cs_iso.keys())
        labels_with_idx = xyzf.add_label_indices(labels)

        key_kind = "atom_label"
        dia_by_key = {
            str(lab_idx): float(val)
            for lab_idx, val in zip(labels_with_idx, data.cs_iso.values())
        }

    else:
        raise ValueError("Unknown file_type")

    # --- reference shielding (keyed by isotope string, e.g. "1H", "13C") ---
    ref_avg_by_isotope: Optional[dict[str, float]] = None

    if ref_values is not None:
        # Explicit values supplied directly — no file needed.
        ref_avg_by_isotope = {str(iso): float(v) for iso, v in ref_values.items()}

    elif isinstance(ref_file_name, dict):
        # Per-isotope file dict: {isotope: path}
        ref_avg_by_isotope = {}
        for isotope, path in ref_file_name.items():
            ref_avg_by_isotope[str(isotope)] = _avg_ref_shielding_from_file(
                path, ref_file_type
            )

    elif len(ref_file_name):
        # Single reference file — average per element, then map to isotope.
        avg_by_nn = _avg_ref_by_label_nn(ref_file_name, ref_file_type)
        ref_avg_by_isotope = _label_nn_to_isotope(avg_by_nn)

    return dia_by_key, key_kind, ref_avg_by_isotope


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _avg_ref_shielding_from_file(path: str, file_type: str) -> float:
    """Return the mean shielding of all atoms of the dominant element in *path*.

    Used for single-isotope per-isotope-file references (e.g. a TMS 1H
    calculation where you only care about the proton shieldings).
    Averages over *all* atoms in the file — for TMS that means all four
    equivalent methyl protons.
    """
    avg_by_nn = _avg_ref_by_label_nn(path, file_type)
    if len(avg_by_nn) == 1:
        return next(iter(avg_by_nn.values()))
    # Multiple element types: caller should use an isotope-keyed dict instead
    # of a per-isotope file.  Return the mean of all as a safe fallback and let
    # the caller's isotope key disambiguate.
    return float(np.mean(list(avg_by_nn.values())))


def _avg_ref_by_label_nn(path: str, file_type: str) -> dict[str, float]:
    """Return {label_nn: avg_shielding} from a reference file."""
    if file_type == "csv":
        ref = read_csv_safe(path)
        if "atom_label" not in ref.columns or "shift" not in ref.columns:
            raise KeyError(
                "Reference CSV must include 'atom_label' and 'shift' columns"
            )
        ref_labels_nn = xyzf.remove_label_indices(
            [str(x) for x in ref["atom_label"].tolist()]
        )
        ref = ref.copy()
        ref["atom_label_nn"] = ref_labels_nn
        grouped = ref.groupby("atom_label_nn")["shift"].mean()
        return {str(k): float(v) for k, v in grouped.items()}

    elif file_type == "dft":
        ref_data = rdrs.QCCS.guess_from_file(path)
        ref_labels = list(ref_data.cs_iso.keys())
        ref_labels_nn = xyzf.remove_label_indices(ref_labels)

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for lab, lab_nn in zip(ref_labels, ref_labels_nn):
            sums[lab_nn] = sums.get(lab_nn, 0.0) + float(ref_data.cs_iso[lab])
            counts[lab_nn] = counts.get(lab_nn, 0) + 1

        return {k: sums[k] / counts[k] for k in sums}

    else:
        raise ValueError(f"Unknown ref_file_type: {file_type!r}")


def _label_nn_to_isotope(avg_by_nn: dict[str, float]) -> dict[str, float]:
    """Convert {label_nn: value} → {isotope: value} using DEFAULT_ISOTOPES.

    Elements not in DEFAULT_ISOTOPES are kept with their label_nn as the key
    (harmless — the nucleus lookup in apply_diamagnetic_shifts will simply not
    find them and raise a clear KeyError).
    """
    return {
        DEFAULT_ISOTOPES.get(nn, nn): v for nn, v in avg_by_nn.items()
    }
