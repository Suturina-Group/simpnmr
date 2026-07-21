# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write molecule CSV files for structure, labels, and pNMR tensors.

Provides CSV parsing and serialization helpers for atom labels, coordinates,
optional chemical labels, canonical split hyperfine tensors, and optional
orbital hyperfine tensors.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from simpnmr.io.csv.csv_util import (
    format_full_precision,
    read_csv_safe,
    write_csv_safe,
)

logger = logging.getLogger(__name__)


def read_molecule_csv(file_name: str) -> dict:
    """Read a molecule CSV file into raw structure, tensor, and label payloads.

    The reader accepts structure-only CSV files as well as canonical hyperfine
    CSV files written with canonical molecule CSV headers. Canonical
    spin-hyperfine columns are reconstructed into full tensors and returned in
    ``tensors``. Canonical orbital hyperfine columns, when present, are
    reconstructed and returned in ``orb_tensors``.

    Returns:
        Dictionary with molecule CSV payload fields:
            - ``labels``: Atom labels in row order.
            - ``coords``: Cartesian coordinates with shape ``(n, 3)``.
            - ``tensors``: Reconstructed full hyperfine tensors or ``None``.
            - ``orb_tensors``: Reconstructed orbital tensors or ``None``.
            - ``chem_labels``: Chemical labels or ``None``.
            - ``chem_math_labels``: Chemical math labels or ``None``.
    """
    data = read_csv_safe(file_name)

    required_cols = ["atom_label ()", "x (Å)", "y (Å)", "z (Å)"]
    split_hyperfine_cols = [
        "A_fc_iso (ppm Å^-3)",
        "A_sd_xx (ppm Å^-3)",
        "A_sd_xy (ppm Å^-3)",
        "A_sd_xz (ppm Å^-3)",
        "A_sd_yy (ppm Å^-3)",
        "A_sd_yz (ppm Å^-3)",
        "A_sd_zz (ppm Å^-3)",
    ]
    orb_hyperfine_cols = [
        "A_orb_xx (ppm Å^-3)",
        "A_orb_xy (ppm Å^-3)",
        "A_orb_xz (ppm Å^-3)",
        "A_orb_yy (ppm Å^-3)",
        "A_orb_yz (ppm Å^-3)",
        "A_orb_zz (ppm Å^-3)",
    ]

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing header(s) {missing} in {file_name}")

    # Detect canonical hyperfine payloads
    has_hyperfine = all(col in data.columns for col in split_hyperfine_cols)
    has_orb = all(col in data.columns for col in orb_hyperfine_cols)

    labels = data["atom_label ()"].tolist()

    coords = np.array([data["x (Å)"], data["y (Å)"], data["z (Å)"]]).T

    if has_hyperfine:
        aiso_key = "A_fc_iso (ppm Å^-3)"
        xx_key = "A_sd_xx (ppm Å^-3)"
        xy_key = "A_sd_xy (ppm Å^-3)"
        xz_key = "A_sd_xz (ppm Å^-3)"
        yy_key = "A_sd_yy (ppm Å^-3)"
        yz_key = "A_sd_yz (ppm Å^-3)"
        zz_key = "A_sd_zz (ppm Å^-3)"

        tensors = [
            np.array(
                [
                    [
                        row[xx_key],
                        row[xy_key],
                        row[xz_key],
                    ],
                    [
                        row[xy_key],
                        row[yy_key],
                        row[yz_key],
                    ],
                    [
                        row[xz_key],
                        row[yz_key],
                        row[zz_key],
                    ],
                ],
                dtype=float,
            )
            + np.eye(3) * float(row[aiso_key])
            for _, row in data.iterrows()
        ]
    else:
        tensors = None

    if has_orb:
        orb_tensors = [
            np.array(
                [
                    [
                        row["A_orb_xx (ppm Å^-3)"],
                        row["A_orb_xy (ppm Å^-3)"],
                        row["A_orb_xz (ppm Å^-3)"],
                    ],
                    [
                        row["A_orb_xy (ppm Å^-3)"],
                        row["A_orb_yy (ppm Å^-3)"],
                        row["A_orb_yz (ppm Å^-3)"],
                    ],
                    [
                        row["A_orb_xz (ppm Å^-3)"],
                        row["A_orb_yz (ppm Å^-3)"],
                        row["A_orb_zz (ppm Å^-3)"],
                    ],
                ],
                dtype=float,
            )
            for _, row in data.iterrows()
        ]
    else:
        orb_tensors = None

    chem_labels = None
    chem_math_labels = None

    if "chem_label ()" in data.columns:
        chem_labels = data["chem_label ()"].tolist()

    if "chem_math_label ()" in data.columns:
        chem_math_labels = data["chem_math_label ()"].tolist()

    r_inv6 = None
    if "r_inv6 (Å^-6)" in data.columns:
        r_inv6 = data["r_inv6 (Å^-6)"].tolist()

    return {
        "labels": labels,
        "coords": coords,
        "tensors": tensors,
        "orb_tensors": orb_tensors,
        "chem_labels": chem_labels,
        "chem_math_labels": chem_math_labels,
        "r_inv6": r_inv6,
    }


def save_molecule_to_csv(
    molecule,
    file_name: str = "molecule.csv",
    verbose: bool = True,
    comment: str = "",
    delimiter: str = ",",
) -> None:
    """Write molecule structure, tensor, and shift data to a CSV file.

    Produces the canonical SimpNMR molecule CSV, which can be read back with
    :func:`read_molecule_csv` and used as a ``method: csv`` hyperfine input.

    **Always-present columns**

    - ``atom_label ()`` — atom label as it appears in the QC output (e.g.
      ``H1``, ``C3``).
    - ``chem_label ()`` — chemical/symmetry label (e.g. ``Ha``, ``C_ring``);
      ``NaN`` when not assigned.
    - ``x (Å)``, ``y (Å)``, ``z (Å)`` — Cartesian coordinates in Ångströms.

    **Spin hyperfine columns** (present when hyperfine data are available)

    The full A-tensor is split into its isotropic Fermi-contact part and the
    traceless spin-dipole (anisotropic) part:

    - ``A_fc_iso (ppm Å⁻³)`` — isotropic Fermi-contact value, ⅓ Tr[**A**_FC].
    - ``A_sd_xx``, ``A_sd_xy``, ``A_sd_xz``, ``A_sd_yy``, ``A_sd_yz``,
      ``A_sd_zz`` (all ``ppm Å⁻³``) — unique elements of the symmetric
      spin-dipole tensor.

    When read back, the full tensor is reconstructed as
    **A** = **A**_SD + *A*_fc_iso · **I**.

    **Orbital hyperfine columns** (present only when orbital contribution is
    available, i.e. the source file contained an ``A(ORB)`` block)

    - ``A_orb_xx``, ``A_orb_xy``, ``A_orb_xz``, ``A_orb_yy``, ``A_orb_yz``,
      ``A_orb_zz`` (all ``ppm Å⁻³``) — unique elements of the symmetric
      orbital hyperfine tensor.

    **Shift columns** (present after shifts have been computed)

    - ``δ_total (ppm)`` — total paramagnetic shift.
    - ``δ_dia (ppm)`` — diamagnetic reference shift.
    - ``δ_pc (ppm)`` — pseudocontact shift.
    - ``δ_fc (ppm)`` / ``δ_fc_g_corr (ppm)`` / ``δ_fc_spin_only (ppm)`` —
      Fermi-contact shift; column name reflects the correction applied.
    - ``δ_fc_spin_only (ppm)``, ``Δδ_fc_g_corr (ppm)`` — additional
      spin-only and g-correction breakdown columns, present when g-corrected
      FC shifts were computed.
    - ``δ_orb (ppm)``, ``δ_orb_iso (ppm)``, ``δ_orb_aniso (ppm)`` — orbital
      shift and its iso/aniso decomposition; present only when orbital
      hyperfine data are available.

    **File format notes**

    - Comment lines beginning with ``#`` are written at the top and skipped on
      re-read.
    - If the molecule was rotated into the chi frame a ``# Coordinates and
      hyperfine tensors in this file are stored in the chi frame.`` comment
      is appended automatically.

    Args:
        molecule: Molecule-like domain object to serialize.
        file_name: Output CSV file name.
        verbose: If True, log the output file path.
        comment: Optional additional comment line to prepend to the file.
        delimiter: CSV delimiter.
    """

    df = _build_molecule_df(molecule)

    frame_value = molecule.metadata.get("frame")
    merged_comment = comment
    if frame_value == "chi":
        frame_comment = (
            "# Coordinates and hyperfine tensors in "
            "this file are stored in the chi frame."
        )
        merged_comment = frame_comment if not comment else f"{comment}\n{frame_comment}"

    # Keep r⁻⁶ values at full precision (they span many orders of magnitude and
    # would lose digits under the table-wide float_format); other columns are
    # unaffected.
    if "r_inv6 (Å^-6)" in df.columns:
        df["r_inv6 (Å^-6)"] = format_full_precision(df["r_inv6 (Å^-6)"])

    write_csv_safe(df, file_name, merged_comment)

    if verbose:
        logger.info("Molecule data written to %s", file_name)

    return


def _build_molecule_df(molecule):
    """Build the canonical molecule export table for CSV serialization."""

    nuc_by_label = {nuc.label: nuc for nuc in molecule.nuclei}
    hfc_by_label = molecule.available_hfc_by_label

    hyperfine_meta = molecule.metadata.get("hyperfine", {})
    has_orb = hyperfine_meta.get("orbital_contribution") == "available"
    has_r_inv6 = any(
        hfc_by_label.get(lab) is not None
        and hfc_by_label[lab].r_inv6 is not None
        for lab in (nuc.label for nuc in molecule.nuclei)
    )
    # The spin-only / g-correction split is computed directly from S and T, so
    # it is reported whenever the canonical susceptibility is g-corrected —
    # including fit-derived susceptibilities that carry no spin-only channel.
    has_fc_gcorr = getattr(molecule.susc, "iso_g_corr", None) is not None
    has_fc_spin_only = (
        getattr(molecule.susc, "iso_spin_only", None) is not None
        and getattr(molecule.susc, "iso_g_corr", None) is None
    )
    if has_fc_gcorr:
        fc_column_name = "δ_fc_g_corr (ppm)"
    elif has_fc_spin_only:
        fc_column_name = "δ_fc_spin_only (ppm)"
    else:
        fc_column_name = "δ_fc (ppm)"

    labels = list(molecule.labels)
    coords = np.asarray(molecule.coords)

    if len(labels) != len(coords):
        raise ValueError("Molecule labels and coordinates must have matching lengths")

    base_specs = [
        ("atom_label ()", lambda ctx: ctx["label"]),
        (
            "chem_label ()",
            lambda ctx: ctx["nuc"].chem_label if ctx["nuc"] is not None else np.nan,
        ),
        ("x (Å)", lambda ctx: ctx["coord"][0]),
        ("y (Å)", lambda ctx: ctx["coord"][1]),
        ("z (Å)", lambda ctx: ctx["coord"][2]),
        (
            "A_fc_iso (ppm Å^-3)",
            lambda ctx: 1.0 / 3.0 * np.trace(ctx["hfc"].fc)
            if ctx["hfc"] is not None
            else np.nan,
        ),
        (
            "A_sd_xx (ppm Å^-3)",
            lambda ctx: ctx["hfc"].sd[0, 0] if ctx["hfc"] is not None else np.nan,
        ),
        (
            "A_sd_xy (ppm Å^-3)",
            lambda ctx: ctx["hfc"].sd[0, 1] if ctx["hfc"] is not None else np.nan,
        ),
        (
            "A_sd_xz (ppm Å^-3)",
            lambda ctx: ctx["hfc"].sd[0, 2] if ctx["hfc"] is not None else np.nan,
        ),
        (
            "A_sd_yy (ppm Å^-3)",
            lambda ctx: ctx["hfc"].sd[1, 1] if ctx["hfc"] is not None else np.nan,
        ),
        (
            "A_sd_yz (ppm Å^-3)",
            lambda ctx: ctx["hfc"].sd[1, 2] if ctx["hfc"] is not None else np.nan,
        ),
        (
            "A_sd_zz (ppm Å^-3)",
            lambda ctx: ctx["hfc"].sd[2, 2] if ctx["hfc"] is not None else np.nan,
        ),
        *(
            [
                (
                    "r_inv6 (Å^-6)",
                    lambda ctx: ctx["hfc"].r_inv6
                    if ctx["hfc"] is not None and ctx["hfc"].r_inv6 is not None
                    else np.nan,
                ),
            ]
            if has_r_inv6
            else []
        ),
        *(
            [
                (
                    "A_orb_xx (ppm Å^-3)",
                    lambda ctx: ctx["hfc"].orb[0, 0]
                    if ctx["hfc"] is not None
                    else np.nan,
                ),
                (
                    "A_orb_xy (ppm Å^-3)",
                    lambda ctx: ctx["hfc"].orb[0, 1]
                    if ctx["hfc"] is not None
                    else np.nan,
                ),
                (
                    "A_orb_xz (ppm Å^-3)",
                    lambda ctx: ctx["hfc"].orb[0, 2]
                    if ctx["hfc"] is not None
                    else np.nan,
                ),
                (
                    "A_orb_yy (ppm Å^-3)",
                    lambda ctx: ctx["hfc"].orb[1, 1]
                    if ctx["hfc"] is not None
                    else np.nan,
                ),
                (
                    "A_orb_yz (ppm Å^-3)",
                    lambda ctx: ctx["hfc"].orb[1, 2]
                    if ctx["hfc"] is not None
                    else np.nan,
                ),
                (
                    "A_orb_zz (ppm Å^-3)",
                    lambda ctx: ctx["hfc"].orb[2, 2]
                    if ctx["hfc"] is not None
                    else np.nan,
                ),
            ]
            if has_orb
            else []
        ),
        (
            "δ_total (ppm)",
            lambda ctx: ctx["nuc"].shift.total if ctx["nuc"] is not None else np.nan,
        ),
        (
            "δ_dia (ppm)",
            lambda ctx: ctx["nuc"].shift.dia if ctx["nuc"] is not None else np.nan,
        ),
        (
            "δ_pc (ppm)",
            lambda ctx: ctx["nuc"].shift.pc if ctx["nuc"] is not None else np.nan,
        ),
        (
            fc_column_name,
            lambda ctx: ctx["nuc"].shift.fc if ctx["nuc"] is not None else np.nan,
        ),
        *(
            [
                (
                    "δ_fc_spin_only (ppm)",
                    lambda ctx: ctx["nuc"].shift.fc_spin_only
                    if ctx["nuc"] is not None
                    else np.nan,
                ),
                (
                    "Δδ_fc_g_corr (ppm)",
                    lambda ctx: ctx["nuc"].shift.fc_delta_g_corr
                    if ctx["nuc"] is not None
                    else np.nan,
                ),
            ]
            if has_fc_gcorr
            else []
        ),
        # Full paramagnetic shift tensor (raw 3x3, non-symmetric): fc + pc + orb.
        # Its trace/3 equals δ_total − δ_dia.
        *[
            (
                f"δ_para_{a}{b} (ppm)",
                lambda ctx, i=i, j=j: (
                    ctx["nuc"].shift.paramag_tensor[i, j]
                    if ctx["nuc"] is not None
                    else np.nan
                ),
            )
            for a, i in (("x", 0), ("y", 1), ("z", 2))
            for b, j in (("x", 0), ("y", 1), ("z", 2))
        ],
    ]

    orb_specs = [
        (
            "δ_orb (ppm)",
            lambda ctx: getattr(ctx["nuc"].shift, "orb", np.nan)
            if ctx["nuc"] is not None
            else np.nan,
        ),
        (
            "δ_orb_iso (ppm)",
            lambda ctx: getattr(ctx["nuc"].shift, "orb_iso", np.nan)
            if ctx["nuc"] is not None
            else np.nan,
        ),
        (
            "δ_orb_aniso (ppm)",
            lambda ctx: getattr(ctx["nuc"].shift, "orb_aniso", np.nan)
            if ctx["nuc"] is not None
            else np.nan,
        ),
    ]

    spec_groups = [
        base_specs,
        orb_specs if has_orb else [],
    ]
    specs = [spec for group in spec_groups for spec in group]

    rows: list[dict[str, object]] = []
    for label, coord in zip(labels, coords):
        ctx = {
            "label": label,
            "coord": coord,
            "nuc": nuc_by_label.get(label),
            "hfc": hfc_by_label.get(label),
        }
        rows.append({name: getter(ctx) for name, getter in specs})

    columns = [name for name, _ in specs]
    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    return df.reset_index(drop=True)
