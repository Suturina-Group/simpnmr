# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Average hyperfine tensors and ⟨r⁻⁶⟩ over a conformer ensemble.

Reads N quantum-chemistry output files (same atom order, different geometries
and HFC values), computes population-weighted averages of the A tensors and the
inverse-sixth-power distances to the paramagnetic centre, and writes a
canonical molecule CSV that can be used directly as a ``method: csv`` hyperfine
input.

Coordinates in the output file are taken from the first (reference) conformer.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from simpnmr.app.loaders.mol_load import load_molecule_from_qca
from simpnmr.core.build.hfc import build_hfc_from_qca
from simpnmr.core.conv.freq_to_ang import a_tensor_mhz_to_ang
from simpnmr.core.domain.tensor import Hyperfine
from simpnmr.io.csv.mol import save_molecule_to_csv
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)

_R_INV6_COL = "r_inv6 (Å^-6)"


def run_average_conformers(
    files: list[str],
    centre_label: str,
    weights: list[float] | None = None,
    output: str | None = None,
    csv_delimiter: str = ",",
) -> int:
    """Average HFC tensors and ⟨r⁻⁶⟩ over a conformer ensemble.

    Args:
        files: Paths to QC output files, one per conformer. Atom order must be
            identical across all files.
        centre_label: Atom label of the paramagnetic centre (e.g. ``"Fe1"``).
            Used to locate the metal position in each conformer for r⁻⁶
            computation.
        weights: Population weights, one per conformer. Need not be normalised.
            Defaults to equal weights when ``None``.
        output: Output CSV file name. Defaults to
            ``conformer_avg_<stem>.csv`` where ``<stem>`` is the stem of the
            first input file.
        csv_delimiter: Delimiter used in the output CSV.

    Returns:
        Exit code (0 on success).

    Raises:
        ValueError: If atom labels are inconsistent across conformers, or if
            ``centre_label`` is not found in a conformer.
    """
    if len(files) < 2:
        raise ValueError(
            f"At least 2 conformer files are required; got {len(files)}."
        )

    # Normalise weights
    if weights is None:
        w = np.ones(len(files), dtype=float) / len(files)
    else:
        if len(weights) != len(files):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of files ({len(files)})."
            )
        w_arr = np.asarray(weights, dtype=float)
        if np.any(w_arr < 0):
            raise ValueError("All weights must be non-negative.")
        total = w_arr.sum()
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        w = w_arr / total

    logger.info(
        "Averaging %d conformers with weights: %s",
        len(files),
        ", ".join(f"{wi:.4f}" for wi in w),
    )

    # --- Load all QCA objects and validate label consistency ----------------
    qca_list = []
    for path in files:
        qca = rdrs.QCA.guess_from_file(path)
        qca_list.append(qca)
        logger.info("Loaded conformer: %s (%d atoms)", path, qca.n_atoms)

    ref_labels = [str(lab) for lab in qca_list[0].labels]
    for i, qca in enumerate(qca_list[1:], 1):
        labels_i = [str(lab) for lab in qca.labels]
        if labels_i != ref_labels:
            raise ValueError(
                f"Atom labels in conformer {i + 1} ({files[i]}) do not "
                f"match reference conformer ({files[0]}). "
                "All conformers must have the same atom order."
            )

    # --- Convert A tensors MHz → ppm Å⁻³ per conformer --------------------
    fc_converted: list[dict[str, np.ndarray]] = []
    sd_converted: list[dict[str, np.ndarray]] = []
    orb_converted: list[dict[str, np.ndarray | None]] = []

    for qca in qca_list:
        fc = {str(k): np.asarray(v, float) for k, v in qca.a_fc.items()}
        sd = {str(k): np.asarray(v, float) for k, v in qca.a_sd.items()}
        fc_converted.append(a_tensor_mhz_to_ang(fc))
        sd_converted.append(a_tensor_mhz_to_ang(sd))

        orb_raw = {str(k): v for k, v in qca.a_orb.items()}
        orb_present = {k: np.asarray(v, float) for k, v in orb_raw.items()
                       if v is not None}
        orb_present_conv = a_tensor_mhz_to_ang(orb_present)
        orb_converted.append(
            {k: orb_present_conv.get(k) for k in orb_raw}
        )

    has_orb = any(
        any(v is not None for v in orb.values())
        for orb in orb_converted
    )

    # --- Weighted average of A tensors -------------------------------------
    fc_avg: dict[str, np.ndarray] = {}
    sd_avg: dict[str, np.ndarray] = {}
    orb_avg: dict[str, np.ndarray] = {}

    for lab in ref_labels:
        fc_stack = []
        sd_stack = []
        orb_stack = []
        for i, (fc_i, sd_i, orb_i) in enumerate(
            zip(fc_converted, sd_converted, orb_converted)
        ):
            if lab in fc_i and lab in sd_i:
                fc_stack.append(w[i] * fc_i[lab])
                sd_stack.append(w[i] * sd_i[lab])
                if has_orb:
                    orb_t = orb_i.get(lab)
                    orb_stack.append(
                        w[i] * (orb_t if orb_t is not None
                                else np.zeros((3, 3)))
                    )

        if fc_stack:
            fc_avg[lab] = sum(fc_stack)
            sd_avg[lab] = sum(sd_stack)
            if has_orb:
                orb_avg[lab] = sum(orb_stack)

    # --- Compute weighted ⟨r⁻⁶⟩ per nucleus --------------------------------
    r_inv6_avg: dict[str, float] = {}

    for lab in ref_labels:
        if lab == centre_label:
            continue
        r6_sum = 0.0
        for i, qca in enumerate(qca_list):
            coords_i = np.asarray(qca.coords, dtype=float)
            labels_i = [str(l) for l in qca.labels]

            if centre_label not in labels_i:
                raise ValueError(
                    f"Paramagnetic centre '{centre_label}' not found in "
                    f"conformer {i + 1} ({files[i]})."
                )
            centre_idx = labels_i.index(centre_label)
            centre_coord = coords_i[centre_idx]

            nuc_idx = labels_i.index(lab)
            nuc_coord = coords_i[nuc_idx]

            r = float(np.linalg.norm(nuc_coord - centre_coord))
            if r < 1e-6:
                logger.warning(
                    "Nucleus '%s' is at the paramagnetic centre in "
                    "conformer %d — skipping r⁻⁶ for this nucleus.",
                    lab, i + 1,
                )
                r6_sum = float("nan")
                break
            r6_sum += w[i] / r**6

        if np.isfinite(r6_sum):
            r_inv6_avg[lab] = r6_sum

    # --- Build molecule from reference conformer and attach averaged HFCs --
    molecule = load_molecule_from_qca(files[0])

    # Assemble averaged Hyperfine objects
    hfc_by_label: dict[str, Hyperfine] = {}
    for lab in ref_labels:
        if lab not in fc_avg:
            continue
        hfc = Hyperfine()
        hfc.fc = fc_avg[lab]
        hfc.sd = sd_avg[lab]
        if has_orb and lab in orb_avg:
            hfc.orb = orb_avg[lab]
        hfc.tensor_full = hfc.fc + hfc.sd + (hfc.orb if has_orb else 0)
        if lab in r_inv6_avg:
            hfc.r_inv6 = r_inv6_avg[lab]
        hfc_by_label[lab] = hfc

    molecule.set_available_hfc_by_label(hfc_by_label)
    molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
        "available" if has_orb else "unavailable"
    )

    # --- Write output CSV --------------------------------------------------
    if output is None:
        stem = os.path.splitext(os.path.basename(files[0]))[0]
        output = f"conformer_avg_{stem}.csv"

    n_conformers = len(files)
    weight_str = " ".join(f"{wi:.6f}" for wi in w)
    comment = (
        f"# Conformer-averaged HFC: {n_conformers} conformers\n"
        f"# weights: {weight_str}\n"
        f"# reference geometry: {files[0]}\n"
        f"# paramagnetic centre: {centre_label}"
    )

    save_molecule_to_csv(
        molecule=molecule,
        file_name=output,
        verbose=True,
        comment=comment,
        delimiter=csv_delimiter,
    )

    return 0
