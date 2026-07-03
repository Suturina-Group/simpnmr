# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Builders for assembling hyperfine data and attaching it to a Molecule."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simpnmr.core.conv.freq_to_ang import a_tensor_mhz_to_ang
from simpnmr.core.domain.mol import Hyperfine, Molecule

logger = logging.getLogger(__name__)


def _assemble_hfc_from_components(
    *,
    fc: np.ndarray,
    sd: np.ndarray,
    orb: np.ndarray | None,
    tensor_full: np.ndarray | None,
    orbital_contribution: str,
    label: str,
) -> Hyperfine:
    """Assemble a canonical `Hyperfine` object from decomposed QC components.

    This helper finalizes one label-indexed HFC payload from already parsed
    physical components. It is the single assembly point used before the
    canonical HFC map is stored on `Molecule`.

    Args:
        fc: Physical Fermi-contact hyperfine tensor for one atom label.
        sd: Physical traceless spin-dipolar hyperfine tensor for one atom
            label.
        orb: Physical orbital hyperfine tensor for one atom label, if
            available.
        tensor_full: Physical full hyperfine tensor for one atom label, if
            available.
        orbital_contribution: Orbital contribution policy. When set to
            ``"on"``, missing orbital data is treated as an error.
        label: Atom label used only for validation/error messages.

    Returns:
        A finalized `Hyperfine` object for the supplied atom label.

    Raises:
        ValueError: If any provided tensor has invalid shape, or if orbital
            contribution is required but orbital data is missing.
    """
    fc_arr = np.asarray(fc, dtype=float)
    sd_arr = np.asarray(sd, dtype=float)

    if fc_arr.shape != (3, 3):
        raise ValueError(f"A(FC) tensor for {label} must be (3,3), got {fc_arr.shape}")
    if sd_arr.shape != (3, 3):
        raise ValueError(f"A(SD) tensor for {label} must be (3,3), got {sd_arr.shape}")

    if orb is None:
        if orbital_contribution == "on":
            raise ValueError(
                f"A(ORB) contribution requested but missing for label {label}"
            )
        orb_arr = np.zeros((3, 3), dtype=float)
    else:
        orb_arr = np.asarray(orb, dtype=float)
        if orb_arr.shape != (3, 3):
            raise ValueError(
                f"A(ORB) tensor for {label} must be (3,3), got {orb_arr.shape}"
            )

    full_arr = None
    if tensor_full is not None:
        full_arr = np.asarray(tensor_full, dtype=float)
        if full_arr.shape != (3, 3):
            raise ValueError(
                f"Full hyperfine tensor for {label} must be (3,3), got {full_arr.shape}"
            )

    hfc = Hyperfine()
    hfc.fc = fc_arr
    hfc.sd = sd_arr
    hfc.orb = orb_arr
    hfc.tensor_full = full_arr
    return hfc


def _assemble_hfc_from_full_tensor(*, tensor_full: np.ndarray, label: str) -> Hyperfine:
    """Assemble a canonical `Hyperfine` object from a full hyperfine tensor.

    This helper is used for CSV-derived HFC payloads where only the full tensor
    is available. The isotropic part is mapped to `fc`, the deviatoric part is
    mapped to `sd`, and orbital contribution is currently represented as a zero
    tensor because CSV-side orbital decomposition is not yet part of the
    contract.

    Args:
        tensor_full: Physical full hyperfine tensor for one atom label.
        label: Atom label used only for validation/error messages.

    Returns:
        A finalized `Hyperfine` object for the supplied atom label.

    Raises:
        ValueError: If the supplied full tensor is not shaped ``(3, 3)``.
    """
    full_arr = np.asarray(tensor_full, dtype=float)
    if full_arr.shape != (3, 3):
        raise ValueError(
            f"Hyperfine tensor for {label} must be (3,3), got {full_arr.shape}"
        )

    iso = float(np.trace(full_arr) / 3.0)
    dt = full_arr - np.eye(3) * iso

    hfc = Hyperfine()
    hfc.fc = np.eye(3, dtype=float) * iso
    hfc.sd = dt
    # TODO(orbital): Preserve CSV-side orbital hyperfine decomposition once
    # the CSV contract exposes it explicitly.
    hfc.orb = np.zeros((3, 3), dtype=float)
    hfc.tensor_full = full_arr
    return hfc


def build_hfc_from_qca(
    molecule: Molecule,
    qca: Any,
    *,
    converter: str | None = "MHz_to_Ang-3",
    orbital_contribution: str = "off",
) -> Molecule:
    """Build hyperfine data from a parsed QC object and attach it to a Molecule.

    Expects `qca` to provide component hyperfine tensors keyed by atom label:
      - a_fc: mapping label -> (3, 3) A(FC) tensor
      - a_sd: mapping label -> (3, 3) A(SD) tensor
      - a_orb: mapping label -> (3, 3) A(ORB) tensor or None
      - labels: list-like (n_atoms,) of indexed per-atom labels (e.g. H1, C2)

    The parsed QC payload is assembled into canonical `Hyperfine` objects keyed
    by atom label. The assembled payload is stored on the `Molecule` and then
    projected onto matching runtime nuclei.

    Args:
        molecule: Existing Molecule to enrich with derived hyperfine data.
        qca: Parsed QC hyperfine object.
        converter: Optional converter string controlling unit conversion. When
            set to "MHz_to_Ang-3", component tensors are converted before
            deriving effective quantities. Defaults to "MHz_to_Ang-3".
        orbital_contribution: Whether to require and preserve `A(ORB)` as a
            separate orbital hyperfine contribution. Supported values are
            "off" and "on".

    Returns:
        The input Molecule enriched via canonical HFC assembly and runtime
        projection onto matching nuclei.

    Raises:
        ValueError: If required fields are missing, tensor shapes are invalid,
            or orbital contribution policy is invalid.
    """
    if not hasattr(qca, "a_fc"):
        raise ValueError("QCA object is missing required attribute: a_fc")
    if not hasattr(qca, "a_sd"):
        raise ValueError("QCA object is missing required attribute: a_sd")
    if not hasattr(qca, "a_orb"):
        raise ValueError("QCA object is missing required attribute: a_orb")
    if not hasattr(qca, "labels"):
        raise ValueError("QCA object is missing required attribute: labels")

    labels = [str(lab) for lab in qca.labels]

    a_fc: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=float) for k, v in qca.a_fc.items()
    }
    a_sd: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=float) for k, v in qca.a_sd.items()
    }
    a_orb: dict[str, np.ndarray | None] = {
        str(k): (None if v is None else np.asarray(v, dtype=float))
        for k, v in qca.a_orb.items()
    }

    if converter is None:
        pass
    elif converter == "MHz_to_Ang-3":
        a_fc = a_tensor_mhz_to_ang(a_fc)
        a_sd = a_tensor_mhz_to_ang(a_sd)

        a_orb_present = {k: v for k, v in a_orb.items() if v is not None}
        a_orb_present = a_tensor_mhz_to_ang(a_orb_present)
        a_orb = {k: a_orb_present.get(k) for k in a_orb}
    else:
        raise ValueError(f"Unknown converter: {converter}")

    if orbital_contribution not in {"off", "on"}:
        raise ValueError(
            f"orbital_contribution must be 'off' or 'on', got {orbital_contribution!r}"
        )

    a_fc_sd: dict[str, np.ndarray] = {}
    for label in labels:
        if label not in a_fc or label not in a_sd:
            continue

        fc = np.asarray(a_fc[label], dtype=float)
        sd = np.asarray(a_sd[label], dtype=float)

        if fc.shape != (3, 3):
            raise ValueError(f"A(FC) tensor for {label} must be (3,3), got {fc.shape}")
        if sd.shape != (3, 3):
            raise ValueError(f"A(SD) tensor for {label} must be (3,3), got {sd.shape}")

        a_fc_sd[label] = fc + sd

    a_tensor_full: dict[str, np.ndarray] = {}
    for label, fc_sd_tensor in a_fc_sd.items():
        total = np.asarray(fc_sd_tensor, dtype=float)

        orb = a_orb.get(label)
        if orb is not None:
            orb_arr = np.asarray(orb, dtype=float)
            if orb_arr.shape != (3, 3):
                raise ValueError(
                    f"A(ORB) tensor for {label} must be (3,3), got {orb_arr.shape}"
                )
            total = total + orb_arr

        a_tensor_full[label] = total

    hfc_by_label: dict[str, Hyperfine] = {}
    for label in labels:
        if label not in a_fc or label not in a_sd:
            continue
        hfc_by_label[label] = _assemble_hfc_from_components(
            fc=a_fc[label],
            sd=a_sd[label],
            orb=a_orb.get(label),
            tensor_full=a_tensor_full.get(label),
            orbital_contribution=orbital_contribution,
            label=label,
        )

    molecule.set_available_hfc_by_label(hfc_by_label)

    return molecule


def build_hfc_from_pdip(
    molecule: Molecule,
) -> Molecule:
    """Build point-dipole hyperfine data and attach it to a Molecule.

    This builder delegates the point-dipole hyperfine calculation to the
    existing domain method on `Molecule` and returns the same enriched domain
    object.

    Args:
        molecule: Existing Molecule to enrich with point-dipole hyperfine data.

    Returns:
        The input Molecule enriched with point-dipole hyperfine data.

    Raises:
        ValueError: If `Molecule.paramagnetic_centre` is not set.
    """
    molecule.calc_pdip()
    return molecule


def build_hfc_from_csv(
    molecule: Molecule,
    payload: dict,
) -> Molecule:
    """Build hyperfine data from a CSV payload and attach it to a Molecule.

    This builder reads full hyperfine tensors from a CSV-derived payload,
    assembles canonical `Hyperfine` objects keyed by atom label, stores them on
    the `Molecule`, and projects matching payloads onto runtime nuclei. If
    chemical labels are present in the payload, they are also applied to the
    domain object.

    Args:
        molecule: Existing Molecule to enrich with CSV-derived hyperfine data.
        payload: CSV payload that may contain:
            - labels
            - tensors
            - chem_labels
            - chem_math_labels

    Returns:
        The input Molecule enriched with canonical CSV-derived HFC assembly,
        runtime projection onto matching nuclei, and optional chemical labels.

    Raises:
        KeyError: If a required tensor is missing for a labelled nucleus.
        ValueError: If tensor shapes or chemical-label lengths are invalid.
    """
    labels: list[str] = payload["labels"]

    tensors = payload.get("tensors")
    chem_labels = payload.get("chem_labels")
    chem_math_labels = payload.get("chem_math_labels")
    r_inv6_list = payload.get("r_inv6")

    r_inv6_by_label: dict[str, float] = {}
    if r_inv6_list is not None:
        for lab, val in zip(labels, r_inv6_list):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                fval = float("nan")
            if np.isfinite(fval):
                r_inv6_by_label[lab] = fval

    if tensors is not None:
        if isinstance(tensors, dict):
            tensor_by_label = {
                k: np.asarray(v, float) for k, v in tensors.items()
            }
        else:
            tensor_by_label = {
                lab: np.asarray(t, float)
                for lab, t in zip(labels, tensors)
            }

        hfc_by_label: dict[str, Hyperfine] = {}
        for lab in labels:
            if lab not in tensor_by_label:
                raise KeyError(
                    f"Missing hyperfine tensor for label: {lab}"
                )
            hfc = _assemble_hfc_from_full_tensor(
                tensor_full=tensor_by_label[lab],
                label=lab,
            )
            if lab in r_inv6_by_label:
                hfc.r_inv6 = r_inv6_by_label[lab]
            hfc_by_label[lab] = hfc

        molecule.set_available_hfc_by_label(hfc_by_label)

    al_to_cl = None
    al_to_cml = None
    if chem_labels is not None:
        if len(chem_labels) != len(labels):
            raise ValueError(
                f"chem_labels length mismatch: {len(chem_labels)} vs {len(labels)}"
            )
        al_to_cl = {lab: str(cl) for lab, cl in zip(labels, chem_labels)}

        if chem_math_labels is not None:
            if len(chem_math_labels) != len(labels):
                raise ValueError(
                    f"chem_math_labels length mismatch: "
                    f"{len(chem_math_labels)} vs {len(labels)}"
                )
            al_to_cml = {lab: str(cml) for lab, cml in zip(labels, chem_math_labels)}

    if al_to_cl is not None:
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

    return molecule
