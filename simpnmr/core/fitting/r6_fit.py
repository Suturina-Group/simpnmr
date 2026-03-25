# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit linewidth or R1 to an r^-6 distance model.

Provides a helper that fits the model ``f(r) = p1 / r**6 + p2`` to
per-nucleus linewidth or longitudinal relaxation rate data extracted from
an assigned experiment.
"""

import logging

import numpy as np
from scipy.optimize import curve_fit

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule

logger = logging.getLogger(__name__)


def _r6_model(r6_inv: np.ndarray, p1: float, p2: float) -> np.ndarray:
    """Evaluate ``p1 / r**6 + p2`` given pre-computed ``1/r**6`` values."""
    return p1 * r6_inv + p2


def fit_r6(
    molecule: Molecule,
    experiment: Experiment,
    observable: str = "r1",
) -> dict:
    """Fit linewidth or R1 to the model ``p1 / r**6 + p2``.

    For each assigned signal the effective ``1/r**6`` is computed as the mean
    over all nuclei sharing the same chemical label (equivalent nuclei). This
    correctly averages the distance-dependent relaxation contribution across
    the group rather than using a single representative distance.

    Args:
        molecule: Molecule with nuclei, coordinates, and
            ``paramagnetic_centre`` set.
        experiment: Experiment whose signals carry the ``r1`` or ``width``
            observable and are already assigned to chemical labels.
        observable: Which quantity to fit — ``"r1"`` (longitudinal relaxation
            rate, s⁻¹) or ``"width"`` (linewidth, ppm).

    Returns:
        A dict with keys:

        - ``"p1"`` / ``"p2"`` — fitted parameters.
        - ``"p1_err"`` / ``"p2_err"`` — one-sigma uncertainties from the
          covariance matrix.
        - ``"r_eff"`` — effective distances ``(mean(1/r**6))**(-1/6)`` in Å,
          one per signal, in the same order as ``labels``.
        - ``"obs"`` — observed values used in the fit.
        - ``"labels"`` — chemical labels in the same order.
        - ``"pred"`` — model-predicted values at each ``r_eff``.
        - ``"rmse"`` — root-mean-square error of the fit (same units as
          observable).

    Raises:
        ValueError: If ``paramagnetic_centre`` is not set on the molecule, if
            ``observable`` is not ``"r1"`` or ``"width"``, or if fewer than
            two data points are available after filtering missing values.
    """
    if observable not in ("r1", "width"):
        raise ValueError("observable must be 'r1' or 'width'")

    if molecule.paramagnetic_centre is None:
        raise ValueError("molecule.paramagnetic_centre must be set")

    centre = np.asarray(molecule.paramagnetic_centre, dtype=float)

    # Build a lookup: chem_label -> list of nuclei
    cl_to_nuclei: dict[str, list] = {}
    for nuc in molecule.nuclei:
        cl_to_nuclei.setdefault(nuc.chem_label, []).append(nuc)

    labels: list[str] = []
    r6_inv_vals: list[float] = []
    obs_vals: list[float] = []

    for sig in experiment.signals:
        cl = sig.assignment
        obs = sig.r1 if observable == "r1" else sig.width

        if obs is None:
            logger.debug("Skipping signal '%s': %s is None", cl, observable)
            continue

        nuclei_in_group = cl_to_nuclei.get(cl)
        if not nuclei_in_group:
            logger.warning("No nuclei found for label '%s', skipping", cl)
            continue

        r6_inv_group = []
        for nuc in nuclei_in_group:
            r = float(np.linalg.norm(np.asarray(nuc.coord, dtype=float) - centre))
            if r < 1e-6:
                logger.warning(
                    "Nucleus '%s' is at the paramagnetic centre, skipping", nuc.label
                )
                continue
            r6_inv_group.append(1.0 / r**6)

        if not r6_inv_group:
            continue

        labels.append(cl)
        r6_inv_vals.append(float(np.mean(r6_inv_group)))
        obs_vals.append(float(obs))

    if len(labels) < 2:
        raise ValueError(
            f"Need at least 2 data points to fit r^-6 model; "
            f"got {len(labels)} after filtering."
        )

    r6_inv = np.array(r6_inv_vals, dtype=float)
    obs = np.array(obs_vals, dtype=float)

    popt, pcov = curve_fit(_r6_model, r6_inv, obs, p0=[1.0, 0.0])
    perr = np.sqrt(np.diag(pcov))

    pred = _r6_model(r6_inv, *popt)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    # Effective distance for plotting: (mean(1/r^6))^(-1/6)
    r_eff = (r6_inv) ** (-1.0 / 6.0)

    logger.info(
        "r^-6 fit (%s): p1 = %.4g ± %.4g, p2 = %.4g ± %.4g, RMSE = %.4g",
        observable,
        popt[0],
        perr[0],
        popt[1],
        perr[1],
        rmse,
    )

    return {
        "p1": popt[0],
        "p2": popt[1],
        "p1_err": perr[0],
        "p2_err": perr[1],
        "r_eff": r_eff,
        "obs": obs,
        "pred": pred,
        "labels": labels,
        "rmse": rmse,
    }
