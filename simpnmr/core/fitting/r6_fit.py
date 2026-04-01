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

from simpnmr.core.build.eff_factors import calc_g_eff, choose_S_eff
from simpnmr.core.const.physics import KB, MU0, MUB
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule

logger = logging.getLogger(__name__)

_ANG_TO_M = 1e-10  # Å → m conversion


def compute_p1_theoretical(
    tau_e_grid: np.ndarray,
    tau_R_grid: np.ndarray,
    omega_I: float,
    omega_S: float,
    gamma_I: float,
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    temperature: float,
    observable: str,
    relaxation_model: str,
) -> np.ndarray:
    """Compute the theoretical r^-6 prefactor p1 on a 2D (τe, τR) grid.

    Evaluates the analytical SBM and/or Curie dipolar prefactors and returns
    p1 in units of ``s⁻¹·Å⁶`` (R1) or ``Hz·Å⁶`` (width/R2).

    The model is ``observable = p1 * mean(1/r^6) + p2`` where ``mean(1/r^6)``
    is in Å⁻⁶.

    Args:
        tau_e_grid: 1D array of τe values (s). Used as τc1=τc2=T1e=T2e.
        tau_R_grid: 1D array of τR values (s).
        omega_I: Nuclear Larmor angular frequency (rad s⁻¹).
        omega_S: Electron Larmor angular frequency (rad s⁻¹).
        gamma_I: Nuclear gyromagnetic ratio (rad s⁻¹ T⁻¹).
        spin: Electron spin quantum number S.
        orbit: Orbital angular momentum L.
        total_momentum_J: Total angular momentum J, or None for spin-only.
        temperature: Temperature in Kelvin.
        observable: ``"r1"`` or ``"width"``.
        relaxation_model: One of ``"sbm"``, ``"curie"``, ``"sbm curie"``.

    Returns:
        2D array of shape ``(len(tau_e_grid), len(tau_R_grid))`` with p1
        values in ``s⁻¹·Å⁶`` (R1) or ``Hz·Å⁶`` (width).
    """
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)
    model = relaxation_model.strip().lower()
    # Conversion: p1_SI [s⁻¹·m⁶] → p1_Ang [s⁻¹·Å⁶]
    ang6 = _ANG_TO_M ** (-6)

    def J(omega, tau):
        return tau / (1.0 + (omega * tau) ** 2)

    # SBM dipolar prefactor (SI, without 1/r^6)
    C_sbm = (
        (MU0 / (4 * np.pi)) ** 2
        * (gamma_I * g_eff * MUB) ** 2
        * S_eff * (S_eff + 1)
    )
    # Curie dipolar prefactor (SI, without 1/r^6)
    C_curie = (
        (MU0 / (4 * np.pi)) ** 2
        * (omega_I / (3.0 * KB * temperature)) ** 2
        * (g_eff * MUB) ** 4
        * (S_eff * (S_eff + 1)) ** 2
    )

    tau_e = tau_e_grid[:, np.newaxis]  # (N_e, 1)
    tau_R = tau_R_grid[np.newaxis, :]  # (1, N_R)

    if observable == "r1":
        # SBM dipolar R1 spectral density
        sd_sbm = (
            3 * J(omega_I, tau_e)
            + 6 * J(omega_I + omega_S, tau_e)
            + J(omega_I - omega_S, tau_e)
        )
        p1_sbm = (1.0 / 10.0) * C_sbm * sd_sbm  # (N_e, 1)

        # Curie R1 spectral density
        sd_curie = 3 * J(omega_I, tau_R)
        p1_curie = (2.0 / 5.0) * C_curie * sd_curie  # (1, N_R)

    else:  # width (R2 / π)
        sd_sbm = (
            4 * J(0, tau_e)
            + 3 * J(omega_I, tau_e)
            + 6 * J(omega_S, tau_e)
            + 6 * J(omega_I + omega_S, tau_e)
            + J(omega_I - omega_S, tau_e)
        )
        p1_sbm = (1.0 / 15.0) * C_sbm * sd_sbm / np.pi

        sd_curie = 4 * J(0, tau_R) + 3 * J(omega_I, tau_R)
        p1_curie = (1.0 / 5.0) * C_curie * sd_curie / np.pi

    if model == "sbm":
        p1_si = p1_sbm + np.zeros_like(tau_R)
    elif model == "curie":
        p1_si = np.zeros_like(tau_e) + p1_curie
    else:  # sbm curie / curie sbm
        p1_si = p1_sbm + p1_curie

    return p1_si * ang6


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
