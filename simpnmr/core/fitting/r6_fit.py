# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit linewidth or R1 to an r^-6 distance model.

Provides a helper that fits the model ``f(r) = p1 / r**6 + p2`` to
per-nucleus linewidth or longitudinal relaxation rate data extracted from
an assigned experiment.
"""

import logging

import numpy as np
from scipy.optimize import least_squares

from simpnmr.core.build.eff_factors import calc_g_eff, choose_S_eff
from simpnmr.core.const.gammas import get_nuclear_gamma
from simpnmr.core.const.physics import EGAMMA, KB, MU0, MUB
from simpnmr.core.conv.ang_to_freq import angstrom_to_mhz, mhz_to_rad_s
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.relaxation.sbm import calc_r1_contact, calc_r2_contact

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
        p1_sbm = (2.0 / 15.0) * C_sbm * sd_sbm  # (N_e, 1)

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


def solve_tau_e_from_p1(
    p1_fit: float,
    tau_R: float,
    *,
    omega_I: float,
    omega_S: float,
    gamma_I: float,
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    temperature: float,
    observable: str,
    relaxation_model: str,
    tau_e_range: list[float] | None = None,
    tau_e_ref: float = 1e-12,
    n_points: int = 400,
) -> float | None:
    """Invert a fitted ``p1`` to the electronic correlation time ``tau_e``.

    At the fixed rotational correlation time ``tau_R``, scans ``tau_e`` and
    finds where the theoretical ``p1`` (see :func:`compute_p1_theoretical`)
    equals the fitted ``p1``. Because the SBM ``p1`` is non-monotonic in
    ``tau_c`` there can be two crossings; the root **closest (in log space) to
    ``tau_e_ref``** is returned.

    Args:
        p1_fit: Fitted ``p1`` prefactor (same units as
            :func:`compute_p1_theoretical`).
        tau_R: Fixed rotational correlation time (s).
        omega_I, omega_S, gamma_I, spin, orbit, total_momentum_J, temperature,
            observable, relaxation_model: Physical parameters forwarded to
            :func:`compute_p1_theoretical`.
        tau_e_range: Optional ``[min, max]`` scan range (s). Defaults to
            ``[1e-14, 1e-10]``.
        tau_e_ref: Reference ``tau_e`` (s); the crossing nearest this value in
            log space is chosen. Defaults to 1 ps.
        n_points: Grid resolution for the scan.

    Returns:
        The selected ``tau_e`` (s), or ``None`` if no crossing is found.
    """
    lo = np.log10(tau_e_range[0]) if tau_e_range else -14
    hi = np.log10(tau_e_range[1]) if tau_e_range else -10
    tau_e = np.logspace(lo, hi, n_points)

    grid = compute_p1_theoretical(
        tau_e, np.array([float(tau_R)]),
        omega_I=omega_I, omega_S=omega_S, gamma_I=gamma_I,
        spin=spin, orbit=orbit, total_momentum_J=total_momentum_J,
        temperature=temperature, observable=observable,
        relaxation_model=relaxation_model,
    )
    p1_curve = np.asarray(grid).reshape(len(tau_e), -1)[:, 0]

    diff = p1_curve - float(p1_fit)
    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]

    roots: list[float] = []
    for i in crossings:
        d0, d1 = diff[i], diff[i + 1]
        if d1 == d0:
            continue
        frac = -d0 / (d1 - d0)
        log_te = (
            np.log10(tau_e[i])
            + frac * (np.log10(tau_e[i + 1]) - np.log10(tau_e[i]))
        )
        roots.append(float(10 ** log_te))

    if not roots:
        return None

    _ref_log = np.log10(max(tau_e_ref, 1e-30))
    return min(roots, key=lambda t: abs(np.log10(t) - _ref_log))


def fit_r6(
    molecule: Molecule,
    experiment: Experiment,
    observable: str = "r1",
    tau_e: float | None = None,
    isotope_filter: str | None = None,
    distance_power: float = 0.0,
) -> dict:
    """Fit linewidth or R1 to the model ``p1 / r**6 + p2``.

    Uses Huber regression (``epsilon=1.35``) rather than ordinary least
    squares, making the fit robust to outliers: residuals smaller than
    1.35 σ are treated as inliers (L2 penalty) while larger residuals are
    down-weighted (L1 penalty).  Parameter uncertainties (``p1_err``,
    ``p2_err``) are estimated by bootstrap (1000 resamples with
    replacement).

    For each assigned signal the effective ``1/r**6`` is computed as the mean
    over all nuclei sharing the same chemical label (equivalent nuclei). This
    correctly averages the distance-dependent relaxation contribution across
    the group rather than using a single representative distance.

    When ``tau_e`` is provided, the Fermi-contact SBM contribution to R1 or
    R2 is computed for each chem_label group and subtracted from the observed
    value before fitting.  The contact contribution depends on the isotropic
    hyperfine coupling (``nuc.A.fc``) and the electronic correlation time
    ``tau_e = T1e = T2e`` (s).  Only nuclei with non-zero A_iso contribute;
    the subtraction is silently skipped when all A_iso are zero (e.g. pdip).

    When ``isotope_filter`` is given, only signals whose assigned chem_label
    belongs to that isotope (as recorded on the molecule's nuclei) are
    included.  This allows separate per-isotope fits when the experiment
    contains mixed-nucleus data (e.g. both ¹H and ¹³C signals).

    Args:
        molecule: Molecule with nuclei, coordinates, and
            ``paramagnetic_centre`` set.
        experiment: Experiment whose signals carry the ``r1`` or ``width``
            observable and are already assigned to chemical labels.
        observable: Which quantity to fit — ``"r1"`` (longitudinal relaxation
            rate, s⁻¹) or ``"width"`` (linewidth, ppm).
        tau_e: Electronic correlation time T1e = T2e (s) used to subtract the
            Fermi-contact relaxation contribution before fitting.  If ``None``
            no contact subtraction is performed.
        isotope_filter: If given, restrict the fit to signals whose chem_label
            corresponds to this isotope (e.g. ``"1H"``).  ``None`` includes
            all signals.
        distance_power: Exponent ``k`` for distance-based weighting.  Each
            data point is weighted by ``r_eff**k``, so larger values give
            more influence to distant (small-linewidth) peaks.  ``k=0``
            (default) gives equal weights; ``k=6`` normalises by the
            expected dipolar signal scale.

    Returns:
        A dict with keys:

        - ``"p1"`` / ``"p2"`` — fitted parameters.
        - ``"p1_err"`` / ``"p2_err"`` — bootstrap standard errors (1000
          resamples with replacement).
        - ``"r_eff"`` — effective distances ``(mean(1/r**6))**(-1/6)`` in Å,
          one per signal, in the same order as ``labels``.
        - ``"obs"`` — observed values used in the fit (contact-corrected when
          ``tau_e`` is given).
        - ``"obs_raw"`` — original observed values before contact subtraction
          (equals ``"obs"`` when ``tau_e`` is ``None``).
        - ``"contact"`` — contact contribution subtracted per signal (zero
          array when ``tau_e`` is ``None``).
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

    # Build chem_label -> isotope mapping for optional filtering
    _cl_to_iso: dict[str, str] = {}
    for nuc in molecule.nuclei:
        if nuc.chem_label not in _cl_to_iso:
            _cl_to_iso[nuc.chem_label] = nuc.isotope

    labels: list[str] = []
    r6_inv_vals: list[float] = []
    obs_vals: list[float] = []

    for sig in experiment.signals:
        if sig.assignment is None:
            continue
        cl = sig.assignment
        obs = sig.r1 if observable == "r1" else sig.width

        if obs is None:
            logger.debug("Skipping signal '%s': %s is None", cl, observable)
            continue

        if (
            isotope_filter is not None
            and _cl_to_iso.get(cl) != isotope_filter
        ):
            logger.debug(
                "Skipping signal '%s': isotope %s != filter %s",
                cl,
                _cl_to_iso.get(cl),
                isotope_filter,
            )
            continue

        nuclei_in_group = cl_to_nuclei.get(cl)
        if not nuclei_in_group:
            logger.warning("No nuclei found for label '%s', skipping", cl)
            continue

        r6_inv_group = []
        for nuc in nuclei_in_group:
            # Use pre-averaged ⟨r⁻⁶⟩ from conformer averaging when available.
            if getattr(nuc.A, "r_inv6", None) is not None:
                r6_inv_group.append(float(nuc.A.r_inv6))
                continue
            r = float(
                np.linalg.norm(np.asarray(nuc.coord, dtype=float) - centre)
            )
            if r < 1e-6:
                logger.warning(
                    "Nucleus '%s' is at the paramagnetic centre, skipping",
                    nuc.label,
                )
                continue
            r6_inv_group.append(1.0 / r**6)

        if not r6_inv_group:
            continue

        _r6_inv_mean = float(np.mean(r6_inv_group))
        # A non-positive or non-finite ⟨1/r⁶⟩ means an effectively infinite
        # distance (or missing geometry/HFC data). Such a point carries no
        # distance information and would yield r_eff = ∞, producing non-finite
        # distance weights that poison the whole fit — so exclude it.
        if not np.isfinite(_r6_inv_mean) or _r6_inv_mean <= 0.0:
            logger.warning(
                "Signal '%s' has non-positive/non-finite mean 1/r^6 (%.3g) "
                "— excluded from the r^-6 (%s) fit.",
                cl, _r6_inv_mean, observable,
            )
            continue

        labels.append(cl)
        r6_inv_vals.append(_r6_inv_mean)
        obs_vals.append(float(obs))

    if len(labels) < 2:
        raise ValueError(
            f"Need at least 2 data points to fit r^-6 model; "
            f"got {len(labels)} after filtering."
        )

    r6_inv = np.array(r6_inv_vals, dtype=float)
    obs_raw = np.array(obs_vals, dtype=float)

    # --- Contact contribution subtraction -----------------------------------
    contact = np.zeros(len(labels), dtype=float)

    if tau_e is not None:
        # Frequency constants from first nucleus with non-zero gamma
        _ref_nuc = next(
            (n for n in molecule.nuclei if n.chem_label in labels), None
        )
        if _ref_nuc is not None:
            _gamma_I = get_nuclear_gamma(_ref_nuc.isotope) * 2 * np.pi * 1e6
            _B0 = experiment.magnetic_field
            _omega_I = -_gamma_I * _B0
            _omega_S = -EGAMMA * _B0 * 2 * np.pi * 1e6
            spin = molecule.electronic.spin_S
            total_J = molecule.electronic.total_J

            # Build per-label A_iso in rad/s (averaged over group)
            _nuc_gamma_cache: dict[str, float] = {}
            _aiso_by_label: dict[str, float] = {}
            for cl in labels:
                group = cl_to_nuclei.get(cl, [])
                aiso_vals = []
                for nuc in group:
                    gamma_nuc = _nuc_gamma_cache.setdefault(
                        nuc.isotope,
                        get_nuclear_gamma(nuc.isotope) * 2 * np.pi * 1e6,
                    )
                    if gamma_nuc == 0:
                        continue
                    a_iso_ang = float(np.trace(nuc.A.fc)) / 3.0
                    a_iso_mhz = float(
                        angstrom_to_mhz(
                            a_iso_ang,
                            get_nuclear_gamma(nuc.isotope),
                        )
                    )
                    # rad/s (angular frequency for the SBM contact rates)
                    aiso_vals.append(float(mhz_to_rad_s(a_iso_mhz)))
                _aiso_by_label[cl] = (
                    float(np.mean(aiso_vals)) if aiso_vals else 0.0
                )

            _omega_I_dict = {cl: _omega_I for cl in labels}

            if observable == "r1":
                _contact_rates = calc_r1_contact(
                    nuclei_labels=labels,
                    Aiso_dict=_aiso_by_label,
                    omega_I_dict=_omega_I_dict,
                    omega_S=_omega_S,
                    tau_e2=tau_e,
                    spin=spin,
                    total_momentum_J=total_J,
                )
                for i, cl in enumerate(labels):
                    contact[i] = _contact_rates.get(cl, 0.0)
            else:  # width (ppm)
                _contact_rates = calc_r2_contact(
                    nuclei_labels=labels,
                    Aiso_dict=_aiso_by_label,
                    omega_I_dict=_omega_I_dict,
                    omega_S=_omega_S,
                    tau_e1=tau_e,
                    tau_e2=tau_e,
                    spin=spin,
                    total_momentum_J=total_J,
                )
                for i, cl in enumerate(labels):
                    # R2 (s⁻¹) → linewidth (ppm): lw = R2 / (π |γI| B0)
                    r2 = _contact_rates.get(cl, 0.0)
                    contact[i] = r2 / (np.pi * abs(_gamma_I) * _B0)

            logger.info(
                "r^-6 fit (%s): contact subtraction with tau_e=%.3g s",
                observable,
                tau_e,
            )

    obs = obs_raw - contact
    # -------------------------------------------------------------------------

    # Distance-based weights: w_i = r_eff_i ** distance_power.
    # Residuals are pre-scaled by sqrt(w) so that least_squares minimises
    # sum w_i * rho(resid_i) rather than sum rho(resid_i).
    r_eff_raw = r6_inv ** (-1.0 / 6.0)
    if distance_power != 0.0:
        sqrt_w = r_eff_raw ** (distance_power / 2.0)
        sqrt_w /= sqrt_w.mean()   # normalise so f_scale stays in obs units
    else:
        sqrt_w = np.ones(len(obs))

    # f_scale sets the Huber transition threshold in units of the residual.
    # Using the data std gives "residuals > 1σ are outliers".
    f_scale = float(np.std(obs)) or 1.0

    def _resid(params):
        return (params[0] * r6_inv + params[1] - obs) * sqrt_w

    result = least_squares(
        _resid, [1.0, 0.0], loss="huber", f_scale=f_scale
    )
    p1_val = float(result.x[0])
    p2_val = float(result.x[1])

    pred = p1_val * r6_inv + p2_val
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    # Bootstrap standard errors (1000 resamples with replacement)
    rng = np.random.default_rng(0)
    n = len(obs)
    boot_p1 = np.empty(1000)
    boot_p2 = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, n, size=n)
        x_b, y_b, sw_b = r6_inv[idx], obs[idx], sqrt_w[idx]
        f_b = float(np.std(y_b)) or f_scale

        def _resid_b(params, xb=x_b, yb=y_b, swb=sw_b):
            return (params[0] * xb + params[1] - yb) * swb

        r_b = least_squares(_resid_b, [p1_val, p2_val], loss="huber", f_scale=f_b)
        boot_p1[i] = r_b.x[0]
        boot_p2[i] = r_b.x[1]
    p1_err = float(np.std(boot_p1, ddof=1))
    p2_err = float(np.std(boot_p2, ddof=1))

    # Effective distance for plotting: (mean(1/r^6))^(-1/6)
    r_eff = (r6_inv) ** (-1.0 / 6.0)

    logger.info(
        "r^-6 fit (%s): p1 = %.4g ± %.4g, p2 = %.4g ± %.4g, RMSE = %.4g",
        observable,
        p1_val,
        p1_err,
        p2_val,
        p2_err,
        rmse,
    )

    # Build math_labels in the same order as labels, falling back to chem_label
    math_labels = [
        cl_to_nuclei[cl][0].chem_math_label if cl in cl_to_nuclei else cl
        for cl in labels
    ]

    return {
        "p1": p1_val,
        "p2": p2_val,
        "p1_err": p1_err,
        "p2_err": p2_err,
        "r_eff": r_eff,
        "obs": obs,
        "obs_raw": obs_raw,
        "contact": contact,
        "pred": pred,
        "labels": labels,
        "math_labels": math_labels,
        "rmse": rmse,
    }
