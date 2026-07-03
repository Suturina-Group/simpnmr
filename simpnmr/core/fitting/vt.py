# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit Curie-normalised susceptibility chiT(T) data.

Computes reduced chiT values and uncertainties and fits linear chiT(T) models
with an optional TIP term.
"""

import numpy as np
from scipy.optimize import curve_fit

from simpnmr.core.const.physics import GE, KB, MU0, MUB, C, H


def fit_chit_linear_model(
    spin: float,
    fit_temps: np.ndarray,
    chi_vals: np.ndarray,
    chi_errors: np.ndarray,
    susc_vt_variables: dict,
    total_J: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
    """
    Fit a linear chiT(T) = A + B / T + tip * T model to
    Curie-normalised susceptibility data

    If valid error estimates are provided, a weighted least-squares fit is performed

    Args:
        spin: Spin quantum number S.
        fit_temps: Temperature values in Kelvin.
        chi_vals: Susceptibility values for a single chi component in Å³.
        chi_errors: Uncertainties associated with ``chi_vals``.
        susc_vt_variables: Variables controlling fit modes and initial values.
            Must include keys ``intercept`` and ``slope``. The optional key
            ``tip`` may be provided as ``["fit", <guess>]`` or
            ``["fix", <value>]``.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the Curie prefactor J(J+1).

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
            - chiT_reduced (np.ndarray): Curie-normalised chiT values (dimensionless).
            - chi_errT_reduced (np.ndarray): Uncertainties for `chiT_reduced`.
            - fit_results (dict[str, float | None | np.ndarray]):
            Fit parameters and statistics.
              Includes `intercept`, `slope`, optional `tip`, and their uncertainties,
              along with `adj_r2`.
              The dict may also include precomputed plotting arrays:
              `fit_y`, `fit_y_low`, `fit_y_high` arrays evaluated on `fit_temps`
              for downstream visualization.
    """

    def _model(T, Intercept, Slope, tip):
        # TIP contributes a temperature-independent term in chi(T), which becomes
        # a linear-in-T term in chiT(T).

        return Intercept + Slope / T + tip * T

    norm_factor = compute_chi_prefactor(spin, total_J)

    fit_param_names: list[str] = []
    x0: list[float] = []
    fixed_intercept: float | None = None
    fixed_slope: float | None = None
    fixed_tip: float | None = 0.0

    mode_i, val_i = susc_vt_variables["intercept"]
    mode_i = str(mode_i).strip().lower()
    if mode_i == "fit":
        fit_param_names.append("intercept")
        x0.append(float(val_i))
    elif mode_i == "fix":
        fixed_intercept = float(val_i)
    else:
        raise ValueError(
            f"Invalid mode {mode_i!r} for 'intercept'. Expected 'fit' or 'fix'."
        )

    mode_s, val_s = susc_vt_variables["slope"]
    mode_s = str(mode_s).strip().lower()
    if mode_s == "fit":
        fit_param_names.append("slope")
        x0.append(float(val_s))
    elif mode_s == "fix":
        fixed_slope = float(val_s)
    else:
        raise ValueError(
            f"Invalid mode {mode_s!r} for 'slope'. Expected 'fit' or 'fix'."
        )

    # TIP is optional; if not provided, it is treated as fixed to 0.0.
    if "tip" in susc_vt_variables:
        mode_t, val_t = susc_vt_variables["tip"]
        mode_t = str(mode_t).strip().lower()
        if mode_t == "fit":
            fit_param_names.append("tip")
            x0.append(float(val_t))
            fixed_tip = None
        elif mode_t == "fix":
            fixed_tip = float(val_t)
        else:
            raise ValueError(
                f"Invalid mode {mode_t!r} for 'tip'. Expected 'fit' or 'fix'."
            )

    def _model_free(T, *theta):
        params = {
            "intercept": (fixed_intercept if fixed_intercept is not None else 0.0),
            "slope": (fixed_slope if fixed_slope is not None else 0.0),
            "tip": (fixed_tip if fixed_tip is not None else 0.0),
        }
        for name, val in zip(fit_param_names, theta, strict=False):
            params[name] = float(val)

        return _model(T, params["intercept"], params["slope"], params["tip"])

    # Compute chiT values and chiT errors at the given temperatures
    chiT = chi_vals * fit_temps
    chi_errT = chi_errors * fit_temps

    # Curie-normalised (dimensionless) values
    chiT_reduced = chiT / norm_factor
    chi_errT_reduced = chi_errT / norm_factor

    # Detect the degenerate case of identically zero chiT
    if np.allclose(chiT_reduced, 0.0):
        fit_results: dict[str, float | None | np.ndarray] = {
            "intercept": 0.0,
            "slope": 0.0,
            "tip": 0.0,
            "intercept_err": 0.0,
            "slope_err": 0.0,
            "tip_err": 0.0,
            "adj_r2": None,
            "fit_y": chiT_reduced.copy(),
            "fit_y_low": None,
            "fit_y_high": None,
        }

        return chiT_reduced, chi_errT_reduced, fit_results

    # Prepare sigma only if there are positive errors; or perform an unweighted fit
    sigma = None
    abs_sigma = False
    _errs = np.asarray(chi_errT_reduced, dtype=float)
    if np.any(_errs > 0):
        sigma = _errs
        abs_sigma = True

    if not fit_param_names:
        yhat = _model(fit_temps, fixed_intercept, fixed_slope, fixed_tip)

        ss_res = np.sum((chiT_reduced - yhat) ** 2)
        ss_tot = np.sum((chiT_reduced - np.mean(chiT_reduced)) ** 2)

        adj_r2 = None
        if ss_tot != 0 and len(chiT_reduced) > 3:
            r2 = 1 - (ss_res / ss_tot)
            adj_r2 = 1 - (1 - r2) * (len(chiT_reduced) - 1) / (
                len(chiT_reduced) - 2 - 1
            )

        fit_results: dict[str, float | None | np.ndarray] = {
            "intercept": float(fixed_intercept),
            "slope": float(fixed_slope),
            "tip": float(fixed_tip if fixed_tip is not None else 0.0),
            "intercept_err": 0.0,
            "slope_err": 0.0,
            "tip_err": 0.0,
            "adj_r2": (None if adj_r2 is None else float(adj_r2)),
            "fit_y": np.asarray(yhat, dtype=float),
            "fit_y_low": None,
            "fit_y_high": None,
        }

        return chiT_reduced, chi_errT_reduced, fit_results

    popt, pcov = curve_fit(
        _model_free,
        fit_temps,
        chiT_reduced,
        p0=x0,
        sigma=sigma,
        absolute_sigma=abs_sigma,
    )
    perr = np.sqrt(np.diag(pcov))

    # Reconstruct full parameter set
    params = {
        "intercept": (fixed_intercept if fixed_intercept is not None else 0.0),
        "slope": (fixed_slope if fixed_slope is not None else 0.0),
        "tip": (fixed_tip if fixed_tip is not None else 0.0),
    }
    for name, val in zip(fit_param_names, popt, strict=False):
        params[name] = float(val)

    # Precompute fit curve and 1-sigma band for downstream visualization.
    fit_y = np.asarray(
        _model(fit_temps, params["intercept"], params["slope"], params["tip"]),
        dtype=float,
    )

    # Build a full 3x3 covariance matrix in (intercept, slope, tip) order.
    pcov_full = np.zeros((3, 3), dtype=float)
    name_to_idx = {"intercept": 0, "slope": 1, "tip": 2}
    for i_name, i in name_to_idx.items():
        if i_name not in fit_param_names:
            continue
        ii = fit_param_names.index(i_name)
        for j_name, j in name_to_idx.items():
            if j_name not in fit_param_names:
                continue
            jj = fit_param_names.index(j_name)
            pcov_full[i, j] = float(pcov[ii, jj])

    # Jacobian of y wrt (intercept, slope, tip): [1, 1/T, T]
    T = np.asarray(fit_temps, dtype=float)
    J = np.column_stack([np.ones_like(T), 1.0 / T, T])  # shape (n, 3)

    # Var(y) ≈ J * pcov_full * J^T (parameter uncertainty only)
    var_y = np.einsum("ni,ij,nj->n", J, pcov_full, J)
    std_y = np.sqrt(np.maximum(var_y, 0.0))

    fit_y_low = fit_y - std_y
    fit_y_high = fit_y + std_y

    # R^2 metrics (guard against zero variance)
    yhat = _model(fit_temps, params["intercept"], params["slope"], params["tip"])
    ss_res = np.sum((chiT_reduced - yhat) ** 2)
    ss_tot = np.sum((chiT_reduced - np.mean(chiT_reduced)) ** 2)

    adj_r2 = None
    if ss_tot != 0 and len(chiT_reduced) > 3:
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (len(chiT_reduced) - 1) / (len(chiT_reduced) - 2 - 1)

    intercept_err = (
        float(perr[fit_param_names.index("intercept")])
        if "intercept" in fit_param_names
        else 0.0
    )

    slope_err = (
        float(perr[fit_param_names.index("slope")])
        if "slope" in fit_param_names
        else 0.0
    )

    tip_err = (
        float(perr[fit_param_names.index("tip")]) if "tip" in fit_param_names else 0.0
    )

    fit_results: dict[str, float | None | np.ndarray] = {
        "intercept": float(params["intercept"]),
        "slope": float(params["slope"]),
        "tip": float(params["tip"]),
        "intercept_err": intercept_err,
        "slope_err": slope_err,
        "tip_err": tip_err,
        "adj_r2": (None if adj_r2 is None else float(adj_r2)),
        "fit_y": fit_y,
        "fit_y_low": fit_y_low,
        "fit_y_high": fit_y_high,
    }

    return chiT_reduced, chi_errT_reduced, fit_results


def compute_chit_high_t_limit(
    spin: float,
    fit_temps: np.ndarray,
    chi_vals: np.ndarray,
    chi_errors: np.ndarray,
    total_J: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
    """
    Evaluate the high-temperature chiT limit assuming a zero slope

    This handler assumes B = 0 and takes the intercept from the
    highest-temperature data point. TIP is not estimated in this mode and is
    returned fixed at 0.0.

    Args:
        spin: Spin quantum number S.
        fit_temps: Temperature values in Kelvin.
        chi_vals: Susceptibility values for a single chi component in Å³.
        chi_errors: Uncertainties associated with ``chi_vals``.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the Curie prefactor J(J+1).

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
            - chiT_reduced (np.ndarray): Curie-normalised chiT values (dimensionless)
            - errT_reduced (np.ndarray): Uncertainties for `chiT_reduced`
            - fit_results (dict[str, float | None | np.ndarray]): Fit parameters for the
            fixed-slope branch (intercept from max(T) point,
            slope=0, tip=0, uncertainties, adj_r2=1)
            The dict may also include precomputed plotting arrays:
            `fit_y`, `fit_y_low`, `fit_y_high` arrays evaluated on `fit_temps`
            for downstream visualization.
    """
    norm_factor = compute_chi_prefactor(spin, total_J)

    # chiT in internal units -> Curie-normalised (dimensionless)
    chiT_reduced = (chi_vals * fit_temps) / norm_factor

    chi_errors = np.asarray(chi_errors, dtype=float)
    errT_reduced = (chi_errors * fit_temps) / norm_factor

    idx = int(np.nanargmax(fit_temps))

    intercept = float(chiT_reduced[idx])

    fit_results: dict[str, float | None | np.ndarray] = {
        "intercept": intercept,
        "slope": 0.0,
        "tip": 0.0,
        "intercept_err": float(errT_reduced[idx] if errT_reduced is not None else 0.0),
        "slope_err": 0.0,
        "tip_err": 0.0,
        "adj_r2": 1.0,
        "fit_y": np.asarray(chiT_reduced, dtype=float),
        "fit_y_low": None,
        "fit_y_high": None,
    }

    return chiT_reduced, errT_reduced, fit_results


def compute_analytic_component(
    chi_component: str,
    temperature: np.ndarray,
    g_sq: dict[str, float],
    D_J: float,
    E_J: float,
    spin: float,
    total_J: float | None = None,
) -> np.ndarray:
    """Evaluate an analytic high-temperature susceptibility component.

    Computes the second-order VT expansion of the iso, axial, or rhombic
    susceptibility component in reduced Curie units (chiT / prefactor, units
    of 1/K).  The ZFS second-order coefficient uses ``(2J−1)(2J+3)`` when
    ``total_J`` is supplied (lanthanide/actinide J-multiplets) or
    ``(2S−1)(2S+3)`` for pure spin systems.

    Args:
        chi_component: Which component to evaluate — ``"iso"``, ``"ax"``,
            or ``"rh"``.
        temperature: Temperature array in Kelvin.
        g_sq: Dict of squared g-tensor invariants with keys ``g_sq_iso``,
            ``g_sq_ax``, ``g_sq_rh``.
        D_J: Axial ZFS parameter in Joules.
        E_J: Rhombic ZFS parameter in Joules.
        spin: Spin quantum number S.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the ZFS coefficient ``(2J_eff−1)(2J_eff+3)``.

    Returns:
        Array of analytic chi/prefactor values (units 1/K) evaluated on
        ``temperature``.

    Raises:
        ValueError: If ``chi_component`` is not ``"iso"``, ``"ax"``, or
            ``"rh"``.
    """
    g_sq_iso = float(g_sq["g_sq_iso"])
    g_sq_ax = float(g_sq["g_sq_ax"])
    g_sq_rh = float(g_sq["g_sq_rh"])

    # Accept both scalar and array temperatures.
    t = np.asarray(temperature, dtype=float)

    # ZFS coefficient: use J when orbital angular momentum is present.
    J_eff = total_J if total_J is not None else spin
    f_S = (2 * J_eff - 1) * (2 * J_eff + 3)

    # Calculate chi component in reduced (Curie) units
    if chi_component == "iso":
        # g-corrected iso formula: χ_iso·T = g_e·g_iso − f(S)/(45kT)·(D·g_e·g_ax + 3E·g_e·g_rh)
        # The g_e·g cross-products are only needed here (ax/rh use g² invariants),
        # so read them lazily — callers computing ax/rh may omit them.
        ge_g_iso = float(g_sq["ge_g_iso"])
        ge_g_ax = float(g_sq["ge_g_ax"])
        ge_g_rh = float(g_sq["ge_g_rh"])
        analytic = (
            ge_g_iso
            - (f_S / (45 * KB * t)) * (D_J * ge_g_ax + 3 * E_J * ge_g_rh)
        ) / t
    elif chi_component == "ax":
        analytic = (
            g_sq_ax
            - (f_S / (30 * KB * t))
            * (D_J * (g_sq_ax + 3 * g_sq_iso) - 3 * E_J * g_sq_rh)
        ) / t
    elif chi_component == "rh":
        analytic = (
            g_sq_rh
            + (f_S / (30 * KB * t)) * (E_J * (g_sq_ax - 3 * g_sq_iso) + D_J * g_sq_rh)
        ) / t
    else:
        raise ValueError(
            f"Unknown chi_component={chi_component!r}; expected 'iso', 'ax', or 'rh'."
        )

    return analytic


def compute_g_sq_components(g_tensor: np.ndarray) -> dict[str, float]:
    """Compute g invariants for susceptibility components.

    Returns both g² invariants (used for ax/rh components) and g_e·g
    cross-products (used for the g-corrected iso component).

    g² invariants:
        g_sq_iso = (g_x² + g_y² + g_z²) / 3
        g_sq_ax  = 3/2 · (g_z² − g_sq_iso)
        g_sq_rh  = (g_x² − g_y²) / 2

    g_e·g cross-products (for the g-corrected iso formula):
        ge_g_iso = g_e · (g_x + g_y + g_z) / 3
        ge_g_ax  = g_e · 3/2 · (g_z − g_iso)
        ge_g_rh  = g_e · (g_x − g_y) / 2

    The g² tensor G = gᵀg is diagonalised; eigenvalues are sorted by
    ascending deviation from their mean — matching the chi-frame convention
    where axes are ordered by |χ_i − χ_iso|. g-values are recovered as
    sqrt of the G eigenvalues.

    Args:
        g_tensor: 3×3 g-tensor matrix (need not be diagonal or symmetric).

    Returns:
        dict with keys ``g_sq_iso``, ``g_sq_ax``, ``g_sq_rh``,
        ``ge_g_iso``, ``ge_g_ax``, ``ge_g_rh``.
    """
    g = np.asarray(g_tensor, dtype=float)
    G = g.T @ g  # symmetric, positive semi-definite
    g_sq_eig = np.linalg.eigvalsh(G)  # ascending order by default
    g_sq_mean = float(np.mean(g_sq_eig))
    order = np.argsort(np.abs(g_sq_eig - g_sq_mean))  # x=min dev, z=max dev
    g_sq_eig = g_sq_eig[order]
    g_x = float(np.sqrt(max(g_sq_eig[0], 0.0)))
    g_y = float(np.sqrt(max(g_sq_eig[1], 0.0)))
    g_z = float(np.sqrt(max(g_sq_eig[2], 0.0)))

    g_sq_iso = float(np.mean(g_sq_eig))
    g_sq_ax = 1.5 * (g_sq_eig[2] - g_sq_iso)
    g_sq_rh = (g_sq_eig[0] - g_sq_eig[1]) / 2.0

    g_iso = (g_x + g_y + g_z) / 3.0
    ge_g_iso = GE * g_iso
    ge_g_ax = GE * 1.5 * (g_z - g_iso)
    ge_g_rh = GE * (g_x - g_y) / 2.0

    return {
        "g_sq_iso": g_sq_iso,
        "g_sq_ax": g_sq_ax,
        "g_sq_rh": g_sq_rh,
        "ge_g_iso": ge_g_iso,
        "ge_g_ax": ge_g_ax,
        "ge_g_rh": ge_g_rh,
    }


def calculate_E_D_components(
    eff_H: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the E and D components of the effective Hamiltonian matrix

    Args:
        rotated_eff_H_tensors (list of ndarray): 3×3 Effective Hamiltonian matrix

    Returns:
        D (float): Axial component converted to Joules
        E (float): Rhombic component converted to Joules
    """

    eff_H_iso = np.trace(eff_H) / 3.0
    eff_H_traceless = eff_H - eff_H_iso * np.eye(3)

    evals, _ = np.linalg.eigh(eff_H_traceless)
    idx = np.argsort(np.abs(evals))
    eff_H_diag = np.diag(evals.real[idx])

    D = 1.5 * eff_H_diag[2, 2]
    E = (eff_H_diag[0, 0] - eff_H_diag[1, 1]) / 2

    # Convert values to Joules
    D_J = D * H * C * 100
    E_J = E * H * C * 100

    return D_J, E_J


def compute_tip_correction(
    ab_initio_chi: float,
    analytic_chi: float,
    spin: float,
    total_J: float | None = None,
) -> float:
    """Compute the TIP correction as the residual between ab initio and analytic chi.

    Converts the ab initio susceptibility to Curie-normalised units and
    returns the difference from the analytic high-temperature value.  The
    result is a dimensionless reduced TIP contribution suitable for use as a
    fixed ``tip`` variable in :func:`fit_chit_linear_model`.

    Args:
        ab_initio_chi: Ab initio susceptibility component in Å³.
        analytic_chi: Analytic chi/prefactor value (units 1/K) at the same
            reference temperature.
        spin: Spin quantum number S.
        total_J: Total angular momentum quantum number J.  When provided,
            J replaces S in the Curie prefactor J(J+1).

    Returns:
        Dimensionless TIP correction (reduced units).
    """
    norm_factor = compute_chi_prefactor(spin, total_J)

    # Convert Å^3 to reduced units
    ab_initio_chi = ab_initio_chi / norm_factor

    chi_tip = ab_initio_chi - analytic_chi

    return chi_tip


def compute_chi_prefactor(
    spin: float,
    total_J: float | None = None,
) -> float:
    """Return the bare angular-momentum susceptibility prefactor (Å³·K).

    Defined as C = μ₀ μ_B² J(J+1) / (3 k_B), with no g-factor of any kind.
    g² enters the analytic susceptibility formula separately through the
    ``g_sq`` dict passed to :func:`compute_analytic_component`.

    When ``total_J`` is ``None``, J(J+1) is replaced by S(S+1).

    Args:
        spin: Spin quantum number S.
        total_J: Total angular momentum J. When provided, replaces S(S+1).

    Returns:
        C in Å³·K.
    """
    J_eff = total_J if total_J is not None else spin
    return (MU0 * MUB**2 * J_eff * (J_eff + 1)) / (3 * KB) * 1e30  # [Å³·K]

