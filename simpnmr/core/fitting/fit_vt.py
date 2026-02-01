# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit Curie-normalised susceptibility chiT(T) data.

Computes reduced chiT values and uncertainties and fits linear chiT(T) models
with an optional TIP term.
"""

import numpy as np
from scipy.optimize import curve_fit

from simpnmr.core.constants.physics import KB, MU0, MUB, C, H


def fit_chit_linear_model(
    spin: float,
    fit_temps: np.ndarray,
    chi_vals: np.ndarray,
    chi_errors: np.ndarray,
    susc_vt_variables: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
    """
    Fit a linear chiT(T) = A + B / T + tip * T model to
    Curie-normalised susceptibility data

    If valid error estimates are provided, a weighted least-squares fit is performed

    Args:
        spin (float): Total spin quantum number S.
        fit_temps (np.ndarray): Temperature values.
        chi_vals (np.ndarray): Susceptibility values for a single chi_component.
        chi_errors (np.ndarray): Uncertainties associated with `chi_vals` as an array
        of the same shape.
        susc_vt_variables (dict): Variables controlling fit modes and initial values.
            Must include keys `intercept` and `slope`. The optional key `tip` may be
            provided as `["fit", <guess>]` or `["fix", <value>]`.

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

    def _model(T, A, B, tip):
        # TIP contributes a temperature-independent term in chi(T), which becomes
        # a linear-in-T term in chiT(T).

        # x = 1 / T
        # return A * x + B + tip * 1 / x
        return A + B / T + tip * T

    norm_factor = compute_curie_prefactor(spin)

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
        # return _model(T, params["slope"], params["intercept"], params["tip"])

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
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
    """
    Evaluate the high-temperature chiT limit assuming a zero slope

    This handler assumes B = 0 and takes the intercept from the
    highest-temperature data point. TIP is not estimated in this mode and is
    returned fixed at 0.0.

    Args:
        temperature (array_like): Temperature values
        chi_value (array_like): Susceptibility values for a single chi_component
        chi_errors: Uncertainties associated with `chi_vals` as an array
        of the same shape

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
    norm_factor = compute_curie_prefactor(spin)

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
    t_max: float,
    g_tensor: np.ndarray,
    D_J: float,
    E_J: float,
    spin: float,
) -> float:
    # Compute g-tensor components
    g_sq_iso = (g_tensor[0, 0] ** 2 + g_tensor[1, 1] ** 2 + g_tensor[2, 2] ** 2) / 3.0
    g_sq_ax = 1.5 * (g_tensor[2, 2] ** 2 - g_sq_iso)
    g_sq_rh = (g_tensor[0, 0] ** 2 - g_tensor[1, 1] ** 2) / 2.0

    # Compute Spin coefficient
    f_S = (2 * spin - 1) * (2 * spin + 3)

    # Calculate chi component in reduced (Curie) units (if I am not mistaken)
    if chi_component == "iso":
        analytic = (
            g_sq_iso - (f_S / (45 * KB * t_max)) * (D_J * g_sq_ax + 3 * E_J * g_sq_rh)
        ) / t_max
    elif chi_component == "ax":
        analytic = (
            g_sq_ax
            - (f_S / (30 * KB * t_max))
            * (D_J * (g_sq_ax + 3 * g_sq_iso) - 3 * E_J * g_sq_rh)
        ) / t_max

    elif chi_component == "rho":
        analytic = (
            g_sq_rh
            + (f_S / (30 * KB * t_max))
            * (E_J * (g_sq_ax - 3 * g_sq_iso) + D_J * g_sq_rh)
        ) / t_max
    else:
        raise ValueError(
            f"Unknown chi_component={chi_component!r}; expected 'iso', 'ax', or 'rho'."
        )

    return analytic


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
) -> float:
    norm_factor = compute_curie_prefactor(spin)

    # Convert Å^3 to reduced units
    ab_initio_chi = ab_initio_chi / norm_factor

    chi_tip = ab_initio_chi - analytic_chi

    return chi_tip


def compute_curie_prefactor(spin: float) -> float:
    """
    Compute the Curie prefactor for a given spin quantum number.

    The prefactor is used to normalise susceptibility data and is returned in
    Å^3·K (using 1 Å^3 = 1e-30 m^3).

    Args:
        spin (float): Total spin quantum number S.

    Returns:
        float: Curie prefactor in Å^3·K.
    """
    return (MU0 * MUB**2 * spin * (spin + 1)) / (3 * KB) * 1e30  # [Å^3·K]
