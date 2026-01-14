# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""
Compute Curie-normalised susceptibility (chiT) linear-fit parameters.

This module provides small helpers used by plotting and fitting scripts to
normalise susceptibility data by the Curie prefactor and to extract linear
model parameters for chiT(T) data.
"""

import numpy as np
from scipy.constants import c, h, k, mu_0, physical_constants
from scipy.optimize import curve_fit

# Physical constants (SI units)
G_E = abs(physical_constants["electron g factor"][0])
MU_B = physical_constants["Bohr magneton"][0]


def fit_chit_linear_model(
    spin,
    fit_temps,
    chi_vals,
    chi_errors,
    tip_corrections,
    susc_vt_variables,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | None]]:
    """
    Fit a linear chiT = A + B / T model to Curie-normalised susceptibility data.

    If valid error estimates are provided, a weighted least-squares fit is used.

    """

    def _model(T, A, B):
        return A + B / T

    norm_factor = compute_curie_prefactor(spin)

    fit_param_names: list[str] = []
    x0: list[float] = []
    fixed_intercept: float | None = None
    fixed_slope: float | None = None

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

    def _model_free(T, *theta):
        params = {
            "intercept": (fixed_intercept if fixed_intercept is not None else 0.0),
            "slope": (fixed_slope if fixed_slope is not None else 0.0),
        }
        for name, val in zip(fit_param_names, theta, strict=False):
            params[name] = float(val)
        return _model(T, params["intercept"], params["slope"])

    # Compute chiT, chiT errors and TIP corrections from available temperatures
    chiT = chi_vals * fit_temps
    chi_errT = chi_errors * fit_temps if chi_errors is not None else None
    tip_corrections_T = tip_corrections * fit_temps

    # Curie-normalised (dimensionless) output
    chiT_reduced = chiT / norm_factor
    chi_errT_reduced = chi_errT / norm_factor

    # Apply TIP corrections (0 if not available)
    chiT_reduced = chiT_reduced - tip_corrections_T

    # Detect degenerate case: identically zero chiT
    if np.allclose(chiT_reduced, 0.0):
        fit_results: dict[str, float | None] = {
            "intercept": 0.0,
            "slope": 0.0,
            "intercept_err": 0.0,
            "slope_err": 0.0,
            "adj_r2": None,
        }

        return chiT_reduced, chi_errT_reduced, fit_results

    # Prepare sigma only if there are positive errors; otherwise, fit unweighted
    sigma = None
    abs_sigma = False
    if chi_errT_reduced is not None:
        _errs = np.asarray(chi_errT_reduced, dtype=float)
        if np.any(_errs > 0):
            sigma = _errs
            abs_sigma = True

    if not fit_param_names:
        yhat = _model(fit_temps, fixed_intercept, fixed_slope)

        ss_res = np.sum((chiT_reduced - yhat) ** 2)
        ss_tot = np.sum((chiT_reduced - np.mean(chiT_reduced)) ** 2)

        adj_r2 = None
        if ss_tot != 0 and len(chiT_reduced) > 3:
            r2 = 1 - (ss_res / ss_tot)
            adj_r2 = 1 - (1 - r2) * (len(chiT_reduced) - 1) / (
                len(chiT_reduced) - 2 - 1
            )

        fit_results: dict[str, float | None] = {
            "intercept": float(fixed_intercept),
            "slope": float(fixed_slope),
            "intercept_err": 0.0,
            "slope_err": 0.0,
            "adj_r2": (None if adj_r2 is None else float(adj_r2)),
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
    }
    for name, val in zip(fit_param_names, popt, strict=False):
        params[name] = float(val)

    # R^2 metrics (guard against zero variance)
    yhat = _model(fit_temps, params["intercept"], params["slope"])
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

    fit_results: dict[str, float | None] = {
        "intercept": float(params["intercept"]),
        "slope": float(params["slope"]),
        "intercept_err": intercept_err,
        "slope_err": slope_err,
        "adj_r2": (None if adj_r2 is None else float(adj_r2)),
    }

    return chiT_reduced, chi_errT_reduced, fit_results


def compute_chit_high_t_limit(
    spin,
    fit_temps,
    chi_vals,
    chi_errors,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | None]]:
    """
    Evaluate the high-temperature chiT limit assuming a zero slope.

    This handler assumes B = 0 and takes the intercept from the
    highest-temperature data point.

    Args:
        temperature (array_like): Temperature values.
        chi_value (array_like): Susceptibility values for a single chi_component.
        chi_errors (array_like | None): Uncertainties associated with `chi_value`.
        norm_factor (float): Curie prefactor used for normalisation.

    Returns:
        tuple[np.ndarray, np.ndarray | None, dict[str, float | None]]:
            - chiT_reduced (np.ndarray): Curie-normalised chiT values (dimensionless).
            - errT_reduced (np.ndarray | None): Uncertainties for `chiT_reduced`,
              or None if unavailable.
            - fit_results (dict[str, float | None]): Fit parameters for the
            fixed-slope branch (intercept from max(T) point,
            slope=0, uncertainties, adj_r2=1).
    """

    norm_factor = compute_curie_prefactor(spin)

    # chiT in internal units -> Curie-normalised (dimensionless)
    chiT_reduced = (chi_vals * fit_temps) / norm_factor

    errT_reduced = None
    if chi_errors is not None:
        chi_errors = np.asarray(chi_errors, dtype=float)
        errT_reduced = (chi_errors * fit_temps) / norm_factor

    idx = int(np.nanargmax(fit_temps))

    intercept = float(chiT_reduced[idx])

    fit_results: dict[str, float | None] = {
        "intercept": intercept,
        "slope": 0.0,
        "intercept_err": float(errT_reduced[idx] if errT_reduced is not None else 0.0),
        "slope_err": 0.0,
        "adj_r2": 1.0,
    }

    return chiT_reduced, errT_reduced, fit_results


def compute_analytic_component(
    chi_component,
    t_max: float,
    g_tensor,
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
            g_sq_iso - (f_S / (45 * k * t_max)) * (D_J * g_sq_ax + 3 * E_J * g_sq_rh)
        ) / t_max
    elif chi_component == "ax":
        analytic = (
            g_sq_ax
            - (f_S / (30 * k * t_max))
            * (D_J * (g_sq_ax + 3 * g_sq_iso) - 3 * E_J * g_sq_rh)
        ) / t_max

    elif chi_component == "rho":
        analytic = (
            g_sq_rh
            + (f_S / (30 * k * t_max))
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
    Calculate E and D Effective Hamiltonian matrix components

    Args:
        rotated_eff_H_tensors (list of ndarray): 3×3 Effective Hamiltonian matrix

    Returns:
        D (float): Axial component of the Effective Hamiltonian matrix
        E (float): Rhombic component of the Effective Hamiltonian matrix
    """

    eff_H_iso = np.trace(eff_H) / 3.0
    eff_H_traceless = eff_H - eff_H_iso * np.eye(3)

    evals, _ = np.linalg.eigh(eff_H_traceless)
    idx = np.argsort(np.abs(evals))
    eff_H_diag = np.diag(evals.real[idx])

    D = 1.5 * eff_H_diag[2, 2]
    E = (eff_H_diag[0, 0] - eff_H_diag[1, 1]) / 2

    # Convert values to Joules
    D_J = D * h * c * 100
    E_J = E * h * c * 100

    return D_J, E_J


def compute_tip_correction(
    ab_initio_chi: float,
    analytic_chi: float,
    spin,
) -> float:
    norm_factor = compute_curie_prefactor(spin)

    # Convert A3 -> reduced units
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
    return (mu_0 * MU_B**2 * spin * (spin + 1)) / (3 * k) * 1e30  # [Å^3·K]
