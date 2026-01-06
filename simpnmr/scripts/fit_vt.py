# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""
Compute Curie-normalised susceptibility (chiT) linear-fit parameters.

This module provides small helpers used by plotting and fitting scripts to
normalise susceptibility data by the Curie prefactor and to extract linear
model parameters for chiT(T) data.
"""

import numpy as np
from scipy.constants import k, mu_0, physical_constants
from scipy.optimize import curve_fit

# Physical constants (SI units)
G_E = abs(physical_constants["electron g factor"][0])
MU_B = physical_constants["Bohr magneton"][0]


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


def compute_chit_linear_parameters(
    method,
    vt_variables,
    component,
    spin: float,
    temperature,
    chi_component,
    y_mode: str,
    use_errors: bool = False,
    chi_errors=None,
    tip_corrections: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | None] | None]:
    """
    Compute chi/chiT outputs and (optionally) fit linear parameters for chiT.

    Depending on `y_mode` and the selected method, this function:
    - returns raw susceptibility values unchanged (for y_mode != "chiT"),
    - evaluates a high-temperature chiT limit (zero slope), or
    - fits a linear model chiT = A + B / T to Curie-normalised chiT.

    All inputs are expected in internal per-particle units (Å^3).

    Args:
        method (str | None): Fit selector. Use "ht_limit" to force a zero-slope
            high-temperature limit. If None, the default behaviour is used.
        vt_variables (dict | None): YAML-style variables mapping from inputs.
            Expected keys:
            - iso_intercept, iso_slope
            - ax_intercept, ax_slope
            - rho_intercept, rho_slope
        component (str): Component selector used to extract the corresponding
            intercept/slope pair from `vt_variables`. Supported values:
            "iso", "ax", "rho".
        spin (float): Total spin quantum number S.
        temperature (array_like): Temperature values.
        chi_component (array_like): Susceptibility values for a single component.
        y_mode (str): Output mode, typically "chi" or "chiT" (case-insensitive).
        use_errors (bool): If True, attempt a weighted fit using `chi_errors`.
        chi_errors (array_like | None): Uncertainties for `chi_component`.
        tip_corrections (dict[str, float] | None): Optional TIP corrections to apply
            to the susceptibility prior to chiT processing. Expected keys are
            "iso_tip", "ax_tip", and "rho_tip". The selected component's TIP
            value is subtracted from `chi_component` (fit - model convention).

    Returns:
        tuple[np.ndarray, np.ndarray | None, dict[str, float | None] | None]:
            - y (np.ndarray): Output values. For y_mode=="chiT" this is the
              Curie-normalised chiT array; otherwise it is the raw chi array.
            - y_err (np.ndarray | None): Output uncertainties, or None if unavailable.
            - fit_results (dict[str, float | None] | None): Fit metadata containing
              intercept/slope (and uncertainties) when a linear analysis is
              performed; otherwise None.
    """

    chi_errors = (
        np.asarray(chi_errors, dtype=float)
        if use_errors and chi_errors is not None
        else None
    )

    temperature = np.asarray(temperature, dtype=float)
    chi_component = np.asarray(chi_component, dtype=float)

    if component not in {"iso", "ax", "rho"}:
        raise ValueError(
            f"Unknown component '{component}'. Expected one of: 'iso', 'ax', 'rho'."
        )

    if y_mode != "chiT":
        return chi_component, chi_errors, None

    norm_factor = compute_curie_prefactor(spin)

    # Apply a constant TIP correction (in chi units) when provided.
    if tip_corrections is not None:
        key_map = {"iso": "iso_tip", "ax": "ax_tip", "rho": "rho_tip"}
        tip_key = key_map[component]
        print(
            f"TIP DEBUG: component={component}, tip_key={tip_key}, "
            f"tip_val={tip_corrections.get(tip_key)}"
        )
        if tip_key not in tip_corrections:
            raise ValueError(
                f"tip_corrections is missing required key '{tip_key}' "
                f"for component '{component}'"
            )
        chi_component = chi_component - float(tip_corrections[tip_key])

    if temperature.size == 1 or method == "ht_limit":
        return _compute_chit_high_t_limit(
            temperature,
            chi_component,
            chi_errors,
            norm_factor,
        )

    # Default behaviour: if VT configuration is not provided at all,
    # fit both parameters starting from 0.0.
    if vt_variables is None:
        return _fit_chit_linear_model(
            temperature,
            chi_component,
            chi_errors,
            norm_factor,
            {"intercept": 0.0, "slope": 0.0},
            {},
        )

    fit_vars: dict[str, float] = {}
    fix_vars: dict[str, float] = {}

    k_int = f"{component}_intercept"
    k_slp = f"{component}_slope"

    mode_i, val_i = vt_variables[k_int]
    if mode_i == "fit":
        fit_vars["intercept"] = val_i
    elif mode_i == "fix":
        fix_vars["intercept"] = val_i
    else:
        raise ValueError(
            f"Invalid mode {mode_i!r} for {k_int}. Expected 'fit' or 'fix'."
        )

    mode_s, val_s = vt_variables[k_slp]
    if mode_s == "fit":
        fit_vars["slope"] = val_s
    elif mode_s == "fix":
        fix_vars["slope"] = val_s
    else:
        raise ValueError(
            f"Invalid mode {mode_s!r} for {k_slp}. Expected 'fit' or 'fix'."
        )

    return _fit_chit_linear_model(
        temperature, chi_component, chi_errors, norm_factor, fit_vars, fix_vars
    )


def _fit_chit_linear_model(
    temperature,
    chi_component,
    chi_errors,
    norm_factor: float,
    fit_vars: dict[str, float],
    fix_vars: dict[str, float],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | None]]:
    """
    Fit a linear chiT = A + B / T model to Curie-normalised susceptibility data.

    If valid error estimates are provided, a weighted least-squares fit is used.

    Args:
        temperature (array_like): Temperature values.
        chi_component (array_like): Susceptibility values for a single component.
        chi_errors (array_like | None): Uncertainties associated with `chi_component`.
        norm_factor (float): Curie prefactor used for normalisation.
        fit_vars (dict[str, float]): Initial guesses for parameters to be fitted.
        fix_vars (dict[str, float]): Fixed parameter values.

    Returns:
        tuple[np.ndarray, np.ndarray | None, dict[str, float | None]]:
            - chiT_reduced (np.ndarray): Curie-normalised chiT values (dimensionless).
            - chi_errT_reduced (np.ndarray | None): Uncertainties for `chiT_reduced`,
              or None if unavailable.
            - fit_results (dict[str, float | None]): Fit parameters
              and basic quality metrics
              (intercept, slope, parameter errors, adj_r2).
    """

    def _model(T, A, B):
        return A + B / T

    fit_param_names = [name for name in ("intercept", "slope") if name in fit_vars]
    x0 = [float(fit_vars[name]) for name in fit_param_names]

    fixed_intercept = (
        float(fix_vars.get("intercept")) if "intercept" in fix_vars else None
    )
    fixed_slope = float(fix_vars.get("slope")) if "slope" in fix_vars else None

    def _model_free(T, *theta):
        params = {"intercept": fixed_intercept, "slope": fixed_slope}
        for name, val in zip(fit_param_names, theta, strict=False):
            params[name] = float(val)
        return _model(T, params["intercept"], params["slope"])

    # chiT in internal units (Å^3·K)
    chiT = chi_component * temperature
    chi_errT = chi_errors * temperature if chi_errors is not None else None

    # Curie-normalised (dimensionless) output
    chiT_reduced = chiT / norm_factor
    chi_errT_reduced = chi_errT / norm_factor if chi_errT is not None else None

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
        yhat = _model(temperature, fixed_intercept, fixed_slope)

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
        temperature,
        chiT_reduced,
        p0=x0,
        sigma=sigma,
        absolute_sigma=abs_sigma,
    )
    perr = np.sqrt(np.diag(pcov))

    # Reconstruct full parameter set
    params = {"intercept": fixed_intercept, "slope": fixed_slope}
    for name, val in zip(fit_param_names, popt, strict=False):
        params[name] = float(val)

    # R^2 metrics (guard against zero variance)
    yhat = _model(temperature, params["intercept"], params["slope"])
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


def _compute_chit_high_t_limit(
    temperature,
    chi_component,
    chi_errors,
    norm_factor: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | None]]:
    """
    Evaluate the high-temperature chiT limit assuming a zero slope.

    This handler assumes B = 0 and takes the intercept from the
    highest-temperature data point.

    Args:
        temperature (array_like): Temperature values.
        chi_component (array_like): Susceptibility values for a single component.
        chi_errors (array_like | None): Uncertainties associated with `chi_component`.
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

    # chiT in internal units -> Curie-normalised (dimensionless)
    chiT_reduced = (chi_component * temperature) / norm_factor

    errT_reduced = None
    if chi_errors is not None:
        chi_errors = np.asarray(chi_errors, dtype=float)
        errT_reduced = (chi_errors * temperature) / norm_factor

    idx = int(np.nanargmax(temperature))

    intercept = float(chiT_reduced[idx])

    fit_results: dict[str, float | None] = {
        "intercept": intercept,
        "slope": 0.0,
        "intercept_err": float(errT_reduced[idx] if errT_reduced is not None else 0.0),
        "slope_err": 0.0,
        "adj_r2": 1.0,
    }

    return chiT_reduced, errT_reduced, fit_results
