# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""
Extract g-tensor principal values from Curie-normalised chiT regression fits.

This script reads a chiT regression CSV (with intercept/slope terms for iso/ax/rh)
and solves for principal g-values either using a full (rhombic) numerical solve
or an axial-only analytic approximation.
"""

import logging

import numpy as np
from scipy.constants import physical_constants
from sympy import nsolve, symbols

from simpnmr.io.csv.fitting import read_chiT_regression_csv

logger = logging.getLogger(__name__)

# Imports
G_E = abs(physical_constants["electron g factor"][0])
MU_B = physical_constants["Bohr magneton"][0]
K = physical_constants["Boltzmann constant"][0]
H = physical_constants["Planck constant"][0]
C = physical_constants["speed of light in vacuum"][0]


def run_get_sh(options) -> int:
    """Run the get_sh workflow (application-layer entrypoint).

    This function is intentionally CLI-agnostic: it receives an options object and
    emits user-facing information via logging. Argument parsing and logging setup
    are handled by the CLI layer.

    Args:
        options: Run options (typically `GetSHRunOptions`).

    Returns:
        Process exit code (0 on success).
    """

    params = read_chiT_regression_csv(options.chiT_regression_csv)

    method_used, g_nominal, g_err = compute_g_tensor(
        options.chiT_regression_csv,
        params=params,
        atol=1e-6,
    )

    logger.info("g-tensor solver method: %s", method_used)

    g_parts = [f"{v:.3f} ± {e:.3f}" for v, e in zip(g_nominal, g_err, strict=True)]

    logger.info("g_x = %s", g_parts[0])
    logger.info("g_y = %s", g_parts[1])
    logger.info("g_z = %s", g_parts[2])

    spin = float(options.spin)

    if spin != 0.5:
        D, E = solve_D_E(params, spin, g_nominal)
        D = 0.0 if abs(D) < 1e-10 else D
        E = 0.0 if abs(E) < 1e-10 else E
        logger.info("D = %.3f cm^-1", D)
        logger.info("E = %.3f cm^-1", E)
    else:
        logger.info(
            "ZFS parameters (D, E) are not defined for S = 1/2 "
            "and are therefore not reported."
        )

    return 0


def compute_g_tensor(
    filename: str,
    *,
    params: dict[str, float] | None = None,
    atol: float = 1e-6,
) -> tuple[
    str,
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Compute g-tensor principal values with 1σ uncertainties via error propagation.

    This is the single entry point that:
      - reads the CSV once
      - selects the solver branch
      - runs a nominal solve
      - propagates intercept 1σ uncertainties via a finite-difference Jacobian

    Args:
        filename (str): Path to the regression CSV file.
        params (dict[str, float] | None): Optional pre-parsed regression parameters.
            If provided, the CSV will not be re-read.
        atol (float): Absolute tolerance used to decide whether rhombicity is ~0.

    Returns:
        tuple[str, tuple[float, float, float], tuple[float, float, float]]:
            (method_used, g_nominal, g_err)
    """

    rho_intercept = float(params.get("rho_intercept", 0.0))

    is_zero_rhombicity = np.isclose(rho_intercept, 0.0, atol=atol)

    method_used = (
        "Analytic (zero rhombicity)"
        if is_zero_rhombicity
        else "Nonlinear solve (non-zero rhombicity)"
    )

    solver = (
        _solve_g_principals_axial_only
        if is_zero_rhombicity
        else _solve_g_principals_full
    )

    # Nominal solution
    g0 = np.asarray(solver(params), dtype=float)

    # Proper 1σ uncertainty via delta-method: finite-difference Jacobian + quadrature.
    # Only intercepts affect g in the current model.
    keys = (
        ("iso_intercept", "ax_intercept")
        if is_zero_rhombicity
        else ("iso_intercept", "ax_intercept", "rho_intercept")
    )

    jac = np.zeros((3, len(keys)), dtype=float)
    sig = np.zeros((len(keys),), dtype=float)

    for i, key in enumerate(keys):
        val = float(params[key])
        err = float(params.get(f"{key}_err", 0.0))

        sig[i] = err

        # If σ is missing, skip contribution (derivative not needed).
        if err <= 0.0:
            continue

        step = err

        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[key] = val + step
        p_minus[key] = val - step

        try:
            g_plus = np.asarray(solver(p_plus), dtype=float)
            g_minus = np.asarray(solver(p_minus), dtype=float)

        except (ValueError, ArithmeticError):
            # If the nonlinear solve fails, fall back to a smaller step.
            step = max(abs(val) * 1e-6, 1e-12)
            p_plus[key] = val + step
            p_minus[key] = val - step
            g_plus = np.asarray(solver(p_plus), dtype=float)
            g_minus = np.asarray(solver(p_minus), dtype=float)

        jac[:, i] = (g_plus - g_minus) / (2.0 * step)

    var = (jac**2) @ (sig**2)
    g_err = np.sqrt(var)

    return method_used, tuple(g0.tolist()), tuple(g_err.tolist())


def solve_D_E(
    params: dict[str, float],
    spin: float,
    g_nominal,
) -> tuple[float | None, float | None]:
    """
    Solve for axial (D) and rhombic (E) ZFS parameters from the ax_slope and rho_slope

    D and E are returned in cm^-1
    """

    ax_slope = params["ax_slope"]
    rho_slope = params["rho_slope"]

    gx_val, gy_val, gz_val = g_nominal

    g2_iso = (gx_val**2 + gy_val**2 + gz_val**2) / 3.0
    g2_ax = 1.5 * (gz_val**2 - g2_iso)
    g2_rh = 0.5 * (gx_val**2 - gy_val**2)

    f_S = (2.0 * spin - 1.0) * (2.0 * spin + 3.0)

    coeff = f_S / (30.0 * K)

    rhs1 = -ax_slope / coeff
    rhs2 = rho_slope / coeff

    A = np.array(
        [
            [g2_ax + 3.0 * g2_iso, -3.0 * g2_rh],
            [g2_rh, g2_ax - 3.0 * g2_iso],
        ],
        dtype=float,
    )

    rhs = np.array([rhs1, rhs2], dtype=float)

    D_J, E_J = np.linalg.solve(A, rhs)

    # Convert to cm^-1
    D = D_J / (H * C * 100)
    E = E_J / (H * C * 100)

    return D, E


def _solve_g_principals_full(params: dict[str, float]) -> tuple[float, float, float]:
    """Solve for principal g-values (gx, gy, gz) using iso/ax/rh fit intercepts.

    This branch uses a numerical solve (sympy.nsolve) for the three principal values
    based on the relationships between g and chiT fit intercepts.

    Args:
        params (dict[str, float]): Flattened fit parameters keyed by
            `{type}_{intercept|slope|intercept_err|slope_err}`.

    Returns:
        tuple[float, float, float]: Principal g-values (gx, gy, gz).
    """

    iso_intercept = params["iso_intercept"]
    ax_intercept = params["ax_intercept"]
    rho_intercept = params["rho_intercept"]

    gx, gy, gz = symbols("gx gy gz", real=True)

    g_iso_fit = iso_intercept / G_E

    g2_iso = (gx**2 + gy**2 + gz**2) / 3.0
    g2_ax = 1.5 * (gz**2 - g2_iso)
    g2_rh = 0.5 * (gx**2 - gy**2)

    eqs = [
        (gx + gy + gz) / 3.0 - g_iso_fit,
        g2_ax - ax_intercept,
        g2_rh - rho_intercept,
    ]

    # Initial guess: nearly isotropic solution around g_iso
    g0 = float(g_iso_fit)
    gx_val, gy_val, gz_val = nsolve(
        eqs, (gx, gy, gz), (g0 - 0.05, g0 + 0.05, g0 + 0.10)
    )

    g_vals = [float(gx_val), float(gy_val), float(gz_val)]
    g_vals.sort()
    return g_vals[0], g_vals[1], g_vals[2]


def _solve_g_principals_axial_only(
    params: dict[str, float],
) -> tuple[float, float, float]:
    """Solve for principal g-values (gx, gy, gz) assuming zero rhombicity.

    This branch assumes gx ≈ gy and uses the analytic solution used by the previous
    axial-only workflow. It is selected when the fitted rhombic intercept is ~0.

    Args:
        params (dict[str, float]): Flattened fit parameters keyed by
            `{type}_{intercept|slope|intercept_err|slope_err}`.

    Returns:
        tuple[float, float, float]: Principal g-values (gx, gy, gz) with gx == gy in
        this approximation.
    """

    iso_intercept = params["iso_intercept"]
    ax_intercept = params["ax_intercept"]

    g_iso = iso_intercept / G_E
    g_sq_ax_val = ax_intercept

    try:
        gx_val = -np.sqrt(g_sq_ax_val / 3.0 + g_iso**2) + 2.0 * g_iso
    except ValueError:
        gx_val = float("nan")

    gz_val = 3.0 * g_iso - 2.0 * gx_val
    gy_val = gx_val

    g_vals = [float(gx_val), float(gy_val), float(gz_val)]
    g_vals.sort()
    return g_vals[0], g_vals[1], g_vals[2]
