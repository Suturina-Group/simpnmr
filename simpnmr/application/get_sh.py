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
from sympy import Expr, nsolve, symbols

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
        D, E, D_err, E_err = solve_D_E(params, spin, g_nominal, g_err)

        logger.info("D = %.3f ± %.3f cm^-1", D, D_err)
        logger.info("E = %.3f ± %.3f cm^-1", E, E_err)

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

    # Delta-method: finite-difference Jacobian + quadrature
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

        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            logger.warning("Delta-method: failed for parameter %d", i)
            continue

        jac[:, i] = (g_plus - g_minus) / (2.0 * step)

    g_err = _delta_method_sigma(jac, sig)

    return method_used, tuple(g0.tolist()), tuple(g_err.tolist())


def solve_D_E(
    params: dict[str, float],
    spin: float,
    g_nominal: tuple[float, float, float],
    g_err: tuple[float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Solve for axial (D) and rhombic (E) ZFS parameters with 1σ uncertainties.

    D and E are solved from the fitted Curie-normalised slopes (`ax_slope`,
    `rho_slope`) and the principal g-values.

    Uncertainties are propagated using the delta method (finite-difference
    Jacobian) assuming independent 1σ uncertainties in:
      - gx, gy, gz (typically from intercept uncertainty propagation)
      - ax_slope, rho_slope (from regression slope uncertainties)

    Args:
        params: Flattened regression parameters.
        spin: Total spin S.
        g_nominal: Nominal (gx, gy, gz).
        g_err: Optional 1σ uncertainties for (gx, gy, gz). If not provided,
            g uncertainties are treated as zero.

    Returns:
        (D, E, D_err, E_err) where D/E are in cm^-1 and errors are 1σ.
    """

    ax_slope = float(params["ax_slope"])
    rho_slope = float(params["rho_slope"])

    gx0, gy0, gz0 = (float(g_nominal[0]), float(g_nominal[1]), float(g_nominal[2]))
    D0, E0 = _solve_zfs_from_g_slopes(spin, gx0, gy0, gz0, ax_slope, rho_slope)

    ax_slope_err = float(params.get("ax_slope_err", 0.0))
    rho_slope_err = float(params.get("rho_slope_err", 0.0))

    if g_err is None:
        gx_err = gy_err = gz_err = 0.0
    else:
        gx_err, gy_err, gz_err = (float(g_err[0]), float(g_err[1]), float(g_err[2]))

    sig = np.array([gx_err, gy_err, gz_err, ax_slope_err, rho_slope_err], dtype=float)
    jac = np.zeros((2, 5), dtype=float)
    x0 = np.array([gx0, gy0, gz0, ax_slope, rho_slope], dtype=float)

    # Finite-difference Jacobian for delta-method uncertainty propagation
    for i in range(5):
        if sig[i] <= 0.0:
            continue

        step = sig[i]
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[i] += step
        x_minus[i] -= step

        try:
            D_plus, E_plus = _solve_zfs_from_g_slopes(
                spin, x_plus[0], x_plus[1], x_plus[2], x_plus[3], x_plus[4]
            )
            D_minus, E_minus = _solve_zfs_from_g_slopes(
                spin, x_minus[0], x_minus[1], x_minus[2], x_minus[3], x_minus[4]
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            logger.warning("Delta-method: failed for parameter %d", i)
            continue

        jac[0, i] = (D_plus - D_minus) / (2.0 * step)
        jac[1, i] = (E_plus - E_minus) / (2.0 * step)

    zfs_err = _delta_method_sigma(jac, sig)
    D_err = float(zfs_err[0])
    E_err = float(zfs_err[1])

    return float(D0), float(E0), D_err, E_err


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

    g2_iso, g2_ax, g2_rh = _compute_g2_invariants(gx, gy, gz)

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


def _compute_g2_invariants(
    gx: float | Expr,
    gy: float | Expr,
    gz: float | Expr,
) -> tuple[float | Expr, float | Expr, float | Expr]:
    """Compute g^2 invariants used across g-tensor and ZFS formulas.

    Returns:
        (g2_iso, g2_ax, g2_rh)

    Notes:
        This helper is intentionally compatible with both numeric inputs (floats)
        and SymPy symbols, since `_solve_g_principals_full` uses symbolic equations.
    """

    g2_iso = (gx**2 + gy**2 + gz**2) / 3.0
    g2_ax = 1.5 * (gz**2 - g2_iso)
    g2_rh = 0.5 * (gx**2 - gy**2)

    return g2_iso, g2_ax, g2_rh


def _solve_zfs_from_g_slopes(
    spin: float,
    gx_val: float,
    gy_val: float,
    gz_val: float,
    ax_slope: float,
    rho_slope: float,
) -> tuple[float, float]:
    """Internal: solve D,E in cm^-1 from explicit inputs."""

    g2_iso, g2_ax, g2_rh = _compute_g2_invariants(gx_val, gy_val, gz_val)

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

    # Convert J -> cm^-1
    D = D_J / (H * C * 100)
    E = E_J / (H * C * 100)

    return float(D), float(E)


def _delta_method_sigma(jac: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Return 1σ output uncertainties via delta method."""
    return np.sqrt((jac**2) @ (sig**2))
