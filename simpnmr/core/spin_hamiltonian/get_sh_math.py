# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Solve g-tensor and ZFS parameters from fit coefficients.

Provides numerical helpers to compute principal g-values and propagate
uncertainties, with optional ZFS (D, E) estimation.
"""

import logging

import numpy as np
from sympy import Expr, nsolve, symbols

from simpnmr.core.constants.physics import GE, KB, C, H

logger = logging.getLogger(__name__)


def compute_g_tensor_from_params(
    params: dict[str, float],
    *,
    atol: float = 1e-6,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
    """Compute principal g-values and their 1σ uncertainties from fit parameters.

    Error propagation uses the delta method with independent inputs.

    Args:
        params: Regression parameters keyed by
            `{type}_{intercept|intercept_err}`.
        atol: Absolute tolerance to decide whether rhombicity is effectively zero.

    Returns:
        (method_used, g_nominal, g_err)
        where method_used is one of {"axial_only", "full"}.
    """

    rho_val = float(params.get("rho_intercept", 0.0))
    if abs(rho_val) <= atol:
        method_used = "axial_only"
        solver = solve_g_principals_axial_only
        keys = ("iso_intercept", "ax_intercept")
    else:
        method_used = "full"
        solver = solve_g_principals_full
        keys = ("iso_intercept", "ax_intercept", "rho_intercept")

    g0 = np.asarray(solver(params), dtype=float)

    # Build parameter vector and uncertainties
    x0 = np.array([float(params[k]) for k in keys], dtype=float)
    sig = np.array([float(params.get(f"{k}_err", 0.0)) for k in keys], dtype=float)

    # Finite-difference Jacobian for delta-method propagation
    jac = np.zeros((3, len(keys)), dtype=float)

    for i, key in enumerate(keys):
        if sig[i] <= 0.0:
            continue

        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[key] = x0[i] + sig[i]
        p_minus[key] = x0[i] - sig[i]

        try:
            g_plus = np.asarray(solver(p_plus), dtype=float)
            g_minus = np.asarray(solver(p_minus), dtype=float)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            logger.warning(
                "Delta-method: failed to evaluate derivative for %s; "
                "contribution skipped",
                key,
            )
            continue

        jac[:, i] = (g_plus - g_minus) / (2.0 * sig[i])

    g_err = delta_method_sigma(jac, sig)

    return method_used, tuple(g0.tolist()), tuple(g_err.tolist())


def solve_D_E(
    params: dict[str, float],
    spin: float,
    g_nominal: tuple[float, float, float],
    g_err: tuple[float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Solve for ZFS parameters (D, E) and their 1σ uncertainties.

    Uncertainties are propagated via the delta method assuming independent inputs.

    Returns:
        (D, E, D_err, E_err) in cm^-1.
    """

    ax_slope = float(params["ax_slope"])
    rho_slope = float(params["rho_slope"])

    gx0, gy0, gz0 = (
        float(g_nominal[0]),
        float(g_nominal[1]),
        float(g_nominal[2]),
    )

    D0, E0 = solve_zfs_from_g_slopes(spin, gx0, gy0, gz0, ax_slope, rho_slope)

    ax_slope_err = float(params.get("ax_slope_err", 0.0))
    rho_slope_err = float(params.get("rho_slope_err", 0.0))

    if g_err is None:
        gx_err = gy_err = gz_err = 0.0
    else:
        gx_err, gy_err, gz_err = (
            float(g_err[0]),
            float(g_err[1]),
            float(g_err[2]),
        )

    sig = np.array([gx_err, gy_err, gz_err, ax_slope_err, rho_slope_err], dtype=float)
    x0 = np.array([gx0, gy0, gz0, ax_slope, rho_slope], dtype=float)

    # Finite-difference Jacobian for delta-method propagation
    jac = np.zeros((2, 5), dtype=float)

    for i in range(5):
        if sig[i] <= 0.0:
            continue

        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[i] += sig[i]
        x_minus[i] -= sig[i]

        try:
            gx_p, gy_p, gz_p, ax_p, rho_p = x_plus
            gx_m, gy_m, gz_m, ax_m, rho_m = x_minus

            D_plus, E_plus = solve_zfs_from_g_slopes(
                spin, gx_p, gy_p, gz_p, ax_p, rho_p
            )
            D_minus, E_minus = solve_zfs_from_g_slopes(
                spin, gx_m, gy_m, gz_m, ax_m, rho_m
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            logger.warning(
                "Delta-method: failed to evaluate derivative for parameter index %d; "
                "contribution skipped",
                i,
            )
            continue

        jac[0, i] = (D_plus - D_minus) / (2.0 * sig[i])
        jac[1, i] = (E_plus - E_minus) / (2.0 * sig[i])

    zfs_err = delta_method_sigma(jac, sig)

    return float(D0), float(E0), float(zfs_err[0]), float(zfs_err[1])


def solve_g_principals_full(params: dict[str, float]) -> tuple[float, float, float]:
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

    g_iso_fit = iso_intercept / GE

    g2_iso, g2_ax, g2_rh = compute_g2_invariants(gx, gy, gz)

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


def solve_g_principals_axial_only(
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

    g_iso = iso_intercept / GE
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


def compute_g2_invariants(
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


def solve_zfs_from_g_slopes(
    spin: float,
    gx_val: float,
    gy_val: float,
    gz_val: float,
    ax_slope: float,
    rho_slope: float,
) -> tuple[float, float]:
    """Internal: solve D,E in cm^-1 from explicit inputs."""

    g2_iso, g2_ax, g2_rh = compute_g2_invariants(gx_val, gy_val, gz_val)

    f_S = (2.0 * spin - 1.0) * (2.0 * spin + 3.0)
    coeff = f_S / (30.0 * KB)

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


def delta_method_sigma(jac: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Return 1σ output uncertainties via delta method."""
    return np.sqrt((jac**2) @ (sig**2))
