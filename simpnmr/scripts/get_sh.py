# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""
Extract g-tensor principal values from Curie-normalised chiT regression fits.

This script reads a chiT regression CSV (with intercept/slope terms for iso/ax/rh)
and solves for principal g-values either using a full (rhombic) numerical solve
or an axial-only analytic approximation.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.constants import physical_constants
from sympy import nsolve, symbols

# Imports
G_E = abs(physical_constants["electron g factor"][0])
MU_B = physical_constants["Bohr magneton"][0]
K = physical_constants["Boltzmann constant"][0]
H = physical_constants["Planck constant"][0]
C = physical_constants["speed of light in vacuum"][0]


def read_chiT_regression_csv(filename: str) -> dict[str, float]:
    """
    Read a Curie-normalised chiT regression CSV file and return fit parameters.

    The CSV is expected to contain columns: `type`, `intercept`, and `slope`. Each row
    is flattened into keys of the form `{type}_intercept` and `{type}_slope`.

    Args:
        filename (str): Path to the regression CSV file.

    Returns:
        dict[str, float]: Flattened fit parameters keyed by `{type}_{intercept|slope}`.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """

    # Read CSV, skipping comment lines
    df = pd.read_csv(filename, comment="#")

    # Sanity check
    required_cols = {"type", "intercept", "slope", "intercept_err", "slope_err"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain {required_cols}")

    params = {}

    for _, row in df.iterrows():
        label = row["type"]
        params[f"{label}_intercept"] = float(row["intercept"])
        params[f"{label}_slope"] = float(row["slope"])
        params[f"{label}_intercept_err"] = float(row["intercept_err"])
        params[f"{label}_slope_err"] = float(row["slope_err"])

    return params


def compute_g_tensor(
    filename: str,
    *,
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
        atol (float): Absolute tolerance used to decide whether rhombicity is ~0.

    Returns:
        tuple[str, tuple[float, float, float], tuple[float, float, float]]:
            (method_used, g_nominal, g_err)
    """

    params = read_chiT_regression_csv(filename)

    rh_intercept = float(params.get("rh_intercept", 0.0))
    method_used = "axial" if np.isclose(rh_intercept, 0.0, atol=atol) else "full"

    solver = (
        _solve_g_principals_axial_only
        if method_used == "axial"
        else _solve_g_principals_full
    )

    # Nominal solution
    g0 = np.asarray(solver(params), dtype=float)

    # Proper 1σ uncertainty via delta-method: finite-difference Jacobian + quadrature.
    # Only intercepts affect g in the current model.
    if method_used == "axial":
        keys = ("iso_intercept", "ax_intercept")
    else:
        keys = ("iso_intercept", "ax_intercept", "rh_intercept")

    jac = np.zeros((3, len(keys)), dtype=float)
    sig = np.zeros((len(keys),), dtype=float)

    for i, key in enumerate(keys):
        val = float(params[key])
        err = float(params.get(f"{key}_intercept_err", 0.0))
        # Backward compat / defensive: allow err to be stored as `<key>_err`.
        if err == 0.0:
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
            # If full nsolve fails for perturbed points, fall back to a smaller step.
            step = max(abs(val) * 1e-6, 1e-12)
            p_plus[key] = val + step
            p_minus[key] = val - step
            g_plus = np.asarray(solver(p_plus), dtype=float)
            g_minus = np.asarray(solver(p_minus), dtype=float)

        jac[:, i] = (g_plus - g_minus) / (2.0 * step)

    var = (jac**2) @ (sig**2)
    g_err = np.sqrt(var)

    return method_used, tuple(g0.tolist()), tuple(g_err.tolist())


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
    rh_intercept = params["rh_intercept"]

    gx, gy, gz = symbols("gx gy gz", real=True)

    g_iso_fit = iso_intercept / G_E

    g2_iso = (gx**2 + gy**2 + gz**2) / 3.0
    g2_ax = 1.5 * (gz**2 - g2_iso)
    g2_rh = 0.5 * (gx**2 - gy**2)

    eqs = [
        (gx + gy + gz) / 3.0 - g_iso_fit,
        g2_ax - ax_intercept,
        g2_rh - rh_intercept,
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


def solve_D_E(
    params: dict[str, float],
    spin: float,
    g_nominal,
) -> tuple[float | None, float | None]:
    """
    Solve for axial (D) and rhombic (E) ZFS parameters from the ax_slope and rh_slope

    D and E are returned in cm^-1
    """

    ax_slope = params["ax_slope"]
    rh_slope = params["rh_slope"]

    gx_val, gy_val, gz_val = g_nominal

    g2_iso = (gx_val**2 + gy_val**2 + gz_val**2) / 3.0
    g2_ax = 1.5 * (gz_val**2 - g2_iso)
    g2_rh = 0.5 * (gx_val**2 - gy_val**2)

    f_S = (2.0 * spin - 1.0) * (2.0 * spin + 3.0)

    coeff = f_S / (30.0 * K)

    rhs1 = -ax_slope / coeff
    rhs2 = rh_slope / coeff

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


def main():
    """
    Parse CLI arguments, select the solve branch, and print the g principal values.

    Returns:
        None
    """

    # Define command-line interface for input file and section choice
    parser = argparse.ArgumentParser(
        description=(
            "Compute principal g-tensor values and zero-field "
            "splitting parameters (D, E)\n"
            "from Curie-normalised chiT regression fits (iso/ax/rh components).\n\n"
            "The script reads a regression CSV file containing "
            "intercepts and slopes for\n"
            "isotropic, axial and rhombic terms, solves for the principal "
            "g-values using\n"
            "either an axial analytic approximation or a full rhombic "
            "numerical solver,\n"
            "and propagates 1σ uncertainties by finite-difference error analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="get_sh csv_file --spin SPIN",
    )

    parser.add_argument("csv_file", help="CSV file containing slope and intercepts")

    parser.add_argument(
        "--spin",
        type=float,
        required=True,
        help="Total spin quantum number (e.g. 2.0)",
    )

    args = parser.parse_args()

    params = read_chiT_regression_csv(args.csv_file)

    method_used, g_nominal, g_err = compute_g_tensor(
        args.csv_file,
        atol=1e-6,
    )

    g_parts = [f"{v:.3f} ± {e:.3f}" for v, e in zip(g_nominal, g_err, strict=True)]

    print(f"gx = {g_parts[0]}")
    print(f"gy = {g_parts[1]}")
    print(f"gz = {g_parts[2]}")

    spin = float(args.spin)

    if spin != 0.5:
        D, E = solve_D_E(params, spin, g_nominal)
        print(f"D = {D:.3f} cm^-1")
        print(f"E = {E:.3f} cm^-1")


if __name__ == "__main__":
    main()
