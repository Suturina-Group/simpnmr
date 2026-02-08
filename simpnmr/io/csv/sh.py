"""CSV IO helpers for Spin Hamiltonian (SH) parameter results."""

from __future__ import annotations

import datetime
import logging

import pandas as pd

from simpnmr.__version__ import __version__

logger = logging.getLogger(__name__)


def write_sh_results_csv(
    file_name: str,
    solver_method: str,
    spin: float,
    g_nominal: tuple[float, float, float] | list[float],
    g_err: tuple[float, float, float] | list[float],
    D: float | None,
    E: float | None,
    D_err: float | None,
    E_err: float | None,
    delimiter: str = ",",
    comment: str = "",
    verbose: bool = True,
) -> None:
    """Write computed SH parameters to a single-row CSV.

    The output is a single header row followed by a single data row.

    Args:
        file_name: Path to the output CSV file.
        solver_method: Human-readable solver method label (e.g., "analytic", "numeric").
        spin: Spin quantum number S.
        g_nominal: Nominal principal g-values in the order (g_x, g_y, g_z).
        g_err: Propagated 1σ standard uncertainties of principal g-values obtained
            via first-order (delta-method) uncertainty propagation assuming
            independent input parameters.
        D: Axial ZFS parameter in cm^-1 (None for S = 1/2).
        E: Rhombic ZFS parameter in cm^-1 (None for S = 1/2).
        D_err: Propagated 1σ standard uncertainty of D in cm^-1 (None for S = 1/2).
        E_err: Propagated 1σ standard uncertainty of E in cm^-1 (None for S = 1/2).
        delimiter: CSV delimiter.
        comment: Optional comment written as CSV header (prefixed with '#').
        verbose: Emit an info-level log message when the file is written.
    """

    if len(g_nominal) != 3:
        raise ValueError("g_nominal must contain exactly 3 values: (g_x, g_y, g_z).")
    if len(g_err) != 3:
        raise ValueError(
            "g_err must contain exactly 3 values: (g_x_err, g_y_err, g_z_err)."
        )

    out: dict[str, str | float | None] = {
        "solver_method": solver_method,
        "spin": float(spin),
        "g_x": float(g_nominal[0]),
        "g_y": float(g_nominal[1]),
        "g_z": float(g_nominal[2]),
        "g_x_err": float(g_err[0]),
        "g_y_err": float(g_err[1]),
        "g_z_err": float(g_err[2]),
        "D_cm^-1": None if D is None else float(D),
        "E_cm^-1": None if E is None else float(E),
        "D_err_cm^-1": None if D_err is None else float(D_err),
        "E_err_cm^-1": None if E_err is None else float(E_err),
    }

    df = pd.DataFrame([out])

    _comment = (
        f"# This file was generated with SimpNMR v{__version__} "
        f"at {datetime.datetime.now():%H:%M:%S %d-%m-%Y}\n"
    )

    if len(comment):
        if comment[0] != "#":
            comment = f"#{comment}"
        _comment += f"{comment}\n"

    with open(file_name, "w") as _f:
        _f.write(_comment)
        df.to_csv(_f, sep=delimiter, header=True, float_format="%.6f", index=None)

    if verbose:
        logger.info("Spin Hamiltonian results written to %s", file_name)

    return
