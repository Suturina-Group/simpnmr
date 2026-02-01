# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute g-tensor parameters from chiT regression results.

Reads regression parameters, solves for principal g-values, and optionally
derives ZFS parameters for S > 1/2.
"""

import logging

from simpnmr.core.spin_hamiltonian import get_sh_math
from simpnmr.io.csv.fitting import read_chiT_regression_csv

logger = logging.getLogger(__name__)


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

    method_used, g_nominal, g_err = get_sh_math.compute_g_tensor_from_params(
        params,
        atol=1e-6,
    )

    method_label = (
        "Analytic (zero rhombicity)"
        if method_used == "axial_only"
        else "Nonlinear solve (non-zero rhombicity)"
    )
    logger.info("g-tensor solver method: %s", method_label)

    g_parts = [f"{v:.3f} ± {e:.3f}" for v, e in zip(g_nominal, g_err, strict=True)]

    logger.info("g_x = %s", g_parts[0])
    logger.info("g_y = %s", g_parts[1])
    logger.info("g_z = %s", g_parts[2])

    spin = float(options.spin)

    if spin != 0.5:
        D, E, D_err, E_err = get_sh_math.solve_D_E(params, spin, g_nominal, g_err)

        logger.info("D = %.3f ± %.3f cm^-1", D, D_err)
        logger.info("E = %.3f ± %.3f cm^-1", E, E_err)

    else:
        logger.info(
            "ZFS parameters (D, E) are not defined for S = 1/2 "
            "and are therefore not reported."
        )

    return 0
