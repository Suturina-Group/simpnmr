# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write fitted susceptibility model parameters to text files.

Provides helpers to serialize fitted model parameters and diagnostics
in a readable tabular format.
"""

import logging

from simpnmr.core.fitting import models

logger = logging.getLogger(__name__)


def write_model_data(
    models: list[models.SusceptibilityModel], file_name: str, verbose: bool = True
) -> None:
    """Writes fitted model parameters for multiple temperatures to a text file.

    Assumes all models in `models` are of the same concrete class.

    Args:
        models: Models (typically one per temperature).
        file_name: Output file path.
        verbose: If ``True``, prints the output file path.

    Returns:
        None.
    """
    f = open(file_name, "w", encoding="utf-8")
    f.write(" {:^12} ".format("T"))

    # Fitted parameters
    for name in models[0].fit_vars.keys():
        f.write("{:^17} {:^17} ".format(name, name + "-s-dev"))

    # Fixed parameters
    for name in models[0].fix_vars.keys():
        f.write("{:^17} ".format(name))

    f.write("{:^12} ".format("r2"))
    f.write("{:^12} ".format("r2_adj"))

    f.write("\n")

    for model in models:
        if not model.fit_status:
            continue
        f.write("{:12.10f} ".format(model.temperature))

        for name in model.fit_vars.keys():
            f.write(
                "{: 1.10E} {: 1.10E} ".format(
                    model.final_var_values[name], model.fit_stdev[name]
                )
            )

        for value in model.fix_vars.values():
            f.write("{: 1.10E} ".format(value))

        f.write("{: 1.10E} ".format(model.r2))
        f.write("{: 1.10E} ".format(model.adj_r2))

        f.write("\n")

    if verbose:
        logger.info("Susceptibility Model parameters written to %s", file_name)
    return
