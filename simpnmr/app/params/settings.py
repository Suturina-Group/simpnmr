# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Apply global runtime settings.

Configures environment variables and matplotlib defaults and returns resolved
runtime settings.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from simpnmr.app.params.options import RuntimeSettings


def apply_runtime_settings(
    plot_profile: str = "paper",
    show_plots: bool = False,
) -> RuntimeSettings:
    """Apply global side effects and return RuntimeSettings."""

    mpl.rcParams["savefig.directory"] = ""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # defaults
    echo_r2 = False
    csv_delimiter = ","

    if os.getenv("pnmr_echo_r2", "").lower() == "true":
        echo_r2 = True

    if os.getenv("pnmr_fontname"):
        plt.rcParams["font.family"] = os.getenv("pnmr_fontname")

    if os.getenv("pnmr_csvdelimiter"):
        csv_delimiter = os.getenv("pnmr_csvdelimiter")

    return RuntimeSettings(
        echo_r2=echo_r2,
        csv_delimiter=csv_delimiter,
        plot_profile=plot_profile,
        show_plots=show_plots,
    )
