# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot temperature-dependent chemical shifts.

Loads experimental data and generates shift-vs-temperature and shift-vs-inverse-
temperature plots.
"""

from simpnmr.application.loaders.experiment import load_experiments
from simpnmr.application.setup.options import PlotShiftTdepRunOptions
from simpnmr.viz.plots.shifts import plot_shift_tdep


def run_plot_shift_tdep(
    experiment_files: list[str],
    options: PlotShiftTdepRunOptions,
) -> int:
    experiments = load_experiments(experiment_files)

    plot_shift_tdep(
        experiments,
        "ShiftT_vs_T",
        show=options.show,
        save=options.save,
        save_name="shift_x_T_vs_T",
    )

    plot_shift_tdep(
        experiments,
        "Shift_vs_1/T",
        show=options.show,
        save=options.save,
        save_name="shift_vs_T-1",
    )

    return 0
