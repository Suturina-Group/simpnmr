# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot temperature-dependent chemical shifts.

Loads experimental data and generates shift-vs-temperature and shift-vs-inverse-
temperature plots.
"""

from simpnmr.app.loaders.exp_load import load_experiments
from simpnmr.app.params.options import PlotShiftTdepRunOptions
from simpnmr.viz.plots.shifts import plot_shift_tdep
from simpnmr.viz.style.theme import apply_profile


def run_plot_shift_tdep(
    experiment_files: list[str],
    options: PlotShiftTdepRunOptions,
) -> int:
    experiments = load_experiments(experiment_files)

    # Build the resolved plotting contract once per run.
    spec = apply_profile(options.runtime.plot_profile)

    with spec.context():
        plot_shift_tdep(
            experiments,
            "ShiftT_vs_T",
            spec=spec,
            show=options.runtime.show_plots,
            save=True,
            save_name="shift_x_T_vs_T",
        )

        plot_shift_tdep(
            experiments,
            "Shift_vs_1/T",
            spec=spec,
            show=options.runtime.show_plots,
            save=True,
            save_name="shift_vs_T-1",
        )

    return 0
