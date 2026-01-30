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
        save_name=f"shift_x_T_vs_T{options.runtime.plot_format}",
    )

    plot_shift_tdep(
        experiments,
        "Shift_vs_1/T",
        show=options.show,
        save=options.save,
        save_name=f"shift_vs_T-1{options.runtime.plot_format}",
    )

    return 0
