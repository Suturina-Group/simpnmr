from simpnmr.core import main
from simpnmr.core.pipelines.setup.options import PlotShiftTdepRunOptions
from simpnmr.viz import visualise as vis


def run_plot_shift_tdep(
    experiment_files: list[str],
    options: PlotShiftTdepRunOptions,
) -> int:
    experiments = main.Experiment.from_file(experiment_files)

    vis.plot_shift_tdep(
        experiments,
        "ShiftT_vs_T",
        show=options.show,
        save=options.save,
        save_name=f"shift_x_T_vs_T{options.runtime.plot_format}",
    )

    vis.plot_shift_tdep(
        experiments,
        "Shift_vs_1/T",
        show=options.show,
        save=options.save,
        save_name=f"shift_vs_T-1{options.runtime.plot_format}",
    )

    return 0
