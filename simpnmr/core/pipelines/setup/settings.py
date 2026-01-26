import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from simpnmr.core.pipelines.setup.options import RuntimeSettings


def apply_runtime_settings():
    """Apply global side effects and return RuntimeSettings."""

    mpl.rcParams["savefig.directory"] = ""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    mpl.rcParams.update({"font.size": 14})

    # defaults
    echo_r2 = False
    plot_format = ".png"
    csv_delimiter = ","

    if os.getenv("pnmr_echo_r2", "").lower() == "true":
        echo_r2 = True

    if os.getenv("pnmr_fontname"):
        plt.rcParams["font.family"] = os.getenv("pnmr_fontname")

    if os.getenv("pnmr_plot_format"):
        plot_format = os.getenv("pnmr_plot_format")
        if not plot_format.startswith("."):
            plot_format = "." + plot_format

    if os.getenv("pnmr_csvdelimiter"):
        csv_delimiter = os.getenv("pnmr_csvdelimiter")

    return RuntimeSettings(
        echo_r2=echo_r2,
        plot_format=plot_format,
        csv_delimiter=csv_delimiter,
    )
